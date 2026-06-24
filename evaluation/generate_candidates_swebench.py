# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Multi-instance candidate generation (diversity sampling) on SWE-bench / Docker.

This is the batch counterpart of `patch_selection/generate_candidates.py`. It reuses
the Docker-per-instance machinery of `run_evaluation.BenchmarkEvaluation` (image pull,
trae-agent build/inject, problem_statement export) but, instead of running the coder
agent **once** per instance, it runs it **serially N times** inside the same container
and collects up to N non-empty candidate patches.

It reproduces the paper's patch-generation component (Section 3.2):
  * multiple serial runs of the same coder agent;
  * terminate once N candidate patches are collected;
  * (no multi-model / Mixture — single model, as currently required).

Temperature (diversity) is NOT set here: it is a per-request sampling parameter the
trae client reads from your trae_config.yaml and sends to the backend (e.g. vLLM via
provider: openai + base_url). Make sure that config declares temperature > 0.

Output: a single JSONL file in the exact format consumed by `patch_selection/selector.py`:
    {"instance_id", "issue", "patches": [...N...], "success_id": [...], "regressions": [...]}
`success_id` is a placeholder (filled later by the SWE-bench harness); `regressions`
defaults to empty lists ("passed all"), to be filled by your regression step if any.

Run from the repo root as a module so the package-relative imports resolve:
    uv run python -m evaluation.generate_candidates_swebench \
        --benchmark SWE-bench --dataset SWE-bench_Verified \
        --config-file trae_config.yaml --num-candidate 10 \
        --output candidates.jsonl --max_workers 4
"""

import argparse
import json
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from .run_evaluation import BenchmarkEvaluation
from .utils import docker_exec


class CandidateGeneration(BenchmarkEvaluation):
    """Generate N candidate patches per instance, serially, inside one container."""

    def generate_for_instance(
        self, instance_id: str, num_candidate: int, max_attempts: int, max_steps: int,
        base_url: str | None = None,
    ) -> dict | None:
        """Run the coder agent serially until N non-empty patches are collected.

        Returns a selector-format candidate entry, or None if the instance is unknown.
        """
        instance = next(
            (inst for inst in self.dataset if inst["instance_id"] == instance_id), None
        )
        if instance is None:
            print(f"Instance {instance_id} not found.")
            return None

        repo_dir = self.config.working_dir(instance_id)  # e.g. "/testbed/" for SWE-bench
        container = self.prepare_experiment_container(instance)

        patches: list[str] = []
        attempt = 0
        try:
            while len(patches) < num_candidate and attempt < max_attempts:
                attempt += 1
                # Reset the repo to its pristine base commit so each run is independent.
                reset = f"cd {repo_dir} && git reset --hard HEAD && git clean -fdx"
                docker_exec(container, f"/bin/bash -c '{reset}'")

                patch_file = f"/instance-data/{instance_id}_cand{attempt}.patch"
                traj_file = f"/instance-data/{instance_id}_cand{attempt}.json"
                base_url_opt = f" --model-base-url {base_url}" if base_url else ""
                command = (
                    f"source trae-agent/.venv/bin/activate && "
                    f"trae-cli run --file /instance-data/problem_statement.txt "
                    f'--working-dir="{repo_dir}" '
                    f"--config-file trae_config.yaml --must-patch "
                    f"--patch-path {patch_file} --trajectory-file {traj_file} "
                    f"--max-steps {max_steps}{base_url_opt}"
                )
                try:
                    return_code, output = docker_exec(container, f"/bin/bash -c '{command}'")
                    if return_code is not None and return_code != 0:
                        print(f"[{instance_id}] attempt {attempt}: trae-cli rc={return_code}")
                except Exception:
                    print(f"[{instance_id}] attempt {attempt} failed.")
                    print(traceback.format_exc())
                    continue

                # The container mounts results/<task_id>/<instance_id> at /instance-data.
                host_patch = (
                    self.task_results_dir / instance_id / f"{instance_id}_cand{attempt}.patch"
                )
                patch_text = host_patch.read_text() if host_patch.exists() else ""
                if patch_text.strip():
                    patches.append(patch_text)
                    print(f"[{instance_id}] collected {len(patches)}/{num_candidate} "
                          f"(attempt {attempt}/{max_attempts})")
                else:
                    print(f"[{instance_id}] attempt {attempt}: empty patch")
        finally:
            container.stop()
            container.remove()

        if len(patches) < num_candidate:
            print(f"[{instance_id}] WARNING: only {len(patches)}/{num_candidate} after "
                  f"{attempt} attempts")

        return {
            "instance_id": instance_id,
            "issue": instance.get("problem_statement", ""),
            "patches": patches,
            "success_id": [-1] * len(patches),      # -1 = unknown (live selection); offline eval fills 0/1
            "regressions": [[] for _ in patches],   # empty = passed all regression tests
        }

    def generate_all(
        self,
        output_path: str,
        num_candidate: int,
        max_attempts: int,
        max_steps: int,
        max_workers: int,
        instance_ids: list[str] | None = None,
        base_urls: list[str] | None = None,
    ) -> None:
        """Generate candidates for every instance and append them to `output_path`.

        Resumable: instances already present in `output_path` with >= num_candidate
        patches are skipped. If `base_urls` is given, instances are spread across those
        vLLM endpoints round-robin (same model, multiple instances, for throughput).
        """
        if instance_ids is None:
            instance_ids = [inst["instance_id"] for inst in self.dataset]

        # Resume: skip instances already completed in the output file.
        done: set[str] = set()
        try:
            with open(output_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    if len(rec.get("patches", [])) >= num_candidate:
                        done.add(rec["instance_id"])
        except FileNotFoundError:
            pass
        todo = [iid for iid in instance_ids if iid not in done]
        if done:
            print(f"Resuming: {len(done)} instances already complete, {len(todo)} to do.")

        write_lock = threading.Lock()

        def _work(iid: str, base_url: str | None):
            entry = self.generate_for_instance(
                iid, num_candidate, max_attempts, max_steps, base_url=base_url
            )
            if entry is None:
                return iid
            with write_lock, open(output_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
            return iid

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _work, iid, base_urls[i % len(base_urls)] if base_urls else None
                ): iid
                for i, iid in enumerate(todo)
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
                iid = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Instance {iid} failed: {e}")
        print(f"Done. Candidates written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", type=str, default="SWE-bench", help="Benchmark name.")
    parser.add_argument("--dataset", type=str, default="SWE-bench_Verified", help="Dataset name.")
    parser.add_argument("--working-dir", type=str, default="./trae-workspace",
                        help="Host workspace dir (build artifacts, configs).")
    parser.add_argument("--config-file", type=str, default="trae_config.yaml",
                        help="Trae config (point at your vLLM endpoint; temperature > 0 there).")
    parser.add_argument("--docker-env-config", type=str, default="", required=False,
                        help="Docker env config file.")
    parser.add_argument("--instance_ids", nargs="+", type=str,
                        help="Subset of instance IDs to run (space separated).")
    parser.add_argument("--run-id", type=str, default="trae-agent-gen",
                        help="Run ID (namespaces the results dir).")
    parser.add_argument("--num-candidate", "-N", type=int, required=True,
                        help="Ensemble size N: collect N candidate patches per instance.")
    parser.add_argument("--max-attempts", type=int, default=None,
                        help="Cap runs per instance (default: 3*N).")
    parser.add_argument("--max-steps", type=int, default=200, help="Max agent steps per run.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL (selector candidate format).")
    parser.add_argument("--max_workers", type=int, default=4,
                        help="Parallel workers ACROSS instances (each instance is serial inside).")
    parser.add_argument("--base-urls", type=str, default=None,
                        help="Comma-separated vLLM base URLs to spread instances across "
                             "round-robin (e.g. http://172.17.0.1:8000/v1,...:8001/v1). "
                             "Overrides the config's base_url per run.")
    args = parser.parse_args()

    base_urls = [u.strip() for u in args.base_urls.split(",")] if args.base_urls else None
    max_attempts = args.max_attempts or (3 * args.num_candidate)

    gen = CandidateGeneration(
        args.benchmark,
        args.working_dir,
        args.config_file,
        args.dataset,
        args.docker_env_config,
        "",  # benchmark_harness_path: not needed for generation
        args.run_id,
        args.max_workers,
        args.instance_ids,
    )
    gen.prepare_trae_agent()
    gen.generate_all(
        output_path=args.output,
        num_candidate=args.num_candidate,
        max_attempts=max_attempts,
        max_steps=args.max_steps,
        max_workers=args.max_workers,
        instance_ids=args.instance_ids,
        base_urls=base_urls,
    )


if __name__ == "__main__":
    main()
