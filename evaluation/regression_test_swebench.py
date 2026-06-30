# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Regression-testing (patch pruning) for SWE-bench candidate patches.

This implements the *tester* of the Trae Agent paper's patch-pruning component
(Section 3.3, "Perform regression testing"), faithfully (path A — no peeking at the
gold FAIL_TO_PASS/PASS_TO_PASS labels by default):

  1. Find passing tests: run the repo's test suite on the pristine codebase and keep
     the tests that PASS. (No LLM.)
  2. Select regression tests: ask the LLM to keep only the subset that a correct fix
     should NOT change (a correct issue resolution may legitimately alter some
     behaviour, so blindly requiring every old test to pass would discard good
     patches). (LLM — via your trae_config.yaml model, e.g. vLLM.)
  3. Validate each candidate: apply the patch, run the selected regression tests, and
     record which fail. (No LLM.)

Per Figure 2 (Patch Pruning = deduplication THEN regression testing), candidates are
deduplicated (via the selector's `clean_patch` normalization) BEFORE step 3, so the
expensive regression tests run once per unique patch and duplicates inherit their
representative's result. (Note: trae's own selector applies these two filters in the
reverse order, but on precomputed data, so it is functionally equivalent there.)

It reuses:
  * the SWE-bench instance Docker images + container plumbing of
    `run_evaluation.BenchmarkEvaluation`;
  * the `swebench` package for per-repo test commands (`MAP_REPO_VERSION_TO_SPECS`)
    and per-repo log parsers (`MAP_REPO_TO_PARSER`) — the messy, repo-specific bits.

Input/Output: it reads the candidate JSONL produced by
`generate_candidates_swebench.py` and writes the SAME format back, but with the
`regressions` field filled per patch:
    regressions[i] == []            -> patch i passed all selected regression tests
    regressions[i] == [test, ...]   -> patch i failed these (selector prunes it)

SWE-bench only (the standard / most common setting). Run from the repo root:
    uv run python -m evaluation.regression_test_swebench \
        --dataset SWE-bench_Verified \
        --candidates candidates.jsonl --output candidates_pruned.jsonl \
        --config-file trae_config.yaml --model-name trae_agent_model --max_workers 4
"""

import argparse
import io
import json
import os
import re
import tarfile
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS, TestStatus
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
from tqdm import tqdm

from trae_agent.utils.config import Config
from trae_agent.utils.llm_clients.llm_basics import LLMMessage
from trae_agent.utils.llm_clients.llm_client import LLMClient
from trae_agent.utils.trajectory_recorder import TrajectoryRecorder

from .patch_selection.trae_selector.utils import clean_patch
from .run_evaluation import BenchmarkEvaluation

# Standard activation prefix for SWE-bench instance images (conda env "testbed",
# repo checked out at /testbed). Overridable via --eval-prefix.
DEFAULT_EVAL_PREFIX = "source /opt/miniconda3/bin/activate testbed && cd /testbed"


def _record_container_setup(stage: str, instance_id: str, t0: float, t1: float,
                            kind: str = "container_setup") -> None:
    """Append a container-setup timing record (epoch wall) to $STEP3_CONTAINER_LOG
    so step3's timeline can show docker container startup as its own lane. No-op
    when the env var is unset (normal non-profiled runs)."""
    import os
    path = os.environ.get("STEP3_CONTAINER_LOG")
    if not path:
        return
    try:
        with open(path, "a") as f:
            f.write(json.dumps({"ts_start": round(t0, 3), "ts_end": round(t1, 3),
                                "wall_s": round(t1 - t0, 3), "stage": stage,
                                "instance_id": instance_id,
                                "kind": kind}) + "\n")
    except OSError:
        pass

SELECT_PROMPT = """You are selecting regression tests for a software issue.

You are given a GitHub issue and a list of tests that currently PASS in the repository.
Identify the subset of these tests that should STILL pass after a correct fix for the
issue — i.e. tests whose behaviour a correct resolution is NOT expected to change.
Exclude tests that the fix may legitimately alter (e.g. tests asserting the very buggy
behaviour described in the issue).

Return ONLY a JSON array of test names, each chosen verbatim from the provided list.

## Issue
{issue}

## Passing tests (choose from these only)
{tests}
"""


class RegressionTester(BenchmarkEvaluation):
    """Fill the `regressions` field of candidate patches via regression testing."""

    def __init__(self, *args, model_name: str, eval_prefix: str, universe: str,
                 max_tests_for_llm: int, test_timeout: int = 300,
                 trajectory_dir: str | None = None, test_parallel: int | None = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_prefix = eval_prefix
        self.universe = universe
        self.max_tests_for_llm = max_tests_for_llm
        self.test_timeout = test_timeout
        # Override the runner's process count for test_cmds that support it (django's
        # runtests.py `--parallel N`). The SWE-bench spec hardcodes `--parallel 1`,
        # which on a many-core host makes the whole-suite find_passing run ~Ncpu times
        # slower than it needs to be. None -> leave the spec's test_cmd untouched.
        self.test_parallel = test_parallel
        # Per-instance trajectory files, so the regression LLM call lands in the same
        # `llm_interactions[]` turn format as the generate/selector stages (normalized).
        self.trajectory_dir = trajectory_dir

        config = Config.create(config_file=self.trae_config_file_name)
        if not config.models or model_name not in config.models:
            raise ValueError(f"Model {model_name} not found in {self.trae_config_file_name}")
        self.model_config = config.models[model_name]
        self.model_config.resolve_config_values()
        self.llm_client = LLMClient(self.model_config)

    # --- container helpers ---------------------------------------------------

    def _start_container(self, instance_id: str):
        return self.docker_client.containers.run(
            self._image_name(instance_id),
            command="/bin/bash",
            detach=True, tty=True, stdin_open=True,
        )

    def _bash(self, container, command: str):
        # Pass argv as a LIST so docker-py sends it as-is instead of shlex-splitting a
        # `bash -c '<command>'` STRING. The regression command embeds the full test_cmd
        # plus up to 300 test ids; quotes/apostrophes inside it break the single-quote
        # wrap (shlex raises "No closing quotation"). A list sidesteps the split entirely.
        rc, out = container.exec_run(cmd=["/bin/bash", "-c", command])
        return rc, out.decode("utf-8")

    def _reset_repo(self, container):
        self._bash(container, "cd /testbed && git checkout -- . && git clean -fdq")

    def _apply_patch(self, container, patch_text: str) -> bool:
        # Copy the patch into the container, then try git apply with patch(1) fallback.
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            data = patch_text.encode()
            info = tarfile.TarInfo(name="candidate.patch")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        tar_stream.seek(0)
        container.put_archive("/tmp", tar_stream.getvalue())
        rc, _ = self._bash(
            container,
            "cd /testbed && (git apply -v /tmp/candidate.patch || "
            "patch --batch --fuzz=5 -p1 -i /tmp/candidate.patch)",
        )
        return rc == 0

    def _container_cpu_usec(self, container) -> int | None:
        """Total CPU-microseconds burned by every process in the container, from its
        cgroup. Host-side, no docker call. exec_run wall-clock can't attribute this
        (the work runs in the container's cgroup, not this host process). Handles both
        cgroup v2 (cpu.stat usage_usec, µs) and v1 (cpuacct.usage, ns)."""
        import glob
        cid = container.id
        # cgroup v2: cpu.stat usage_usec (microseconds)
        v2 = [f"/sys/fs/cgroup/system.slice/docker-{cid}.scope/cpu.stat"]
        v2 += glob.glob(f"/sys/fs/cgroup/**/docker-{cid}.scope/cpu.stat", recursive=True)
        for path in v2:
            try:
                for line in open(path):
                    if line.startswith("usage_usec"):
                        return int(line.split()[1])
            except OSError:
                continue
        # cgroup v1: cpuacct.usage (nanoseconds) -> microseconds
        v1 = [f"/sys/fs/cgroup/cpu,cpuacct/docker/{cid}/cpuacct.usage",
              f"/sys/fs/cgroup/cpuacct/docker/{cid}/cpuacct.usage"]
        v1 += glob.glob(f"/sys/fs/cgroup/cpu,cpuacct/**/docker-{cid}.scope/cpuacct.usage",
                        recursive=True)
        v1 += glob.glob(f"/sys/fs/cgroup/cpu,cpuacct/**/{cid}/cpuacct.usage", recursive=True)
        for path in v1:
            try:
                return int(open(path).read().strip()) // 1000
            except (OSError, ValueError):
                continue
        return None

    def _run_tests(self, container, test_cmd: str, test_ids: list[str] | None,
                   instance_id: str = "", phase: str = "test") -> str:
        ids = (" " + " ".join(test_ids)) if test_ids else ""
        # Wrap in `timeout` so a hanging test (e.g. requests network tests with no
        # connectivity) can't block the whole stage indefinitely. On timeout the run is
        # killed; tests with no PASSED line are then treated as failed (conservative).
        # `env` so a test_cmd with a leading VAR=val (e.g. sympy's
        # "PYTHONWARNINGS='...' bin/test") works: without it `timeout` treats the
        # assignment as the program name and dies with rc 127 in <1s (0 passing).
        cmd = f"{self.eval_prefix} && timeout {self.test_timeout} env {test_cmd}{ids}"
        # Measure the test-suite run: wall-clock + container CPU-seconds (cgroup delta
        # around the exec). Excludes container startup (already running) and the
        # reset/apply done in separate exec calls — just this test run + its eval_prefix.
        import time
        t0 = time.time()
        c0 = self._container_cpu_usec(container)
        _, out = self._bash(container, cmd)
        t1 = time.time()
        c1 = self._container_cpu_usec(container)
        self._record_test_run(instance_id, phase, t0, t1, c0, c1, test_ids)
        return out

    def _record_test_run(self, instance_id, phase, t0, t1, c0, c1, test_ids) -> None:
        """Append one test-run timing record (epoch ts + wall + container CPU-seconds)
        to <trajectory_dir>/<instance_id>_test_runs.jsonl, on the SAME clock as step3."""
        if not self.trajectory_dir:
            return
        cpu_s = (c1 - c0) / 1e6 if (c0 is not None and c1 is not None) else None
        rec = {"ts_start": round(t0, 3), "ts_end": round(t1, 3),
               "wall_s": round(t1 - t0, 3),
               "cpu_s": round(cpu_s, 3) if cpu_s is not None else None,
               "phase": phase, "instance_id": instance_id,
               "n_tests": len(test_ids) if test_ids else None}
        try:
            os.makedirs(self.trajectory_dir, exist_ok=True)
            path = os.path.join(self.trajectory_dir, f"{instance_id}_test_runs.jsonl")
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError as e:
            print(f"[regr] {instance_id}: failed to record test run: {e}")

    def _parse(self, repo: str, version: str, instance_id: str, log: str) -> dict[str, str]:
        parser = MAP_REPO_TO_PARSER[repo]
        stub = SimpleNamespace(repo=repo, version=version, instance_id=instance_id)
        try:
            return parser(log, stub)
        except Exception:
            return {}

    # --- pipeline steps ------------------------------------------------------

    def _test_cmd(self, repo: str, version: str) -> str:
        cmd = MAP_REPO_VERSION_TO_SPECS[repo][version]["test_cmd"]
        # Bump parallelism if requested AND the runner supports it. Only rewrite an
        # existing `--parallel N` token (django); never inject the flag elsewhere
        # (pytest/sympy/etc. don't understand it), so this is a no-op for those repos.
        if self.test_parallel and "--parallel" in cmd:
            cmd = re.sub(r"--parallel\s+\d+", f"--parallel {self.test_parallel}", cmd)
        return cmd

    def find_passing_tests(self, container, repo, version, instance_id, instance) -> list[str]:
        """Step 1: tests that pass in the pristine repo."""
        if self.universe == "pass_to_pass":
            # Fast (less fair): trust SWE-bench's curated pass-to-pass set as the universe.
            return json.loads(instance.get("PASS_TO_PASS", "[]"))
        # Faithful: run the whole suite (bare test_cmd runs everything) and keep passers.
        self._reset_repo(container)
        log = self._run_tests(container, self._test_cmd(repo, version), None,
                              instance_id, "find_passing")
        status = self._parse(repo, version, instance_id, log)
        return [t for t, s in status.items() if s == TestStatus.PASSED.value]

    def select_regression_tests(self, issue: str, passing: list[str],
                                instance_id: str = "") -> list[str]:
        """Step 2: LLM keeps the subset that a correct fix should not change."""
        if not passing:
            return []
        shown = passing
        if len(shown) > self.max_tests_for_llm:
            print(f"[regr] {len(shown)} passing tests > cap {self.max_tests_for_llm}; truncating "
                  f"for the LLM prompt (rest are dropped from the regression set)")
            shown = shown[: self.max_tests_for_llm]
        prompt = SELECT_PROMPT.format(issue=issue, tests="\n".join(shown))
        messages = [LLMMessage(role="user", content=prompt)]
        resp = self.llm_client.chat(messages, self.model_config, None, reuse_history=False)
        self._record_turn(instance_id, messages, resp)
        selected = self._extract_json_list(resp.content)
        passing_set = set(shown)
        # Keep only valid picks; if the LLM returns nothing usable, fall back to all shown.
        chosen = [t for t in selected if t in passing_set]
        return chosen or shown

    def _record_turn(self, instance_id: str, messages, response) -> None:
        """Record the single regression LLM call as one turn in `llm_interactions[]`,
        matching the generate/selector trajectory format. One file per instance.

        A fresh recorder per call keeps this thread-safe across the instance-level
        ThreadPoolExecutor (the shared self.llm_client has no recorder attached)."""
        if not self.trajectory_dir:
            return
        try:
            provider = self.model_config.model_provider.provider
            path = Path(self.trajectory_dir) / f"{instance_id or 'unknown'}.json"
            recorder = TrajectoryRecorder(str(path))
            recorder.start_recording(
                task=f"regression test selection: {instance_id}",
                provider=provider,
                model=self.model_config.model,
                max_steps=1,
            )
            recorder.record_llm_interaction(
                messages, response, provider, self.model_config.model, None
            )
            recorder.finalize_recording(True, response.content)
        except Exception as e:
            print(f"[regr] {instance_id}: failed to record trajectory: {e}")

    @staticmethod
    def _extract_json_list(text: str) -> list[str]:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
            return [str(x) for x in data] if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []

    def validate_candidate(self, container, repo, version, instance_id, regression_tests,
                           patch_text) -> list[str]:
        """Step 3: failed regression tests for one candidate patch."""
        if not regression_tests:
            return []
        self._reset_repo(container)
        if not self._apply_patch(container, patch_text):
            # Unappliable patch -> treat as failing everything (will be pruned).
            return list(regression_tests)
        log = self._run_tests(container, self._test_cmd(repo, version), regression_tests,
                              instance_id, "validate")
        status = self._parse(repo, version, instance_id, log)
        failed = []
        for t in regression_tests:
            # A regression test must PASS; anything else (fail/error/missing) is a failure.
            if status.get(t) != TestStatus.PASSED.value:
                failed.append(t)
        return failed

    def process_instance(self, entry: dict) -> dict:
        instance_id = entry["instance_id"]
        instance = next((i for i in self.dataset if i["instance_id"] == instance_id), None)
        if instance is None:
            print(f"[regr] {instance_id}: not in dataset, skipping")
            return entry
        repo, version = instance["repo"], instance["version"]
        if repo not in MAP_REPO_VERSION_TO_SPECS or repo not in MAP_REPO_TO_PARSER:
            print(f"[regr] {instance_id}: repo {repo} unsupported by swebench, skipping")
            return entry

        import time as _time
        _c0 = _time.time()
        container = self._start_container(instance_id)
        _record_container_setup("prune", instance_id, _c0, _time.time())
        try:
            passing = self.find_passing_tests(container, repo, version, instance_id, instance)
            regression = self.select_regression_tests(entry.get("issue", ""), passing, instance_id)

            # Figure 2 order: deduplicate BEFORE regression testing, so the expensive
            # tests run once per unique patch (duplicates share their representative's
            # result). Uses the same clean_patch normalization as the selector.
            patches = entry["patches"]
            keys = []
            for i, p in enumerate(patches):
                try:
                    keys.append(clean_patch(p))
                except Exception:
                    keys.append(f"__unparsable_{i}__")  # treat as its own class
            rep_of: dict[str, int] = {}
            for i, k in enumerate(keys):
                rep_of.setdefault(k, i)
            unique_reps = sorted(set(rep_of.values()))
            print(f"[regr] {instance_id}: {len(passing)} passing -> {len(regression)} regression "
                  f"tests; {len(patches)} candidates -> {len(unique_reps)} unique (dedup first)")

            rep_failed = {}
            for i in unique_reps:
                failed = self.validate_candidate(
                    container, repo, version, instance_id, regression, patches[i]
                )
                rep_failed[i] = failed
                print(f"[regr]   {instance_id} cand {i}: "
                      f"{'PASS' if not failed else f'{len(failed)} failed'}")

            # Propagate each representative's result to its duplicates.
            entry["regressions"] = [rep_failed[rep_of[keys[i]]] for i in range(len(patches))]
            entry["regression_done"] = True
        finally:
            # SIGKILL + remove in one; skip docker stop's 10s SIGTERM grace (the
            # ephemeral container's PID 1 ignores SIGTERM). Saves ~10s per instance.
            _td0 = _time.time()
            container.remove(force=True)
            _record_container_setup("prune", instance_id, _td0, _time.time(),
                                    kind="container_teardown")
        return entry

    def run(self, candidates_path: str, output_path: str, max_workers: int):
        with open(candidates_path) as f:
            entries = [json.loads(line) for line in f if line.strip()]

        # Resume: skip entries already processed in the output file.
        done = set()
        try:
            with open(output_path) as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        if rec.get("regression_done"):
                            done.add(rec["instance_id"])
        except FileNotFoundError:
            pass
        todo = [e for e in entries if e["instance_id"] not in done]
        if done:
            print(f"Resuming: {len(done)} done, {len(todo)} to do.")

        write_lock = threading.Lock()

        def _work(entry):
            try:
                result = self.process_instance(entry)
            except Exception:
                print(f"[regr] {entry['instance_id']} failed:\n{traceback.format_exc()}")
                result = entry
            with write_lock, open(output_path, "a") as f:
                f.write(json.dumps(result) + "\n")
            return entry["instance_id"]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_work, e): e["instance_id"] for e in todo}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Regression testing"):
                fut.result()
        print(f"Done. Pruned candidates written to {output_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="SWE-bench_Verified",
                   help="SWE-bench dataset name (default: SWE-bench_Verified).")
    p.add_argument("--candidates", required=True, help="Input candidate JSONL (from generation).")
    p.add_argument("--output", required=True, help="Output JSONL with `regressions` filled.")
    p.add_argument("--config-file", default="trae_config.yaml",
                   help="trae config (for the LLM selector model; e.g. vLLM).")
    p.add_argument("--model-name", default="trae_agent_model",
                   help="Model key in the config used for regression-test selection.")
    p.add_argument("--working-dir", default="./trae-workspace", help="Host workspace dir.")
    p.add_argument("--universe", choices=["suite", "pass_to_pass"], default="suite",
                   help="Source of the passing-test universe. 'suite' (default, faithful) runs "
                        "the whole suite; 'pass_to_pass' trusts SWE-bench's curated set (faster, "
                        "less fair — leaks the gold test universe).")
    p.add_argument("--max-tests-for-llm", type=int, default=300,
                   help="Cap on passing tests shown to the LLM selector.")
    p.add_argument("--eval-prefix", default=DEFAULT_EVAL_PREFIX,
                   help="Shell prefix to activate the test env inside the image.")
    p.add_argument("--test-timeout", type=int, default=300,
                   help="Per test-run timeout (seconds) inside the container, so a hanging "
                        "test cannot block the stage.")
    p.add_argument("--trajectory-dir", default=None,
                   help="Dir for per-instance trajectory files recording the regression "
                        "LLM call as a turn (normalized with generate/selector). Defaults to "
                        "<output_dir>/regression_trajectories; pass '' to disable.")
    p.add_argument("--max_workers", type=int, default=4, help="Parallel workers across instances.")
    p.add_argument("--test-parallel", type=int, default=None,
                   help="Override the runner's process count for test_cmds that support it "
                        "(django runtests.py '--parallel N'). No-op for runners without the flag.")
    args = p.parse_args()

    # Default: write trajectories next to the pruned output so the turn counter finds
    # them alongside the generate/selector turns. Pass --trajectory-dir '' to disable.
    if args.trajectory_dir is None:
        args.trajectory_dir = os.path.join(
            os.path.dirname(os.path.abspath(args.output)), "regression_trajectories"
        )

    # Only touch (and pull images for) the instances present in the candidate file.
    with open(args.candidates) as f:
        candidate_ids = [json.loads(line)["instance_id"] for line in f if line.strip()]

    tester = RegressionTester(
        "SWE-bench",
        args.working_dir,
        args.config_file,
        args.dataset,
        "",   # docker_env_config
        "",   # benchmark_harness_path (unused)
        "trae-agent-regr",
        args.max_workers,
        candidate_ids,
        model_name=args.model_name,
        eval_prefix=args.eval_prefix,
        universe=args.universe,
        max_tests_for_llm=args.max_tests_for_llm,
        test_timeout=args.test_timeout,
        trajectory_dir=args.trajectory_dir,
        test_parallel=args.test_parallel,
    )
    tester.run(args.candidates, args.output, args.max_workers)


if __name__ == "__main__":
    main()
