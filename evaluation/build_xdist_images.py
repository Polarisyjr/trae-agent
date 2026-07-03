# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Pre-build `:xdist` derived images — bake pytest-xdist into each pytest SWE-bench
instance image so `pytest -n N` works in-container with zero per-run install cost.

`regression_test_swebench.RegressionTester` prefers `<repo>:xdist` for pytest repos (see
`_pick_image`); this tool creates those images once. For each pytest-repo instance image:

  1. start a container,
  2. record the testbed pytest version,
  3. `pip install pytest-xdist` WITH pytest PINNED (so xdist can never silently upgrade
     pytest — that would change test behaviour, worse than no parallelism),
  4. verify: pytest version unchanged AND `import xdist` AND pytest registered the `-n`
     option,
  5. `docker commit` -> `<repo>:xdist`.

Images where the pinned install has no compatible xdist (or would change pytest) are
SKIPPED and listed in the fallback file; they simply run pytest serially — xdist is a
speed optimization, not a correctness dependency, so serial is always a valid fallback.

Only the 10 pytest repos are relevant (django uses native --parallel; sympy uses
bin/test --split); non-pytest instances are ignored.

Usage (from the trae-agent repo root):
    uv run python -m evaluation.build_xdist_images --dataset SWE-bench_Verified
    uv run python -m evaluation.build_xdist_images --instances id1,id2
    uv run python -m evaluation.build_xdist_images --local-only     # only present images
    uv run python -m evaluation.build_xdist_images --rebuild        # ignore existing :xdist
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from docker import from_env
from tqdm import tqdm

from .regression_test_swebench import PYTEST_REPOS

EVAL_PREFIX = "source /opt/miniconda3/bin/activate testbed"


def image_name(instance_id: str) -> str:
    """SWE-bench instance image, honouring TRAE_SWEBENCH_REGISTRY (matches evaluation)."""
    if os.environ.get("TRAE_SWEBENCH_REGISTRY", "swebench").lower() == "epoch":
        return f"ghcr.io/epoch-research/swe-bench.eval.x86_64.{instance_id}:latest"
    return f"swebench/sweb.eval.x86_64.{instance_id.replace('__', '_1776_')}:latest"


def _exec(container, cmd: str):
    rc, out = container.exec_run(cmd=["/bin/bash", "-lc", f"{EVAL_PREFIX} && {cmd}"])
    return rc, out.decode("utf-8", "replace")


def build_one(client, instance_id: str, rebuild: bool) -> tuple[str, str]:
    """Return (instance_id, status) where status in {built, exists, skip-<reason>, error}."""
    base = image_name(instance_id)
    repo = base.rsplit(":", 1)[0]
    xdist_tag = f"{repo}:xdist"

    if not rebuild:
        try:
            client.images.get(xdist_tag)
            return instance_id, "exists"
        except Exception:
            pass
    try:
        base_img = client.images.get(base)
    except Exception:
        return instance_id, "skip-no-base-image"

    # Override the image ENTRYPOINT with a keepalive — some images (e.g. requests) ship a
    # custom entrypoint that runs a setup script and exits, which would kill the container
    # before we can install into it. `tail -f /dev/null` keeps any image alive. We RESTORE
    # the base image's Entrypoint/Cmd on commit (below) so the `:xdist` image behaves like
    # the base for a normal `docker run` (else it inherits the tail keepalive).
    base_cfg = base_img.attrs.get("Config", {}) or {}
    container = client.containers.run(base, entrypoint=["tail", "-f", "/dev/null"],
                                      detach=True)
    try:
        rc, ver = _exec(container, "python -c 'import pytest; print(pytest.__version__)'")
        before = ver.strip().splitlines()[-1] if rc == 0 else ""
        if not before:
            return instance_id, "skip-no-pytest"

        # Pin pytest so xdist resolves a compatible version and can never upgrade it.
        rc, out = _exec(container, f'pip install -q pytest-xdist "pytest=={before}"')
        if rc != 0:
            return instance_id, "skip-install-failed"

        rc, ver2 = _exec(container, "python -c 'import pytest; print(pytest.__version__)'")
        after = ver2.strip().splitlines()[-1] if rc == 0 else ""
        rc_imp, _ = _exec(container, "python -c 'import xdist'")
        rc_opt, opt = _exec(container, 'pytest -h 2>&1 | grep -- "-n numprocesses" || true')
        if after != before:
            return instance_id, f"skip-pytest-changed({before}->{after})"
        if rc_imp != 0:
            return instance_id, "skip-xdist-import-failed"
        if "numprocesses" not in opt:
            return instance_id, "skip-n-flag-missing"

        # Restore the base image's entrypoint/cmd (undo the tail keepalive override) so the
        # committed image runs normally under `docker run`. Pass [] (not None) when the base
        # has no Entrypoint/Cmd: docker's commit MERGES conf over the container config and
        # IGNORES null fields, so None would leave the `tail -f /dev/null` keepalive in the
        # committed image (verified). An empty list explicitly clears it back to the base's.
        container.commit(repository=repo, tag="xdist",
                         conf={"Entrypoint": base_cfg.get("Entrypoint") or [],
                               "Cmd": base_cfg.get("Cmd") or []})
        return instance_id, "built"
    except Exception as e:  # noqa: BLE001 - report and continue with the rest
        return instance_id, f"error:{type(e).__name__}"
    finally:
        container.remove(force=True)


def resolve_instances(args) -> list[str]:
    if args.instances:
        ids = [i for i in args.instances.split(",") if i]
    else:
        import warnings
        warnings.filterwarnings("ignore")
        from datasets import load_dataset
        hf = {"SWE-bench": "princeton-nlp/SWE-bench",
              "SWE-bench_Lite": "princeton-nlp/SWE-bench_Lite",
              "SWE-bench_Verified": "princeton-nlp/SWE-bench_Verified"}
        ds = load_dataset(hf[args.dataset], split="test")
        ids = [r["instance_id"] for r in ds if r["repo"] in PYTEST_REPOS]
    if args.local_only:
        client = from_env()
        keep = []
        for iid in ids:
            try:
                client.images.get(image_name(iid))
                keep.append(iid)
            except Exception:
                pass
        ids = keep
    return ids


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="SWE-bench_Verified")
    p.add_argument("--instances", default=None, help="comma-list of instance_ids (else dataset)")
    p.add_argument("--local-only", action="store_true", help="only images already present")
    p.add_argument("--rebuild", action="store_true", help="rebuild even if :xdist exists")
    p.add_argument("--max-workers", type=int, default=8, help="concurrent builds")
    p.add_argument("--fallback-out", default="xdist_fallback.json",
                   help="write instances left WITHOUT xdist (run serial) here")
    args = p.parse_args()

    ids = resolve_instances(args)
    print(f"pytest instances to process: {len(ids)}")
    client = from_env()

    results: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {ex.submit(build_one, client, iid, args.rebuild): iid for iid in ids}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="building :xdist"):
            iid, status = fut.result()
            results[iid] = status

    from collections import Counter
    summary = Counter(s.split("(")[0].split(":")[0] for s in results.values())
    print("\n=== summary ===")
    for k, v in summary.most_common():
        print(f"  {v:4d}  {k}")
    fallback = sorted(i for i, s in results.items() if s not in ("built", "exists"))
    with open(args.fallback_out, "w") as f:
        json.dump({"fallback_serial": fallback,
                   "detail": results}, f, indent=2)
    print(f"\n{len(fallback)} instance(s) WITHOUT xdist (serial pytest) -> {args.fallback_out}")


if __name__ == "__main__":
    main()
