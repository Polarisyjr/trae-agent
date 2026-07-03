# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Backfill `pytest-timeout` AND the `regr_stream` plugin into EXISTING `<repo>:xdist`
images.

Both ride the same "an `:xdist` image is fully provisioned" contract that the regression
tester assumes (it gates `--timeout` / `-p regr_stream` on the xdist image's mere
presence, not a per-run check):
  * pytest-timeout — backs the per-test `--timeout` (regression.per_test_timeout).
  * regr_stream    — streams per-test "<STATUS> <nodeid>" lines so a SIGKILLed whole-suite
                     find_passing keeps the tests that passed before the kill (loaded via
                     `-p regr_stream`; see regr_stream_plugin.py).
`build_xdist_images.py` now bakes BOTH into fresh images, but the ~194 already-built
`:xdist` images predate them and would fail pytest with 'unrecognized arguments: --timeout'
or 'no module named regr_stream'. This does the minimal incremental install (no xdist
rebuild): for each existing `:xdist` image, install whichever of the two is missing and
commit back to the SAME tag, preserving its entrypoint/cmd. Idempotent — images that
already have both are left untouched.

    uv run python -m evaluation.backfill_pytest_timeout            # all local :xdist images
    uv run python -m evaluation.backfill_pytest_timeout --instances astropy__astropy-14995
    uv run python -m evaluation.backfill_pytest_timeout --workers 4
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from docker import from_env
from tqdm import tqdm

from .regr_stream_plugin import install_regr_stream

EVAL_PREFIX = "source /opt/miniconda3/bin/activate testbed"


def _exec(container, cmd: str):
    rc, out = container.exec_run(cmd=["/bin/bash", "-lc", f"{EVAL_PREFIX} && {cmd}"])
    return rc, out.decode("utf-8", "replace")


def _xdist_tags(client, instances: list[str] | None) -> list[str]:
    if instances:
        return [f"swebench/sweb.eval.x86_64.{i.replace('__', '_1776_')}:xdist"
                for i in instances]
    tags = []
    for img in client.images.list():
        tags += [t for t in (img.tags or []) if t.endswith(":xdist")]
    return sorted(set(tags))


def backfill_one(client, xdist_tag: str) -> tuple[str, str]:
    try:
        img = client.images.get(xdist_tag)
    except Exception:
        return xdist_tag, "skip-no-image"
    cfg = img.attrs.get("Config", {}) or {}
    # Keepalive entrypoint so we can exec into it; RESTORE the image's real entrypoint/cmd
    # on commit (else the tag would inherit the tail keepalive).
    container = client.containers.run(xdist_tag, entrypoint=["tail", "-f", "/dev/null"],
                                      detach=True)
    try:
        need_timeout = _exec(container, "python -c 'import pytest_timeout' 2>/dev/null")[0] != 0
        need_stream = _exec(container, "python -c 'import regr_stream' 2>/dev/null")[0] != 0
        if not need_timeout and not need_stream:
            return xdist_tag, "already-present"
        did = []
        if need_timeout:
            rc, out = _exec(container, "pip install -q pytest-timeout")
            if rc != 0:
                return xdist_tag, f"install-failed:{out.strip().splitlines()[-1][:80] if out.strip() else ''}"
            if _exec(container, "python -c 'import pytest_timeout'")[0] != 0:
                return xdist_tag, "import-failed"
            did.append("timeout")
        if need_stream:
            ok, msg = install_regr_stream(lambda c: _exec(container, c))
            if not ok:
                return xdist_tag, f"regr-stream-{msg}"
            did.append("regr_stream")
        repo = xdist_tag.rsplit(":", 1)[0]
        container.commit(repository=repo, tag="xdist",
                         conf={"Entrypoint": cfg.get("Entrypoint") or [],
                               "Cmd": cfg.get("Cmd") or []})
        return xdist_tag, "backfilled:" + "+".join(did)
    except Exception as e:  # noqa: BLE001
        return xdist_tag, f"error:{type(e).__name__}"
    finally:
        try:
            container.remove(force=True)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default="",
                    help="comma-list of instance_ids; default = all local :xdist images")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    client = from_env()
    instances = [i for i in args.instances.split(",") if i] or None
    tags = _xdist_tags(client, instances)
    if not tags:
        print("no :xdist images found")
        return
    print(f"backfilling pytest-timeout into {len(tags)} :xdist image(s), "
          f"workers={args.workers}")

    results: dict[str, list[str]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(backfill_one, client, t): t for t in tags}
        for f in tqdm(as_completed(futs), total=len(futs)):
            tag, status = f.result()
            key = status.split(":", 1)[0]
            results.setdefault(key, []).append(tag)

    print("\n=== summary ===")
    for status, ts in sorted(results.items()):
        print(f"  {status:18s}: {len(ts)}")
    for status, ts in sorted(results.items()):
        if status not in ("backfilled", "already-present"):
            for t in ts:
                print(f"    {status}: {t}")


if __name__ == "__main__":
    main()
