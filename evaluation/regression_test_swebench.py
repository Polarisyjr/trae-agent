# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Regression-testing (patch pruning) for SWE-bench candidate patches.

This implements the *tester* of the Trae Agent paper's patch-pruning component
(Section 3.3, "Perform regression testing"), faithfully (path A — no peeking at the
gold FAIL_TO_PASS/PASS_TO_PASS labels by default):

  1. Find passing tests: run the repo's test suite on the pristine codebase and keep
     the tests that PASS. (No LLM.)
  2. Select regression tests: ask the LLM to name only the FEW passing tests a correct
     fix may legitimately change (e.g. a test asserting the buggy behaviour); everything
     else is kept as a regression test. Exclusion framing (vs. "list the subset to keep")
     keeps the LLM output tiny — usually empty — instead of forcing it to re-emit the
     whole passing list verbatim, which is slow and degenerates into a blind echo.
     (LLM — via your trae_config.yaml model, e.g. vLLM.)
  3. Validate each candidate: apply the patch, run the selected regression tests, and
     record which fail. (No LLM.)

Test names are adapted per repo before they can be re-run (`test_directives`): the
parser emits names in a runner-specific format that is not always a valid CLI selector
(django's "method (module.Class)" must become "module.Class.method"; sympy is parsed by
`parse_sympy_addressable` into file-addressable "file.py::test_name" and selected at exact
test granularity via an in-process matches() patch).
Parallelism (`_shard_plan` / `_run_partitioned`) prefers NATIVE in-container parallelism
— one container, the runner parallelizes inside — which far outperforms a many-container
fan-out (a single Python process driving many blocking exec streams stalls under
GIL/docker/disk contention). django uses `runtests.py --parallel N`; pytest uses
`pytest -n N` on a pre-built `:xdist` image (see build_xdist_images.py); sympy, which has
neither native parallelism nor a CLI test selector, runs n `bin/test --split i/n` shards
BACKGROUNDED inside ONE container (concurrency container-side, so the host drains a single
exec stream). The `test_parallel` knob (n) drives all three.

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
import hashlib
import io
import json
import os
import random
import re
import shlex
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


class RegressionExecError(RuntimeError):
    """The test command could not be executed AT ALL — the docker exec layer rejected
    it before the framework ran (e.g. the command string exceeded the kernel's
    MAX_ARG_STRLEN and returned E2BIG "argument list too long"). Such a run produces
    NO test output, so its result must not be read as "every selected test failed":
    that would silently mark all regressions failed for every candidate and prune them
    all. Raised so the failure is loud (caught + logged per-instance in run()) instead
    of corrupting the regression set with a phantom all-fail."""


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
            f.write(json.dumps({"ts_start": t0, "ts_end": t1,
                                "wall_s": t1 - t0, "stage": stage,
                                "instance_id": instance_id,
                                "kind": kind}) + "\n")
    except OSError:
        pass

# Repos whose parser (MAP_REPO_TO_PARSER) emits test names that are NOT valid
# arguments to their own test_cmd, so the names must be adapted before they can be
# passed back to select those tests. See `test_directives` below for the full audit.
_DJANGO_NAME_RE = re.compile(r"^(?P<method>\S+)\s+\((?P<path>[\w.]+)\)$")

# sympy `bin/test --verbose` groups each file's results under a header line
# `sympy/.../test_file.py[N]`, then lists bare `test_name ok|F|E`. The stock swebench
# parser keeps only the bare name (loses the file, collides across files, and can't be
# re-selected). We parse the header + results ourselves to name tests uniquely and
# addressably as `sympy/.../test_file.py::test_name`.
_SYMPY_FILE_RE = re.compile(r"^(sympy/[\w./]+\.py)\[\d+\]")
_SYMPY_RESULT_RE = re.compile(r"^(test_\w+)\s+(ok|F|E|Skipped|s)\b")


def parse_sympy_addressable(log: str) -> dict[str, str]:
    """Parse sympy `bin/test --verbose` output into {`file.py::test_name`: status},
    tracking the per-file header so names are unique and file-addressable. Concatenated
    `--split` shard logs parse correctly since each shard carries its own file headers."""
    out: dict[str, str] = {}
    cur = None
    for line in log.splitlines():
        s = line.strip()
        mf = _SYMPY_FILE_RE.match(s)
        if mf:
            cur = mf.group(1)
            continue
        mr = _SYMPY_RESULT_RE.match(s)
        if mr and cur:
            v = mr.group(2)
            out[f"{cur}::{mr.group(1)}"] = (
                TestStatus.PASSED.value if v == "ok"
                else TestStatus.SKIPPED.value if v in ("s", "Skipped")
                else TestStatus.FAILED.value)
    return out

# django: unittest's verbose description for a test WITH a docstring is TWO lines —
# the real id, then the docstring first line carrying the " ... ok" suffix — so the
# stock SWE-bench parser records the docstring text as the "test name" ("Verify base
# formset honors DELETE field"): a pseudo-entry that cannot be re-selected as a
# runtests.py label. This parser attributes such a status to the preceding bare-id
# line, recording BOTH keys: the real id (label-addressable) and the docstring text
# (so results stay comparable with dataset PASS_TO_PASS names, which use the stock
# parser's pseudo-names).
_DJANGO_TAIL_ID_RE = re.compile(r"(\S+ \([\w.]+\))$")


def parse_django_addressable(log: str) -> dict[str, str]:
    out: dict[str, str] = {}
    prev_id = None

    def put(name: str, status: str):
        nonlocal prev_id
        name = name.strip()
        if not _DJANGO_NAME_RE.match(name):
            if "..." in name:
                # output glued to the status line (e.g. "Applying <migration>...test_x
                # (mod.Class)" — the stock parser's django-7188 special case, generalized)
                cand = name.rsplit("...", 1)[-1].strip()
                if _DJANGO_NAME_RE.match(cand):
                    name = cand
            m = _DJANGO_TAIL_ID_RE.search(name)
            if not _DJANGO_NAME_RE.match(name) and m and _DJANGO_NAME_RE.match(m.group(1)):
                name = m.group(1)
        if not _DJANGO_NAME_RE.match(name) and prev_id:
            out[prev_id] = status          # docstring line: credit the real id too
        out[name] = status
        prev_id = None

    for raw in log.splitlines():
        line = raw.strip()
        if not line:
            prev_id = None
            continue
        matched = False
        for suffix, status in ((" ... ok", TestStatus.PASSED.value),
                               (" ... OK", TestStatus.PASSED.value),
                               (" ...  OK", TestStatus.PASSED.value),
                               (" ... FAIL", TestStatus.FAILED.value),
                               (" ... ERROR", TestStatus.ERROR.value)):
            if line.endswith(suffix):
                put(line[: -len(suffix)], status)
                matched = True
                break
        if matched:
            continue
        if " ... skipped" in line:
            put(line.split(" ... skipped")[0], TestStatus.SKIPPED.value)
            continue
        if line.startswith(("FAIL:", "ERROR:")):
            status = (TestStatus.FAILED.value if line.startswith("FAIL:")
                      else TestStatus.ERROR.value)
            name = line.split(":", 1)[1].strip()
            out[name if _DJANGO_NAME_RE.match(name) else name.split()[0]] = status
            prev_id = None
            continue
        prev_id = line if _DJANGO_NAME_RE.match(line) else None
    return out


# The SWE-bench-Verified repos whose test_cmd is pytest-based (directly, or via tox's
# `--`). These get in-container `pytest -n N` parallelism using the pre-built `:xdist`
# image (see build_xdist_images.py). django uses native `runtests.py --parallel`; sympy
# uses `bin/test --split`; neither is in this set.
PYTEST_REPOS = {
    "sphinx-doc/sphinx", "matplotlib/matplotlib", "scikit-learn/scikit-learn",
    "astropy/astropy", "pydata/xarray", "pytest-dev/pytest",
    "pylint-dev/pylint", "psf/requests", "mwaskom/seaborn", "pallets/flask",
}


def test_directives(repo: str, names: list[str]) -> list[str] | None:
    """Unified adapter: turn parser-emitted test names into CLI selector tokens for
    `repo`'s test_cmd, so `find_passing` names round-trip into a runnable `validate`
    selection across ALL SWE-bench(-Verified) repos.

    Returns None only if a repo's output truly cannot address tests; all 12 Verified repos
    are addressable, so in practice this returns a (possibly file-granular) selector list.

    Audit of the 12 SWE-bench-Verified repos (parser -> emitted format -> runner arg):
      * django/django   parse_log_django   "method (dotted.module.Class)"  ->  runtests.py
        wants "dotted.module.Class.method". FLIP required (else the whole cmd errors out
        and every test is spuriously marked failed).
      * sympy/sympy     parse_sympy_addressable  "file.py::test_name" (our file-aware
        parser). Names round-trip unchanged; the sympy command builder runs the covering
        files in-process with matches() patched to an EXACT name set (subprocess=False), so
        selection is TEST-granular and slow tests co-located in a file are not executed.
        Fixes the stock parser's bare, colliding, unrunnable names.
      * pylint / requests  parse_log_pytest_options truncates path-like "[/a/b/c]"
        parametrize ids to "[/c]", so the exact node id is unrecoverable. Drop the
        path-like param and select at function granularity (a safe superset; the
        specific param's status is still read back from the re-parsed output).
      * everyone else (sphinx, matplotlib, scikit-learn, astropy, xarray, pytest,
        seaborn, flask): pytest node ids "path::Class::test[param]" ARE valid pytest
        args and round-trip as-is (sphinx appends them after tox's `--`). Identity.
    """
    if repo == "django/django":
        out = []
        for n in names:
            m = _DJANGO_NAME_RE.match(n.strip())
            out.append(f"{m['path']}.{m['method']}" if m else n)
        return out
    if repo == "sympy/sympy":
        # Names are "file.py::test_name" (parse_sympy_addressable). Keep TEST granularity:
        # return the exact names (deduped). bin/test's CLI can only select by FILE, but the
        # sympy command builder runs the covering files in-process with matches() patched to
        # an EXACT name set (subprocess=False), so slow tests co-located in the same file
        # (e.g. test_integrals.py::test_issue_4737 among its 108 tests) are never executed.
        return sorted(set(names))
    if repo in ("pylint-dev/pylint", "psf/requests"):
        # Strip a path-like "[/...]" param (the parser already mangled it) and select
        # the parametrized test at function level; dedupe while preserving order.
        stripped = [re.sub(r"\[/.*\]$", "", n) for n in names]
        return list(dict.fromkeys(stripped))
    return list(names)  # pytest node ids round-trip unchanged


SELECT_PROMPT = """You are selecting regression tests for a software issue.

You are given a GitHub issue and a NUMBERED list of tests that currently PASS in the
pristine repository. Select ONLY a small, truly representative subset — the few most
IMPORTANT tests that best guard the behavior a correct fix for this issue could affect —
roughly {k} tests (do NOT greatly exceed {k}). Prefer quality over quantity: choose the
minimal meaningful set, not a broad sample. After a CORRECT fix for the issue, every
selected test must still pass.

The qualitative framing ("a small, truly representative subset") and the numeric anchor
("roughly {k}") are BOTH load-bearing: measured on a 6-batch django run, wording alone
swings wildly (0–1514 per 2000-test batch) and a bare count over-selects ~4x; together
they hold ~1.7x of {k} with low variance.

Guidelines:
- First include tests covering code the fix could plausibly touch (the same
  module/subsystem the issue is about) — these are the likeliest to catch a bad fix.
- Spread the remainder across DIFFERENT modules/files for breadth; prefer covering
  many distinct files over many tests from the same file.
- Do NOT select a test that appears to assert behavior the issue says is buggy,
  obsolete, or intentionally changed by the fix (a correct fix legitimately changes
  those).

Return ONLY a JSON array of the selected test NUMBERS (integers from the list below),
e.g. [0, 17, 42]. Do not return test names.

## Issue
{issue}

## Passing tests
{tests}
"""


class RegressionTester(BenchmarkEvaluation):
    """Fill the `regressions` field of candidate patches via regression testing."""

    def __init__(self, *args, model_name: str, eval_prefix: str, universe: str,
                 max_tests_for_llm: int, test_timeout: int = 300,
                 per_test_timeout: int = 0,
                 trajectory_dir: str | None = None, test_parallel: int | None = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_prefix = eval_prefix
        self.universe = universe
        self.max_tests_for_llm = max_tests_for_llm
        # Two nested timeouts guard the stage:
        #   test_timeout      — wall-clock cap on a whole test-RUN (the batched command),
        #                       enforced host-side by `timeout(1)`; on expiry the entire
        #                       run is SIGKILLed and its unfinished tests count as failed.
        #   per_test_timeout  — cap on a SINGLE test function, enforced INSIDE the runner
        #                       (sympy `--timeout`, pytest `--timeout`); a slow test is
        #                       marked failed on its own while the rest of the run
        #                       continues. 0 disables it. This is the unified knob;
        #                       `_per_test_*` helpers translate it per runner, and django
        #                       (no runner support) falls back to test_timeout alone.
        self.test_timeout = test_timeout
        self.per_test_timeout = per_test_timeout
        # Unified parallelism degree n: each test run is partitioned into n shards, each
        # executed in its OWN fresh container concurrently (see `_shard_plan`). Works for
        # EVERY SWE-bench repo — django/pytest shard the test-id list, sympy shards via
        # `bin/test --split`, and django's whole-suite discovery uses native --parallel n.
        # None/1 -> no fan-out (single container, serial). This replaces the old
        # django-only `--parallel N` knob.
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

    @staticmethod
    def _xdist_tag(base_image: str) -> str:
        """`<repo>:latest` -> `<repo>:xdist` (the derived image with pytest-xdist baked
        in by build_xdist_images.py)."""
        return f"{base_image.rsplit(':', 1)[0]}:xdist"

    def _pick_image(self, instance_id: str, repo: str) -> str:
        """Prefer the pre-built `:xdist` image for pytest repos so `pytest -n N` works
        without a per-container install. Falls back to the base image (serial pytest)
        when no xdist image was built for this instance."""
        base = self._image_name(instance_id)
        if repo in PYTEST_REPOS:
            xt = self._xdist_tag(base)
            try:
                self.docker_client.images.get(xt)
                return xt
            except Exception:
                pass
        return base

    def _start_container(self, instance_id: str, repo: str = ""):
        # Override the image ENTRYPOINT with a keepalive: some SWE-bench images ship a
        # custom entrypoint that runs a setup script and EXITS (e.g. requests seds a
        # timeout into test_requests.py), which would make `command=/bin/bash` a no-op arg
        # and kill the container immediately. `tail -f /dev/null` keeps every image alive
        # regardless of its entrypoint; we exec our own test commands into it.
        return self.docker_client.containers.run(
            self._pick_image(instance_id, repo),
            entrypoint=["tail", "-f", "/dev/null"],
            detach=True,
            labels=({"multiagent.trae_sweep": os.environ["TRAE_SWEEP_RUN_ID"]}
                    if os.environ.get("TRAE_SWEEP_RUN_ID") else None),
        )

    def _has_xdist(self, instance_id: str, repo: str) -> bool:
        return self._pick_image(instance_id, repo) != self._image_name(instance_id)

    def _bash(self, container, command: str):
        # Pass argv as a LIST so docker-py sends it as-is instead of shlex-splitting a
        # `bash -c '<command>'` STRING. The regression command embeds the full test_cmd
        # plus up to 300 test ids; quotes/apostrophes inside it break the single-quote
        # wrap (shlex raises "No closing quotation"). A list sidesteps the split entirely.
        rc, out = container.exec_run(cmd=["/bin/bash", "-c", command])
        return rc, out.decode("utf-8")

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
                with open(path) as handle:
                    for line in handle:
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
                   instance_id: str = "", phase: str = "test", wrap: bool = True,
                   timeout: int | None = None) -> str:
        to = self.test_timeout if timeout is None else timeout
        # Pin PYTHONHASHSEED so find_passing and validate use the SAME hash ordering.
        # validate is a differential re-run (does a test that passed in find_passing still
        # pass?); hash-randomization-dependent tests (common in sympy's symbolic code) would
        # otherwise flip pass<->fail between the two runs and show up as spurious failures.
        # Exported once here, it propagates to all child workers (django --parallel, pytest
        # -n xdist, sympy --split subprocesses), which then inherit it via `env`.
        prefix = f"{self.eval_prefix} && export PYTHONHASHSEED=0"
        if wrap:
            # shlex.quote each id: pytest node ids carry parametrize brackets and, for
            # some suites (astropy), shell metacharacters like '<'/'>' inside the param
            # (e.g. "test_quantity[...-<f8]"). Unquoted on the `bash -c` line, '<f8]'
            # is parsed as an input redirection from a nonexistent file, aborting the
            # WHOLE command before the framework starts -> empty output -> every
            # selected test spuriously marked failed. Quoting makes each id a literal arg.
            ids = (" " + " ".join(shlex.quote(t) for t in test_ids)) if test_ids else ""
            # Wrap in `timeout` so a hanging test (e.g. requests network tests with no
            # connectivity) can't block the whole stage indefinitely. On timeout the run is
            # killed; tests with no PASSED line are then treated as failed (conservative).
            # `env` so a test_cmd with a leading VAR=val (e.g. sympy's
            # "PYTHONWARNINGS='...' bin/test") works: without it `timeout` treats the
            # assignment as the program name and dies with rc 127 in <1s (0 passing).
            cmd = f"{prefix} && timeout {to} env {test_cmd}{ids}"
        else:
            # test_cmd is already a complete shell command (e.g. sympy's background
            # `--split` fan-out with its own per-process `timeout`/`env`). Run as-is so we
            # don't double-wrap `env`/`timeout` around a compound command. Wrap it in a
            # SUBSHELL so the leading `&&` gates the WHOLE compound: `prefix && (a; b & wait)`
            # aborts everything if eval_prefix/cd fails, whereas `prefix && a; b` would gate
            # only `a` and still run the rest in the wrong env/cwd (spurious all-fail).
            cmd = f"{prefix} && ({test_cmd})"
        # Measure the test-suite run: wall-clock + container CPU-seconds (cgroup delta
        # around the exec). Excludes container startup (already running) and the
        # reset/apply done in separate exec calls — just this test run + its eval_prefix.
        import time
        t0 = time.time()
        c0 = self._container_cpu_usec(container)
        rc, out = self._bash(container, cmd)
        t1 = time.time()
        c1 = self._container_cpu_usec(container)
        self._record_test_run(
            instance_id, phase, t0, t1, c0, c1, test_ids, cmd, rc, out,
            container.image.id, (container.image.tags or [container.image.id])[0],
        )
        # docker exec returns rc 255 with an "exec …" diagnostic when bash itself could
        # not start — most importantly "argument list too long" (E2BIG) when the command
        # string exceeds the kernel's MAX_ARG_STRLEN (~128 KB). The framework never ran,
        # so `out` is empty and would parse to an all-fail. No real test framework exits
        # 255 here (pytest 0-5, unittest 0/1, timeout 124/137), so this is unambiguous.
        if rc == 255 and ("argument list too long" in out or out.startswith("exec ")):
            raise RegressionExecError(
                f"{instance_id} [{phase}]: test command failed to exec (rc={rc}, "
                f"cmd={len(cmd)} bytes): {out.strip()[:200]!r}")
        return out

    def _record_test_run(
        self, instance_id, phase, t0, t1, c0, c1, test_ids, command, exit_code, output,
        image_id, image_ref,
    ) -> None:
        """Append one test-run timing record (epoch ts + wall + container CPU-seconds)
        to <trajectory_dir>/<instance_id>_test_runs.jsonl, on the SAME clock as step3."""
        if not self.trajectory_dir:
            return
        cpu_s = (c1 - c0) / 1e6 if (c0 is not None and c1 is not None) else None
        rec = {"ts_start": t0, "ts_end": t1,
               "wall_s": t1 - t0,
               "cpu_s": cpu_s,
               "phase": phase, "instance_id": instance_id,
               "n_tests": len(test_ids) if test_ids else None,
               "command": command, "exit_code": exit_code,
               "stdout_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
               "image_id": image_id, "image_ref": image_ref}
        try:
            os.makedirs(self.trajectory_dir, exist_ok=True)
            path = os.path.join(self.trajectory_dir, f"{instance_id}_test_runs.jsonl")
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError as e:
            print(f"[regr] {instance_id}: failed to record test run: {e}")

    def _parse(self, repo: str, version: str, instance_id: str, log: str) -> dict[str, str]:
        # sympy: use our file-aware parser so tests are uniquely named and addressable
        # ("file.py::test_name") instead of the stock parser's bare, colliding names.
        if repo == "sympy/sympy":
            try:
                return parse_sympy_addressable(log)
            except Exception:
                return {}
        # django: docstring-aware parser (real ids instead of docstring pseudo-names).
        if repo == "django/django":
            try:
                return parse_django_addressable(log)
            except Exception:
                return {}
        parser = MAP_REPO_TO_PARSER[repo]
        stub = SimpleNamespace(repo=repo, version=version, instance_id=instance_id)
        try:
            return parser(log, stub)
        except Exception:
            return {}

    # --- pipeline steps ------------------------------------------------------

    def _test_cmd(self, repo: str, version: str, parallel: int | None = None) -> str:
        cmd = MAP_REPO_VERSION_TO_SPECS[repo][version]["test_cmd"]
        # Rewrite django's in-runner `--parallel N` token when asked (native DB-cloning
        # parallelism, used for the whole-suite find_passing where we have no ids to
        # shard). No-op for runners without the flag (pytest/sympy).
        if parallel is not None and "--parallel" in cmd:
            cmd = re.sub(r"--parallel\s+\d+", f"--parallel {parallel}", cmd)
        return cmd

    # --- unified parallelism (native-in-container, xdist baked in) -----------

    # sympy has neither native in-process parallelism nor id selection. It IS parallelized
    # by n concurrent `bin/test --split i/n` processes — but backgrounded INSIDE ONE
    # container (shell `&` ... `wait`), NOT as n Python-driven containers. The core rule:
    # concurrency lives container-side so the host's single Python process only ever drains
    # ONE exec_run stream (each split writes its own /tmp log, cat'd at the end), which is
    # why a high split count is fine here — unlike a Python-driven container fan-out, which
    # stalls (GIL + docker + disk). Splits beyond the number of test files just run fewer
    # (or zero) files each, which is harmless.

    @staticmethod
    def _inject_pytest_n(cmd: str, n: int) -> str:
        """Add `-n N` (pytest-xdist). Appending works both for a bare `pytest ...` cmd and
        for tox's `... --` form (args after `--` go to pytest). Only inject when n>1 (i.e.
        the `:xdist` image is present); otherwise leave the cmd serial."""
        return f"{cmd} -n {n}" if n and n > 1 else cmd

    def _pytest_cmd(self, repo: str, version: str, pytest_n: int,
                    has_timeout: bool = False) -> str:
        """Build the pytest command for a pytest repo with two speed flags that don't
        change the `-rA` PASSED/FAILED lines the parser reads:
          * `--tb=no`  — skip traceback formatting (we only need pass/fail).
          * `-W ignore` — don't collect/aggregate the end-of-run warnings summary. On huge
            suites this single-threaded pass DOMINATES the wall time after the parallel
            workers finish (astropy: ~500s -> ~64s). It's self-consistent: find_passing and
            validate use the SAME command, so any warning-filter effect cancels between the
            two stages and never produces a spurious validate diff.
        Then append `-n N` (xdist) when the `:xdist` image is present.

        When `has_timeout` (the `:xdist` image also carries pytest-timeout) and a per-test
        cap is set, append `--timeout=N --timeout-method=signal`: signal (SIGALRM, main
        thread) fails the ONE slow test and lets the run continue, unlike the `thread`
        method which kills the whole worker. Gated on the plugin's presence — passing
        `--timeout` to a pytest without it aborts the run with 'unrecognized arguments'.

        Also load `-p regr_stream` on the `:xdist` image (baked in alongside xdist/timeout;
        see regr_stream_plugin.py). It streams each test's "<STATUS> <nodeid>" line as the
        test completes, so a whole-suite find_passing that is SIGKILLed on its wall-clock
        `timeout` (heavy suites like scikit-learn) still yields the tests that passed BEFORE
        the kill — the `-rA` summary the parser reads is emitted only at the end and is
        otherwise lost, zeroing the passing set (and silently the whole prune). Same
        image-provisioning contract as `--timeout`: `-p` on an image lacking the module
        would abort, so it rides the `:xdist` gate and the backfill provisions old images."""
        cmd = self._test_cmd(repo, version) + " --tb=no -W ignore"
        if has_timeout:
            cmd += " -p regr_stream"
            if self.per_test_timeout > 0:
                cmd += f" --timeout={self.per_test_timeout} --timeout-method=signal"
        return self._inject_pytest_n(cmd, pytest_n)

    # Fixed sympy test seed. sympy re-seeds Python's `random` module once per run with a
    # FRESH random value (printed as "random seed: N"), which PYTHONHASHSEED does NOT
    # control. Randomized numeric tests (evalf/heurisch/…) then flip pass<->fail between
    # runs on different draws — so a test can pass find_passing under a lucky seed and fail
    # validate under another (a spurious "regression" with no patch). Pinning the seed makes
    # every run deterministic; the retry then decides regressions differentially (see
    # `_retry_failures`) to also cancel run-ARRANGEMENT effects a fixed seed can't.
    _SYMPY_SEED = 0

    # In-process EXACT-match runner (validate/retry). `sys.argv[1:-1]` = files, `[-1]` =
    # "i/k". matches() is patched to an exact name set (from $SYMPY_WANT) before any test is
    # collected; subprocess=False keeps that patch in the process that actually runs the
    # tests (sympy's default per-file subprocess would bypass a parent-process patch).
    # verbose=True keeps the parser's "file[N]" + "name ok" lines; colors=False keeps them
    # ANSI-free; seed=$SYMPY_SEED pins the per-test random draws. sympy's substring `-k`
    # can't do the selection (it over-selects prefix siblings, and its CLI takes only ONE
    # keyword); the Python API's per-function predicate can.
    # sympy.test's `timeout` (SIGALRM per test function) is read from $SYMPY_TIMEOUT;
    # 0/absent -> falsy -> no per-test timeout (unchanged behaviour).
    _SYMPY_EXACT = (
        'import os,sys,sympy;'
        'from sympy.utilities import runtests;'
        '_w=set(os.environ["SYMPY_WANT"].split(","));'
        'runtests.SymPyTests.matches=(lambda self,x: x.__name__ in _w);'
        'raise SystemExit(0 if sympy.test(*sys.argv[1:-1], split=sys.argv[-1],'
        ' subprocess=False, verbose=True, colors=False,'
        ' seed=int(os.environ["SYMPY_SEED"]),'
        ' timeout=int(os.environ.get("SYMPY_TIMEOUT","0") or 0)) else 1)'
    )

    def _sympy_split_loop(self, version: str, k: int,
                          directives: list[str] | None = None,
                          timeout: int | None = None) -> str:
        """A single shell command that runs k `--split i/k` shards in the BACKGROUND
        (concurrency container-side), waits, then cats every shard's log so the parser sees
        the union. Each shard carries its own `timeout … env` (wrap=False in `_run_tests`).

          * directives is None (find_passing): whole-suite `bin/test -C --verbose --split
            i/k` — sympy's native split partitions the WHOLE suite across k shards.
          * directives are "file.py::test_name" (validate/retry): each shard gets its own
            SLICE of the covering files (file_list[i::k]) run whole (split 1/1) via
            `python -c` with matches() patched to the EXACT name set, so slow tests
            co-located in those files are never executed. Slicing (vs full-list + i/k in
            every shard) keeps the compound command well under the exec arg-size limit.
            `-C`'s no-cache is reproduced with SYMPY_USE_CACHE=no so find_passing and
            validate stay consistent.
        Both branches pin `--seed`/`seed=` to _SYMPY_SEED for deterministic runs.
        """
        to = self.test_timeout if timeout is None else timeout
        pt = self.per_test_timeout
        base = self._test_cmd("sympy/sympy", version)
        if directives:
            env_prefix = base.split("bin/test", 1)[0].strip()  # PYTHONWARNINGS='…'
            file_list = sorted({d.split("::", 1)[0] for d in directives})
            want = ",".join(sorted({d.split("::", 1)[1]
                                    for d in directives if "::" in d}))
            # Give each shard its OWN slice of the covering files (file_list[i::k]) and run
            # that slice whole (split 1/1). The naive form — hand EVERY shard the FULL file
            # list plus `i/k` and let sympy.test partition internally — repeats the entire
            # file list AND the exact-name set (SYMPY_WANT) in all k shards, so the single
            # `bash -c` string balloons (95 shards x ~17 KB -> ~1.6 MB for a big suite) and
            # docker's exec aborts before bash even starts with "argument list too long"
            # (E2BIG, the kernel's ~128 KB MAX_ARG_STRLEN) -> empty output -> every selected
            # test silently marked failed. Slicing keeps each file in exactly ONE shard
            # (union == all files, same result and same k-way concurrency, no split-boundary
            # double runs); SYMPY_WANT is identical across shards, so export it ONCE instead
            # of inlining it k times. matches() still restricts execution to the exact names,
            # so co-located slow tests never run — the test-granularity guarantee is unchanged.
            # sympy.test reads the per-test timeout from $SYMPY_TIMEOUT.
            shard_files = [file_list[i::k] for i in range(k)]
            launches = f"export SYMPY_WANT='{want}'; " + " ".join(
                f"timeout {to} env {env_prefix} SYMPY_USE_CACHE=no "
                f"SYMPY_SEED={self._SYMPY_SEED} SYMPY_TIMEOUT={pt} "
                f"python -c '{self._SYMPY_EXACT}' {' '.join(sf)} 1/1 "
                f"> /tmp/sy_{i + 1}.log 2>&1 &"
                for i, sf in enumerate(shard_files) if sf
            )
        else:
            # whole-suite find_passing: bin/test's own `--timeout N` (SIGALRM per test).
            pt_flag = f"--timeout {pt} " if pt > 0 else ""
            launches = " ".join(
                f"timeout {to} env {base} {pt_flag}--seed {self._SYMPY_SEED} "
                f"--split {i + 1}/{k} > /tmp/sy_{i + 1}.log 2>&1 &"
                for i in range(k)
            )
        return f"rm -f /tmp/sy_*.log; {launches} wait; cat /tmp/sy_*.log"

    def _shard_plan(self, repo: str, version: str, directives: list[str] | None,
                    n: int, pytest_n: int, timeout: int | None = None,
                    has_pytest_timeout: bool = False) -> list[tuple[str, list[str] | None, bool]]:
        """Return (cmd, ids, wrap) shards. The design goal is find_passing-style NATIVE
        in-container parallelism (one container, the runner parallelizes inside), which
        massively outperforms a many-container fan-out. Every runner yields ONE shard:

          * django  -> `runtests.py --parallel n <labels>` (DB-cloning parallelism inside).
          * pytest  -> `pytest -n <pytest_n> <ids>` on the pre-built `:xdist` image
            (pytest_n = n when xdist is available, else 1 = serial).
          * sympy   -> one container running n backgrounded `--split i/n` shards
            (concurrency container-side; wrap=False as the loop is a complete command).
            directives, if given, are "file.py::test" names run at EXACT test granularity
            (in-process matches() patch); None -> whole-suite `bin/test` (find_passing).

        wrap=True means _run_tests adds the standard `timeout … env` + ids; wrap=False runs
        the cmd as-is (sympy's loop already carries per-split timeout/env)."""
        n = max(1, n)
        if repo == "django/django":
            # django's runtests.py (unittest) has NO per-test timeout knob; per_test_timeout
            # does not apply here — it relies on the per-RUN test_timeout as the only guard.
            return [(self._test_cmd(repo, version, parallel=n), directives, True)]
        if repo in PYTEST_REPOS:
            return [(self._pytest_cmd(repo, version, pytest_n, has_pytest_timeout),
                     directives, True)]
        if repo == "sympy/sympy":
            # directives (if any) are "file.py::test_name" from test_directives; None ->
            # whole suite. Always route through the split loop: its exact-match branch
            # consumes the file::test directives, which bin/test's CLI can't take.
            k = max(1, n)
            if directives:
                # `--split i/k` partitions the covering FILE list, so more shards than files
                # just spawn extra `python -c 'import sympy…'` processes that collect ZERO
                # tests. Cap k at the file count (whole-suite keeps k=n for the big suite).
                k = min(k, len({d.split("::", 1)[0] for d in directives}))
            return [(self._sympy_split_loop(version, max(1, k), directives, timeout),
                     None, False)]
        # any unknown runner: one plain run.
        return [(self._test_cmd(repo, version), directives, True)]

    def _run_partitioned(self, repo: str, version: str, instance_id: str, phase: str,
                         directives: list[str] | None, patch_text: str | None,
                         n: int, timeout: int | None = None) -> tuple[dict[str, str], bool]:
        """Run the shard plan and merge the per-shard parsed statuses. Every runner yields
        exactly ONE shard now (parallelism is container-side: django --parallel, pytest
        -n, sympy backgrounded --split), so this is a single container per call. Returns
        (merged_status, apply_failed).

        Each shard runs in its own fresh container (already at base_commit, so no git
        reset — validate just applies the candidate patch)."""
        import time
        # The `:xdist` image (when present) also carries pytest-timeout, so per-test
        # --timeout is only safe to pass for those instances.
        has_xd = self._has_xdist(instance_id, repo)
        pytest_n = n if has_xd else 1
        shards = self._shard_plan(repo, version, directives, n, pytest_n, timeout,
                                  has_pytest_timeout=has_xd)
        merged: dict[str, str] = {}
        apply_failed = False
        # Every runner yields exactly one shard (parallelism is container-side: django
        # --parallel, pytest -n, sympy backgrounded --split), so run the shard(s) in fresh
        # containers sequentially and merge — no thread pool / lock needed.
        for cmd, ids, wrap in shards:
            t0 = time.time()
            container = self._start_container(instance_id, repo)
            _record_container_setup(phase, instance_id, t0, time.time())
            try:
                if patch_text is not None and not self._apply_patch(container, patch_text):
                    apply_failed = True
                    continue
                log = self._run_tests(container, cmd, ids, instance_id, phase, wrap=wrap,
                                      timeout=timeout)
                merged.update(self._parse(repo, version, instance_id, log))
            finally:
                _td0 = time.time()
                container.remove(force=True)
                _record_container_setup(phase, instance_id, _td0, time.time(),
                                        kind="container_teardown")
        return merged, apply_failed

    def find_passing_tests(self, repo, version, instance_id, instance) -> list[str]:
        """Step 1: tests that pass in the pristine repo (whole-suite run, sharded)."""
        if self.universe == "pass_to_pass":
            # Fast (less fair): trust SWE-bench's curated pass-to-pass set as the universe.
            return json.loads(instance.get("PASS_TO_PASS", "[]"))
        # Faithful: run the whole suite and keep passers. directives=None -> the shard
        # plan uses the whole-suite parallelism for this runner (django --parallel /
        # sympy --split / pytest serial).
        status, _ = self._run_partitioned(repo, version, instance_id, "find_passing",
                                           directives=None, patch_text=None,
                                           n=self.test_parallel or 1)
        return [t for t, s in status.items() if s == TestStatus.PASSED.value]

    # Max passing-test names to put in ONE select prompt. The full list is processed in
    # batches of this size (per-batch selections unioned), so a large suite (django's
    # ~11.5k passing tests) never overflows the model context in a single call. Sized
    # well under a 128k-token window, leaving room for the issue text and the ~10%
    # keep-list output.
    _LLM_BATCH = 2000
    # Tests the LLM is asked to keep PER BATCH (the prompt's "{k}"). A concrete small
    # absolute target rather than a fraction: the model over-selects badly against a
    # "~10%" fraction (returned 300–1450 per 2000-name batch), so anchor on a small
    # count instead.
    _SELECT_TARGET = 50
    # Hard ceiling on the regression set: above this, a seeded sample brings it back
    # down so validate's label-selection path always applies. Keyed on instance_id so
    # every candidate of an instance validates against the SAME set.
    _SAMPLE_CAP = 1000

    @staticmethod
    def _addressable_names(repo: str, names: list[str]) -> list[str]:
        """Names that can become CLI labels for `repo` (validate's label path).
        django: id-shaped "method (module.Class)" only — drops docstring pseudo-entries
        (their real ids are ALSO in the list via parse_django_addressable's dual
        recording, so no coverage is lost)."""
        if repo == "django/django":
            return [n for n in names if _DJANGO_NAME_RE.match(n.strip())]
        return list(names)

    def select_regression_tests(self, issue: str, passing: list[str],
                                instance_id: str = "", repo: str = "") -> list[str]:
        """Step 2: LLM picks a ~10% REPRESENTATIVE subset of the passing tests.

        Selection framing (ask for a small keep-list): the regression set stays at
        ~_SELECT_FRAC of the suite, so validate runs it by label selection instead of
        re-running the whole suite per candidate. The old exclusion rule is folded into
        the prompt: tests asserting the issue's buggy behavior must NOT be selected.

        The tests are shown NUMBERED and the model returns indices: models reliably
        copy small integers but mangle long test names (django's "method (module.Class)"
        came back as bare method names, zeroing the selection), so name-matching is
        replaced by index lookup. Unaddressable pseudo-names are dropped from the shown
        universe up front so the selected set qualifies for the label path.

        The list is processed in context-safe batches of _LLM_BATCH (per-batch
        keep-lists unioned). A batch whose response is empty/garbage falls back to a
        seeded random sample of that batch, so no region of the suite is silently
        dropped. The union is capped at _SAMPLE_CAP by a seeded sample."""
        if not passing:
            return []
        universe = self._addressable_names(repo, passing)
        if len(universe) < len(passing):
            print(f"[regr] {instance_id}: dropped {len(passing) - len(universe)} "
                  f"unaddressable pseudo-name(s) from the selection universe")
        if not universe:
            return []
        shown = universe[: self.max_tests_for_llm]
        if len(universe) > self.max_tests_for_llm:
            print(f"[regr] {len(universe)} passing tests > cap {self.max_tests_for_llm}; "
                  f"considering the first {self.max_tests_for_llm}")
        selected: list[str] = []
        turns = []
        for start in range(0, len(shown), self._LLM_BATCH):
            batch = shown[start:start + self._LLM_BATCH]
            k = min(self._SELECT_TARGET, len(batch))
            numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(batch))
            prompt = SELECT_PROMPT.format(issue=issue, k=k, tests=numbered)
            messages = [LLMMessage(role="user", content=prompt)]
            resp = self.llm_client.chat(messages, self.model_config, None, reuse_history=False)
            turns.append((messages, resp))
            # In-range integer indices (dedup, keep order). Lenient: over-selection can
            # blow the max_tokens budget and truncate the array mid-token with no closing
            # `]`, which a strict json.loads rejects — so pull every complete integer from
            # the response and keep the in-range ones. The last (possibly half-written)
            # number is dropped by the range check.
            idxs = [i for i in (int(m) for m in re.findall(r"\d+", resp.content or ""))
                    if 0 <= i < len(batch)]
            picked = [batch[i] for i in dict.fromkeys(idxs)]
            if not picked:
                picked = random.Random(f"{instance_id}:{start}").sample(batch, k)
                print(f"[regr] {instance_id}: batch@{start} returned no usable selection; "
                      f"falling back to a seeded {k}-test sample")
            selected.extend(picked)
        self._record_turns(instance_id, turns)
        if len(selected) > self._SAMPLE_CAP:
            print(f"[regr] {instance_id}: {len(selected)} selected > cap "
                  f"{self._SAMPLE_CAP}; sampling down")
            selected = random.Random(instance_id).sample(selected, self._SAMPLE_CAP)
        return selected

    def _record_turns(self, instance_id: str, turns) -> None:
        """Record the regression LLM call(s) as turns in `llm_interactions[]`, matching
        the generate/selector trajectory format. One file per instance; batched selection
        makes >1 call, so record them all as consecutive turns.

        A fresh recorder here keeps this thread-safe across the instance-level
        ThreadPoolExecutor (the shared self.llm_client has no recorder attached)."""
        if not self.trajectory_dir or not turns:
            return
        try:
            provider = self.model_config.model_provider.provider
            path = Path(self.trajectory_dir) / f"{instance_id or 'unknown'}.json"
            recorder = TrajectoryRecorder(str(path))
            recorder.start_recording(
                task=f"regression test selection: {instance_id}",
                provider=provider,
                model=self.model_config.model,
                max_steps=len(turns),
            )
            last = None
            for messages, response in turns:
                recorder.record_llm_interaction(
                    messages, response, provider, self.model_config.model, None
                )
                last = response
            recorder.finalize_recording(True, last.content if last else "")
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

    # Above this many regression tests, don't pass them as a CLI label list — run the
    # whole suite and filter instead. select_regression_tests already samples the set
    # down to _SAMPLE_CAP, so this now matches it as a pure safety net (label lists of
    # this size are still far below ARG_MAX); only unaddressable names force the
    # whole-suite path in practice.
    _VALIDATE_LABEL_CAP = 1000

    @staticmethod
    def _num_addressable(repo: str, names: list[str]) -> int:
        """How many names can be turned into a valid CLI selector for `repo` (django needs
        the "method (module.Class)" shape — its parser also emits pseudo-entries like
        "--version is equivalent to version" that are NOT valid labels; pytest node ids and
        sympy "file.py::test" names are all addressable — sympy at test granularity)."""
        if repo == "django/django":
            return sum(1 for n in names if _DJANGO_NAME_RE.match(n.strip()))
        return len(names)

    def validate_candidate(self, repo, version, instance_id, regression_tests,
                           patch_text) -> list[str]:
        """Step 3: failed regression tests for one candidate patch.

        Runs by label-selection (only the chosen tests) when the set is small AND every
        name is addressable; otherwise runs the WHOLE suite and filters by re-parsing.
        Whole-suite is mandatory when a name can't be addressed — e.g. sympy's bare names,
        or a django parser pseudo-entry ("--version is equivalent to version"): one such
        token as a label aborts the entire runtests invocation, yielding zero results and
        spuriously failing every candidate. It is also no more expensive than selection
        once the regression set approaches the whole suite (the uncapped case)."""
        if not regression_tests:
            return []
        directives = test_directives(repo, regression_tests)
        whole_suite = (
            directives is None
            or self._num_addressable(repo, regression_tests) < len(regression_tests)
            or len(regression_tests) > self._VALIDATE_LABEL_CAP
        )
        status, apply_failed = self._run_partitioned(
            repo, version, instance_id, "validate",
            directives=None if whole_suite else directives,
            patch_text=patch_text, n=self.test_parallel or 1)
        if apply_failed:
            # Unappliable patch -> treat as failing everything (will be pruned).
            return list(regression_tests)
        # A regression test must PASS; anything else (fail/error/missing) is a failure.
        # Whole-suite re-parses every test (artifacts re-emitted as PASSED, exactly as
        # find_passing saw them), so a missing status genuinely means not-passed.
        failed = [t for t in regression_tests if status.get(t) != TestStatus.PASSED.value]
        # Strip flaky-under-load noise: re-run a MODEST failure set at LOW parallelism once
        # (see _retry_failures). A test that passes with little parallel contention was a
        # timing/port/state/order flake, not a real regression. A huge failure set is real
        # breakage (the patch), so skip it.
        if 0 < len(failed) <= self._RETRY_CAP:
            failed = self._retry_failures(repo, version, instance_id, failed, patch_text)
        return failed

    _RETRY_CAP = 100
    # Retry at LOW (not zero) parallelism. The flakiness came from extreme 96-way
    # contention; a small degree leaves each test ample headroom (so it still un-flakes)
    # while being ~N faster than serial — important for sympy, whose retry re-runs whole
    # slow FILES (e.g. test_integrals.py), which serial would drag out. Scale the retry
    # degree as a fixed 1/8 fraction of the main parallelism (not a hard-coded count) so it
    # tracks the configured `test_parallel` — 96 -> 12, 8 -> 1 — with a floor of 1.
    _RETRY_DIVISOR = 8

    def _retry_failures(self, repo, version, instance_id, failed, patch_text) -> list[str]:
        """Re-run the failed tests at LOW parallelism (no meaningful contention) once, and
        keep only those that STILL don't pass. Distinguishes genuine regressions (fail even
        with headroom) from flaky-under-load tests (pass once the 96-way contention is
        gone)."""
        # sympy: its "failures" are RecursionErrors from CROSS-TEST STATE POLLUTION in a
        # shared bin/test process (verified: every such test passes when run alone). A
        # low-parallelism re-run of the whole FILE still reproduces the pollution, so retry
        # each failed test in its OWN fresh process instead (see below).
        if repo == "sympy/sympy":
            return self._retry_sympy_isolated(instance_id, failed, patch_text)
        directives = test_directives(repo, failed)
        if not directives:
            return failed  # nothing addressable to re-run
        # Same addressability guard as validate_candidate: if any name can't be turned into a
        # valid CLI label (e.g. a django parser pseudo-entry like "--version is equivalent to
        # version"), passing it as a runtests label aborts the ENTIRE invocation, so the retry
        # would parse zero results and spuriously keep every test failed. Fall back to the
        # whole suite + re-parse filter, exactly as validate does.
        whole_suite = self._num_addressable(repo, failed) < len(failed)
        n = max(1, (self.test_parallel or 1) // self._RETRY_DIVISOR)
        # No time limit on the retry (`timeout 0` = unlimited): the failure set is small and
        # low-parallelism, so it must run to completion to give a definitive verdict. A cap
        # here would re-introduce the very timeout-boundary flakiness the retry exists to
        # remove (a co-located slow file eats the budget before the flaky test is reached).
        status, apply_failed = self._run_partitioned(
            repo, version, instance_id, "retry",
            directives=None if whole_suite else directives,
            patch_text=patch_text, n=n, timeout=0)
        if apply_failed:
            return failed
        return [t for t in failed if status.get(t) != TestStatus.PASSED.value]

    _baseline_lock = threading.Lock()

    def _retry_sympy_isolated(self, instance_id, failed, patch_text) -> list[str]:
        """Decide sympy regressions DIFFERENTIALLY under a per-test-isolated, fixed-seed
        re-run: a test is a regression iff it FAILS with the patch but PASSES on the pristine
        repo, both isolated. This cancels the three things that otherwise make a test fail
        validate with NO patch to blame:
          (a) run-to-run flakiness — sympy re-seeds `random` freshly each run, so pin it;
          (b) cross-test global-state pollution — give each test its own fresh interpreter;
          (c) find_passing's whole-suite PASS disagreeing with an isolated re-run on the
              PRISTINE code (a test that only passes in a lucky suite arrangement, e.g.
              test_heurisch::test_issue_3609). Such a test fails the pristine baseline too,
              so the differential drops it instead of blaming the patch."""
        patched = self._sympy_isolated_status(instance_id, failed, patch_text)
        if patched is None:
            return failed  # patch didn't apply -> conservatively fail all (will be pruned)
        # A regression can only survive if it FAILS under the patch, so only those need a
        # pristine baseline; tests that un-flaked under the patch (the common case) are
        # dropped regardless of pristine, so skip computing it for them (and skip the
        # pristine container entirely when nothing is still failing).
        still_failing = [t for t in failed if not patched.get(t, False)]
        if not still_failing:
            return []
        pristine = self._sympy_pristine_isolated(instance_id, still_failing)
        return [t for t in still_failing if pristine.get(t, False)]

    def _sympy_isolated_status(self, instance_id, tests, patch_text) -> dict | None:
        """Run each sympy test in its OWN fresh interpreter with seed pinned, returning
        {test: passed_bool}. The fresh process removes cross-test state pollution; the pinned
        seed (+ inherited PYTHONHASHSEED=0) removes randomized-test flakiness. Reuses the
        shared `_SYMPY_EXACT` runner with a one-element want set and `--split 1/1` (a single
        file, un-split). Returns None if the patch does not apply."""
        container = self._start_container(instance_id, "sympy/sympy")
        try:
            if patch_text is not None and not self._apply_patch(container, patch_text):
                return None
            # Each test in its own interpreter (isolation is the point), but run up to `par`
            # at a time instead of strictly serial — a large failure set would otherwise cost
            # len(tests) * up-to-300s. Bounded container-side concurrency (background each,
            # `wait` every `par` jobs), mirroring _sympy_split_loop; par tracks test_parallel.
            par = max(1, (self.test_parallel or 1) // self._RETRY_DIVISOR)
            parts = []
            for idx, name in enumerate(tests):
                file, _, func = name.partition("::")
                # Own interpreter per test; generous per-test timeout guards a hang.
                job = (
                    f"if timeout 300 env SYMPY_USE_CACHE=no SYMPY_WANT='{func}' "
                    f"SYMPY_SEED={self._SYMPY_SEED} python -c '{self._SYMPY_EXACT}' {file} 1/1 "
                    f'>/dev/null 2>&1; then echo "RPASS::{name}"; else echo "RFAIL::{name}"; fi')
                parts.append(f"{{ {job} ; }} &")
                if (idx + 1) % par == 0:
                    parts.append("wait")
            parts.append("wait")
            # Newline-join: `{ …; } &` and a bare `wait` need a real command separator
            # between them — a space gives `wait { …` which bash rejects (`wait` is a simple
            # command, so `{` is a literal arg, not a group). Newlines separate cleanly.
            out = self._run_tests(container, "\n".join(parts), None, instance_id, "retry",
                                  wrap=False, timeout=0)
            passed = {
                line[len("RPASS::") :]
                for line in out.splitlines()
                if line.startswith("RPASS::")
            }
            return {t: (t in passed) for t in tests}
        finally:
            container.remove(force=True)

    def _sympy_pristine_isolated(self, instance_id, tests) -> dict:
        """Pristine {test: passed_bool} under the isolated fixed-seed method, cached per
        instance (patch-independent, so it is reused across every candidate)."""
        with self._baseline_lock:
            if not hasattr(self, "_baseline_cache"):
                self._baseline_cache = {}
            cache = self._baseline_cache.setdefault(instance_id, {})
            todo = [t for t in tests if t not in cache]
        if todo:
            res = self._sympy_isolated_status(instance_id, todo, None) or {}
            with self._baseline_lock:
                for t in todo:
                    cache[t] = res.get(t, False)
        with self._baseline_lock:
            return {t: cache[t] for t in tests}

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

        # Containers are now created per shard inside _run_partitioned (container-per-
        # shard parallelism), so there is no instance-level container to manage here.
        passing = self.find_passing_tests(repo, version, instance_id, instance)
        regression = self.select_regression_tests(entry.get("issue", ""), passing,
                                                  instance_id, repo=repo)

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
                repo, version, instance_id, regression, patches[i]
            )
            rep_failed[i] = failed
            print(f"[regr]   {instance_id} cand {i}: "
                  f"{'PASS' if not failed else f'{len(failed)} failed'}")

        # Propagate each representative's result to its duplicates.
        entry["regressions"] = [rep_failed[rep_of[keys[i]]] for i in range(len(patches))]
        entry["regression_done"] = True
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
                   help="Per test-RUN timeout (seconds) inside the container, so a hanging "
                        "test cannot block the stage (kills the whole batched run).")
    p.add_argument("--per-test-timeout", type=int, default=0,
                   help="Per single-TEST timeout (seconds), enforced inside the runner "
                        "(sympy/pytest --timeout). A slow test is marked failed on its own "
                        "while the rest of the run continues; it then drops out of the "
                        "passing set and never reaches validate. 0 disables. django has no "
                        "runner support and falls back to --test-timeout.")
    p.add_argument("--trajectory-dir", default=None,
                   help="Dir for per-instance trajectory files recording the regression "
                        "LLM call as a turn (normalized with generate/selector). Defaults to "
                        "<output_dir>/regression_trajectories; pass '' to disable.")
    p.add_argument("--max_workers", type=int, default=4, help="Parallel workers across instances.")
    p.add_argument("--test-parallel", type=int, default=None,
                   help="Unified parallelism degree n: partition each test run into n "
                        "shards, each in its own container (works for every repo — id "
                        "sharding for django/pytest, 'bin/test --split' for sympy, native "
                        "--parallel for django's whole-suite discovery). None/1 = serial.")
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
        per_test_timeout=args.per_test_timeout,
        trajectory_dir=args.trajectory_dir,
        test_parallel=args.test_parallel,
    )
    tester.run(args.candidates, args.output, args.max_workers)


if __name__ == "__main__":
    main()
