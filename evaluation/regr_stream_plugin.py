# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""The `regr_stream` pytest plugin baked into `:xdist` images, plus a helper to
install it into a container.

Why: the regression tester's `find_passing` runs the WHOLE suite under a wall-clock
`timeout`. pytest's `-rA` PASSED/FAILED summary — the lines the SWE-bench log parser
reads — is emitted only AT THE END of the run, so when a heavy suite (e.g.
scikit-learn) is SIGKILLed on timeout before it finishes, the log has ZERO status
lines and `find_passing` returns an EMPTY passing set. That silently zeroes the whole
prune stage: `select_regression_tests` early-returns on an empty set (no LLM call), and
validate is skipped, so no candidate is ever pruned.

The plugin streams one `"<STATUS> <nodeid>"` line to stdout AS EACH TEST COMPLETES (via
the `pytest_runtest_logreport` hook, which under `-n N` fires on the xdist CONTROLLER as
each worker reports back). Those lines are the exact format `parse_log_pytest_v2` already
reads, so a killed run still yields every test that passed before the kill with NO parser
or harvest change — the partial stdout docker returns on SIGKILL is enough.

Loaded via `-p regr_stream` (see `_pytest_cmd`), gated on the `:xdist` image being
present (the plugin is baked in alongside pytest-xdist / pytest-timeout).
"""

# The plugin module written verbatim into the container's site-packages. Kept as a
# string (not just this file) so build_xdist_images / backfill can bake it into images
# without copying a file in. `report.when == "call"` is the pass/fail phase; a
# setup-phase skip/error never reaches "call", so surface those too. Flush every line so
# nothing is lost in stdio buffers when the process is SIGKILLed mid-run.
REGR_STREAM_SRC = '''\
import sys


def pytest_runtest_logreport(report):
    if report.when == "call":
        status = {"passed": "PASSED", "failed": "FAILED"}.get(report.outcome)
    elif report.when == "setup" and report.outcome in ("skipped", "failed"):
        status = "SKIPPED" if report.outcome == "skipped" else "ERROR"
    else:
        return
    if status:
        # LEADING newline: pytest's terminal reporter writes per-test progress ("." / an
        # xdist "[gw0] [ 12%] ...") with NO trailing newline, so a bare write would land as
        # ".PASSED <nodeid>" on the same line and fail the parser's startswith() check.
        # Starting on a fresh line makes "<STATUS> <nodeid>" the first token of its line.
        sys.stdout.write(f"\\n{status} {report.nodeid}\\n")
        sys.stdout.flush()
'''

MODULE_NAME = "regr_stream"


def install_regr_stream(exec_fn) -> tuple[bool, str]:
    """Write the plugin into the testbed env's site-packages and verify it imports.

    `exec_fn(cmd)` runs `cmd` in the target container under the testbed env and returns
    (rc, output) — the same `_exec` both build_xdist_images and backfill already use.
    Idempotent: rewrites the file (cheap) and re-verifies. Returns (ok, short_message).
    The source is shipped base64-encoded so no quoting survives the bash round-trip.
    """
    import base64
    b64 = base64.b64encode(REGR_STREAM_SRC.encode()).decode()
    # Land it next to the other testbed site-packages so `-p regr_stream` imports it.
    write = (
        "python - <<'PY'\n"
        "import base64, os, sysconfig\n"
        f"src = base64.b64decode({b64!r}).decode()\n"
        "dst = os.path.join(sysconfig.get_paths()['purelib'], 'regr_stream.py')\n"
        "open(dst, 'w').write(src)\n"
        "print(dst)\n"
        "PY"
    )
    rc, out = exec_fn(write)
    if rc != 0:
        return False, f"write-failed:{out.strip().splitlines()[-1][:80] if out.strip() else ''}"
    rc_imp, _ = exec_fn("python -c 'import regr_stream'")
    if rc_imp != 0:
        return False, "import-failed"
    return True, "ok"
