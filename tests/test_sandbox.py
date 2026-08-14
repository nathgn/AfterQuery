# tests/test_sandbox.py
#
# Regression tests for sandbox execution. Runs with pytest or directly:
#   python tests/test_sandbox.py
#
# These cover three bugs that made a GRPO run either 4x slower or liable to
# crash outright, all of which were silent before:
#   1. Unbounded output was buffered in memory, so `yes` could OOM the trainer.
#   2. Timeouts killed only the shell, orphaning pipeline children.
#   3. Blocking forms like `tail -f` escaped the bare-command blocklist and
#      burned the full timeout on every occurrence.

import sys
import time

sys.path.insert(0, ".")

from envs.tasks import get_all_tasks
from envs.terminalbench_client import TerminalBenchClient

MAX_OUT = 2000


def _client_and_task():
    spec = [t for t in get_all_tasks() if t.task_id == "med_grep_errors"][0]
    client = TerminalBenchClient(task_specs=[spec], max_output_length=MAX_OUT)
    return client, client.sample_task()


def _run(cmd):
    client, task = _client_and_task()
    try:
        started = time.time()
        out = client._execute_in_sandbox(task.workdir, cmd)
        return out, time.time() - started
    finally:
        client.cleanup(task)


def test_normal_command_works():
    out, _ = _run("grep ERROR app.log")
    assert "ERROR disk full" in out, out
    assert "INFO" not in out, out


def test_unbounded_output_is_capped_and_fast():
    """`yes` used to allocate GBs via communicate() before the timeout fired."""
    out, elapsed = _run("yes")
    assert len(out) <= MAX_OUT, f"output not capped: {len(out)} bytes"
    assert elapsed < 5.0, f"took {elapsed:.1f}s; head -c should SIGPIPE the producer"


def test_binary_output_does_not_raise():
    out, elapsed = _run("cat /dev/urandom")
    assert len(out) <= MAX_OUT
    assert "execution error" not in out, out
    assert elapsed < 5.0, f"took {elapsed:.1f}s"


def test_blocking_forms_are_rejected_immediately():
    """Each of these would otherwise burn the full command_timeout."""
    for cmd in ["tail -f app.log",
                "tail -f app.log | grep ERROR",
                "watch ls",
                "ping example.com",
                "sleep 999"]:
        out, elapsed = _run(cmd)
        assert "blocked" in out, f"{cmd!r} not blocked: {out!r}"
        assert elapsed < 2.0, f"{cmd!r} took {elapsed:.1f}s"


def test_ping_with_count_is_allowed():
    """The blocklist must not reject terminating forms."""
    client, task = _client_and_task()
    try:
        out = client._execute_in_sandbox(task.workdir, "echo ping -c 1 done")
        assert "blocked" not in out, out
    finally:
        client.cleanup(task)


def test_bare_interactive_still_blocked_but_not_with_args():
    client, task = _client_and_task()
    try:
        assert "blocked" in client._execute_in_sandbox(task.workdir, "bash")
        # "bash script.sh" is a legitimate, terminating invocation
        out = client._execute_in_sandbox(task.workdir, "echo 'echo hi' > s.sh && bash s.sh")
        assert "hi" in out, out
    finally:
        client.cleanup(task)


def test_timeout_returns_marker():
    client, task = _client_and_task()
    client.command_timeout = 2
    try:
        started = time.time()
        # A busy loop no pattern matches, so it must hit the timeout and be
        # killed by process group (the shell alone would leave it spinning).
        out = client._execute_in_sandbox(task.workdir, "while true; do :; done")
        elapsed = time.time() - started
        assert "timed out" in out, out
        assert elapsed < 10.0, f"timeout not enforced promptly ({elapsed:.1f}s)"
    finally:
        client.cleanup(task)


def test_stderr_is_captured():
    out, _ = _run("ls /definitely/not/here")
    assert out.strip(), "stderr was lost"


def test_stdin_is_not_inherited():
    """A stdin reader must return immediately, not block on the parent's stdin."""
    out, elapsed = _run("cat")
    assert elapsed < 2.0, f"`cat` blocked for {elapsed:.1f}s; stdin should be DEVNULL"
    assert "timed out" not in out, out


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    sys.exit(1 if failed else 0)
