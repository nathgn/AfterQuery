# envs/terminalbench_client.py

import os
import random
import re
import shutil
import signal
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
from uuid import uuid4

from envs.tasks import TaskSpec, get_all_tasks


# Find Git Bash on Windows; fall back to plain "bash"
_BASH = shutil.which("bash")
if _BASH is None:
    for candidate in [
        r"C:\Program Files\Git\usr\bin\bash.exe",
        r"C:\Program Files (x86)\Git\usr\bin\bash.exe",
    ]:
        if os.path.isfile(candidate):
            _BASH = candidate
            break
if _BASH is None:
    _BASH = "bash"


def _win_to_posix(path: str) -> str:
    """Convert a Windows path to a POSIX path for Git Bash."""
    path = path.replace("\\", "/")
    if len(path) >= 2 and path[1] == ":":
        drive = path[0].lower()
        path = f"/{drive}{path[2:]}"
    return path


@dataclass
class TaskState:
    task_id: str
    instance_id: str
    difficulty: str
    description: str
    workdir: str
    history: List[Tuple[str, str]] = field(default_factory=list)
    terminal_output: str = ""
    done: bool = False
    score: float = 0.0  # normalized 0-1


class TerminalBenchClient:
    """
    Real client for terminalbench tasks.

    Each task runs in an isolated temp directory. Commands are executed
    via Git Bash subprocess.
    """

    def __init__(
        self,
        task_specs: list[TaskSpec] | None = None,
        command_timeout: int = 10,
        max_output_length: int = 2000,
    ):
        self.task_specs = task_specs or get_all_tasks()
        self.command_timeout = command_timeout
        self.max_output_length = max_output_length
        self._active_workdirs: Dict[str, tempfile.TemporaryDirectory] = {}

    def sample_task(self) -> TaskState:
        spec = random.choice(self.task_specs)
        workdir = tempfile.TemporaryDirectory(prefix=f"tb_{spec.task_id}_")
        spec.setup(workdir.name)
        instance_id = f"{spec.task_id}_{uuid4().hex[:8]}"
        self._active_workdirs[instance_id] = workdir
        return TaskState(
            task_id=spec.task_id,
            instance_id=instance_id,
            difficulty=spec.difficulty,
            description=spec.description,
            workdir=workdir.name,
        )

    def reset_task(self, task: TaskState) -> TaskState:
        task.history.clear()
        task.terminal_output = ""
        task.done = False
        task.score = 0.0
        return task

    def execute(self, task: TaskState, command: str) -> TaskState:
        output = self._execute_in_sandbox(task.workdir, command)
        task.history.append((command, output))
        task.terminal_output = output
        task.score = self._get_spec(task.task_id).verify(task.workdir)
        task.done = task.score >= 1.0
        return task

    # Commands that open interactive sessions and will hang forever
    _BLOCKED_COMMANDS = frozenset([
        "bash", "sh", "zsh", "fish", "csh", "dash",
        "python", "python3", "python2", "node", "irb", "ruby",
        "vi", "vim", "nvim", "nano", "emacs", "less", "more", "man",
        "top", "htop", "ssh", "telnet", "ftp", "mysql", "psql",
    ])

    # Forms that block forever *with* arguments, so the bare-command check
    # above never catches them. Each one burns the full timeout and can never
    # complete a benchmark task, so reject them immediately instead: during
    # training these dominate wall-clock (a weak policy emits them often).
    _BLOCKING_PATTERNS = (
        re.compile(r"\btail\s+(-\S*f|--follow)", re.I),
        re.compile(r"\bwatch\b", re.I),
        re.compile(r"\bjournalctl\s+.*(-\S*f|--follow)", re.I),
        re.compile(r"\bping\b(?!.*\s-c\b)", re.I),
        re.compile(r"\bsleep\s+(\d{3,}|infinity)", re.I),
        re.compile(r"\b(nc|netcat)\s+-\S*l", re.I),
    )

    def _execute_in_sandbox(self, workdir: str, command: str) -> str:
        posix_wd = _win_to_posix(workdir)
        # Sanitize: only take the first line, strip leading $ prompts, limit length
        command = command.split("\n")[0].strip()[:500]
        command = command.lstrip("$ ").strip()
        if not command:
            return "[empty command]"

        # Block interactive commands (bare invocation only, not "bash script.sh")
        parts = command.split()
        base_cmd = parts[0].split("/")[-1] if parts else ""
        is_bare = len(parts) == 1
        if base_cmd in self._BLOCKED_COMMANDS and is_bare:
            return f"[blocked: '{base_cmd}' is interactive and not allowed]"

        for pat in self._BLOCKING_PATTERNS:
            if pat.search(command):
                return "[blocked: command would block until timeout]"

        # Truncate inside the shell rather than in Python. proc.communicate()
        # buffers everything in memory, so an unbounded producer like `yes` or
        # `cat /dev/urandom` allocates gigabytes before the timeout can fire --
        # enough to get the training process OOM-killed. Piping through
        # `head -c` bounds memory and SIGPIPEs the producer, so those commands
        # now finish instantly instead of running until the timeout.
        # `head` masks the exit status, which is fine: only output is used.
        limit = self.max_output_length
        wrapped = f'cd "{posix_wd}" && {{ {command} ; }} 2>&1 | head -c {limit}'
        env = os.environ.copy()
        env["HOME"] = workdir
        env["TERM"] = "dumb"
        try:
            proc = subprocess.Popen(
                [_BASH, "-c", wrapped],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                # Binary output (e.g. `cat /dev/urandom`) must degrade to
                # replacement chars, not raise a UnicodeDecodeError.
                errors="replace",
                env=env,
                # Own process group so a timeout kills the whole pipeline.
                # proc.kill() alone leaves children (e.g. the producer feeding
                # a pipe) orphaned and running for the rest of the session.
                start_new_session=True,
            )
            try:
                output, _ = proc.communicate(timeout=self.command_timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    proc.kill()
                try:
                    proc.communicate(timeout=5)
                except (subprocess.TimeoutExpired, OSError):
                    pass
                return "[command timed out]"
        except Exception as e:
            output = f"[execution error: {e}]"

        return (output or "")[: self.max_output_length]

    def _get_spec(self, task_id: str) -> TaskSpec:
        for spec in self.task_specs:
            if spec.task_id == task_id:
                return spec
        raise ValueError(f"Unknown task_id: {task_id}")

    def cleanup(self, task: TaskState) -> None:
        wd = self._active_workdirs.pop(task.instance_id, None)
        if wd:
            # On Windows, bash subprocesses may still hold file handles
            # briefly after proc.communicate() returns. Retry with a
            # small delay, then give up silently.
            import time
            for attempt in range(3):
                try:
                    shutil.rmtree(wd.name, ignore_errors=False)
                    wd._finalizer.detach()  # prevent double-cleanup
                    return
                except PermissionError:
                    time.sleep(0.2)
            # Final attempt: just ignore errors
            shutil.rmtree(wd.name, ignore_errors=True)
            wd._finalizer.detach()

    def render_prompt(self, task: TaskState, step_index: int, max_steps: int) -> str:
        history_str = ""
        for i, (cmd, out) in enumerate(task.history, start=1):
            history_str += f"$ {cmd}\n{out}\n"

        prompt = f"""[terminalbench]
Task: {task.description}
Difficulty: {task.difficulty}
Step: {step_index}/{max_steps}

Terminal history:
{history_str}
You are a CLI assistant. Issue the next bash command to complete the task.
Command:"""
        return prompt
