# envs/terminalbench_client.py

import os
import random
import shutil
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

        wrapped = f'cd "{posix_wd}" && {command}'
        env = os.environ.copy()
        env["HOME"] = workdir
        env["TERM"] = "dumb"
        try:
            proc = subprocess.Popen(
                [_BASH, "-c", wrapped],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            try:
                stdout, stderr = proc.communicate(timeout=self.command_timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.communicate(timeout=5)
                except (subprocess.TimeoutExpired, OSError):
                    pass
                return "[command timed out]"
            output = stdout
            if stderr:
                output = output + stderr if output else stderr
        except Exception as e:
            output = f"[execution error: {e}]"

        return output[: self.max_output_length]

    def _get_spec(self, task_id: str) -> TaskSpec:
        for spec in self.task_specs:
            if spec.task_id == task_id:
                return spec
        raise ValueError(f"Unknown task_id: {task_id}")

    def cleanup(self, task: TaskState) -> None:
        wd = self._active_workdirs.pop(task.instance_id, None)
        if wd:
            wd.cleanup()

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
