"""
Stub client for terminalbench2.

In production, replace with real Harbor integration:
  pip install harbor
  harbor datasets list
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import random


@dataclass
class TaskState:
    task_id: str
    difficulty: str
    description: str
    domain: str = "general"
    history: List[Tuple[str, str]] = field(default_factory=list)
    terminal_output: str = ""
    done: bool = False
    score: float = 0.0


class TerminalBenchClient:
    def __init__(self, task_subset: Optional[List[str]] = None):
        self.tasks = self._load_tasks()
        if task_subset is not None:
            self.tasks = {k: v for k, v in self.tasks.items() if k in task_subset}
        self.task_ids = list(self.tasks.keys())

    def _load_tasks(self) -> Dict[str, Dict[str, Any]]:
        return {
            "tb2_001": {
                "difficulty": "easy",
                "domain": "system_admin",
                "description": (
                    "Create a file named foo.txt in the current directory "
                    "containing exactly the text 'hello world'."
                ),
            },
            "tb2_002": {
                "difficulty": "easy",
                "domain": "data_processing",
                "description": (
                    "List all .py files recursively in the current directory "
                    "and write the count to a file named count.txt."
                ),
            },
            "tb2_003": {
                "difficulty": "medium",
                "domain": "debugging",
                "description": (
                    "The file buggy.py has a syntax error on line 15. "
                    "Find and fix the error so that 'python3 buggy.py' "
                    "runs without errors and prints 'OK'."
                ),
            },
            "tb2_004": {
                "difficulty": "medium",
                "domain": "software_engineering",
                "description": (
                    "Write a bash script called deploy.sh that: "
                    "(1) creates a virtual environment in ./venv, "
                    "(2) installs requirements from requirements.txt, "
                    "(3) runs pytest, and (4) prints 'DEPLOY OK' if tests pass."
                ),
            },
            "tb2_005": {
                "difficulty": "hard",
                "domain": "security",
                "description": (
                    "The web server in ./server/ has a known path traversal "
                    "vulnerability. Patch the vulnerability in server.py so "
                    "that requests for files outside the document root are "
                    "rejected with a 403 status code. Verify by running "
                    "the test suite in ./tests/."
                ),
            },
        }

    def sample_task(self) -> TaskState:
        task_id = random.choice(self.task_ids)
        meta = self.tasks[task_id]
        return TaskState(
            task_id=task_id,
            difficulty=meta["difficulty"],
            domain=meta.get("domain", "general"),
            description=meta["description"],
        )

    def reset_task(self, task: TaskState) -> TaskState:
        task.history.clear()
        task.terminal_output = ""
        task.done = False
        task.score = 0.0
        return task

    def execute(self, task: TaskState, command: str) -> TaskState:
        output = self._simulate_command(task, command)
        task.history.append((command, output))
        task.terminal_output = output
        task.score = self.compute_score(task)
        task.done = self.is_task_done(task)
        return task

    def compute_score(self, task: TaskState) -> float:
        if task.task_id == "tb2_001":
            for cmd, _ in task.history:
                if "foo.txt" in cmd.lower() and "hello world" in cmd.lower():
                    return 1.0
            return 0.0
        elif task.task_id == "tb2_002":
            for cmd, _ in task.history:
                if "find" in cmd and ".py" in cmd and "count.txt" in cmd:
                    return 1.0
            return 0.0
        elif task.task_id == "tb2_003":
            has_fix = any(
                "sed" in c or "vim" in c or "nano" in c
                for c, _ in task.history
            )
            has_run = any("python3 buggy.py" in c for c, _ in task.history)
            if has_fix and has_run:
                return 1.0
            elif has_fix:
                return 0.5
            return 0.0
        return 0.0

    def is_task_done(self, task: TaskState) -> bool:
        return task.score >= 1.0

    def render_prompt(
        self, task: TaskState, step_index: int, max_steps: int
    ) -> str:
        history_str = ""
        for i, (cmd, out) in enumerate(task.history, start=1):
            history_str += f"{i}> {cmd}\nOUT: {out}\n"

        return f"""[terminalbench2]
Task ID: {task.task_id}
Difficulty: {task.difficulty}
Domain: {task.domain}
Step: {step_index}/{max_steps}

Task:
{task.description}

Current terminal output:
{task.terminal_output}

Command history:
{history_str}
You are a CLI assistant. Issue the next command to progress toward completing the task.
Next command:"""

    def _simulate_command(self, task: TaskState, command: str) -> str:
        cmd = command.strip().lower()
        if not cmd:
            return "Error: empty command"
        if cmd.startswith("echo") and ">" in cmd:
            filename = cmd.split(">")[-1].strip()
            return f"(wrote to {filename})"
        if cmd.startswith("find"):
            return "./main.py\n./utils.py\n./test_main.py\n3 files found"
        if cmd.startswith("cat"):
            return "(file contents displayed)"
        if "python3" in cmd:
            if task.task_id == "tb2_003":
                has_fix = any(
                    "sed" in c or "vim" in c or "nano" in c
                    for c, _ in task.history
                )
                return "OK" if has_fix else "SyntaxError: unexpected indent (line 15)"
            return "(python output)"
        if cmd.startswith("ls"):
            return "foo.txt  bar.py  README.md"
        if cmd.startswith("sed") or cmd.startswith("vim") or cmd.startswith("nano"):
            return "(file edited)"
        if "not_a_command" in cmd or "asdfgh" in cmd:
            return "bash: command not found"
        if cmd.startswith("sudo") and "rm" in cmd:
            return "permission denied"
        return f"(executed: {command.strip()[:50]})"
