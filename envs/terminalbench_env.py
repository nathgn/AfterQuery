"""
RLVR environment for terminalbench2.
Exposes reset() and step() operating on strings.
"""

from typing import Any, Dict, Tuple
from envs.terminalbench_client import TerminalBenchClient, TaskState


class TerminalBenchEnv:
    def __init__(
        self,
        tb_client: TerminalBenchClient,
        max_steps: int = 20,
        step_penalty: float = 0.01,
        w_success: float = 1.0,
        w_eff: float = 0.1,
        w_quality: float = 0.05,
    ):
        self.tb_client = tb_client
        self.max_steps = max_steps
        self.step_penalty = step_penalty
        self.w_success = w_success
        self.w_eff = w_eff
        self.w_quality = w_quality
        self.task: TaskState | None = None
        self.step_count: int = 0
        self.done: bool = False

    def reset(self) -> str:
        self.task = self.tb_client.sample_task()
        self.task = self.tb_client.reset_task(self.task)
        self.step_count = 0
        self.done = False
        return self._build_prompt()

    def step(self, action_text: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("step() called on terminated episode")
        self.step_count += 1
        self.task = self.tb_client.execute(self.task, action_text)

        success_score = self.task.score
        done = self.task.done or self.step_count >= self.max_steps
        success_reward = success_score if done else 0.0
        efficiency_reward = -self.step_penalty
        quality_reward = self._compute_quality_reward(
            action_text, self.task.terminal_output
        )

        reward = (
            self.w_success * success_reward
            + self.w_eff * efficiency_reward
            + self.w_quality * quality_reward
        )

        obs = self._build_prompt()
        self.done = done

        info = {
            "success_score": success_score,
            "step_count": self.step_count,
            "success_reward": success_reward,
            "efficiency_reward": efficiency_reward,
            "quality_reward": quality_reward,
            "task_id": self.task.task_id,
            "difficulty": self.task.difficulty,
            "domain": self.task.domain,
        }
        return obs, reward, done, info

    def _build_prompt(self) -> str:
        assert self.task is not None
        return self.tb_client.render_prompt(
            self.task, self.step_count, self.max_steps
        )

    def _compute_quality_reward(
        self, command: str, terminal_output: str
    ) -> float:
        penalty = 0.0
        out = terminal_output.lower()
        cmd = command.strip().lower()

        if "command not found" in out:
            penalty -= 1.0
        if "permission denied" in out:
            penalty -= 0.5
        if "syntax error" in out or "syntaxerror" in out:
            penalty -= 0.3

        dangerous_patterns = [
            "rm -rf /", "rm -rf /*", "mkfs", "dd if=/dev/zero",
            ":(){:|:&};:", "chmod -R 777 /",
        ]
        for pattern in dangerous_patterns:
            if pattern in cmd:
                penalty -= 2.0

        if not cmd:
            penalty -= 0.5

        if penalty == 0.0 and out and "error" not in out:
            penalty += 0.01

        return penalty
