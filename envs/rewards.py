# envs/rewards.py
#
# Shared command extraction and reward computation for RLVR training.
# Used by scripts/train_rlvr.py, scripts/evaluate_terminalbench.py, and the
# colab/ notebooks (which clone this repo). Keeping this logic in one place
# prevents the training notebooks from drifting out of sync with the scripts.

import re

from envs.tasks import get_all_tasks
from envs.terminalbench_client import TerminalBenchClient
from envs.terminalbench_env import TerminalBenchEnv


def extract_command(text: str) -> str:
    """Extract a single shell command from model output.

    The model often generates markdown, explanations, and code blocks.
    This tries to pull out just the command.
    """
    text = text.strip()
    if not text:
        return "echo noop"

    # Try to find a ```bash ... ``` code block
    m = re.search(r"```(?:bash|sh)?\s*\n(.+?)```", text, re.DOTALL)
    if m:
        for line in m.group(1).splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line

    # Otherwise take the first non-empty, non-comment, non-prose line
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Skip lines that look like English prose
        if re.match(r"^[A-Z][a-z].*\s(the|a|an|is|to|you|can|this)\s", line):
            continue
        return line

    # Fallback: first line
    return text.splitlines()[0].strip() or "echo noop"


def make_reward_func(
    command_timeout: int = 10,
    max_output_length: int = 2000,
    step_penalty: float = 0.01,
    w_success: float = 1.0,
    w_eff: float = 0.1,
    w_quality: float = 0.05,
):
    """Create a TRL-compatible reward function for terminalbench tasks.

    Two properties matter for a correct training signal:

    1. Task matching: the prompt text is parsed to recover the task
       description and matched to its TaskSpec, so the command is scored
       against the task the model was actually responding to (an env with
       a random `reset()` would score against an unrelated task ~24/25
       of the time).
    2. Partial credit: the scoring env uses max_steps=1 so the episode is
       terminal after the single command, letting graded verify() scores
       (0.3, 0.5, 0.7, ...) flow into the reward instead of being gated
       behind score >= 1.0.

    Each call creates a fresh client/env so reward computation is isolated.
    """
    all_tasks = get_all_tasks()
    task_by_desc = {t.description: t for t in all_tasks}

    def _match_task(prompt_text: str):
        for desc, spec in task_by_desc.items():
            if desc in prompt_text:
                return spec
        return None

    def reward_func(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        rewards = []
        for prompt, completion in zip(prompts, completions):
            command = extract_command(completion)

            spec = _match_task(prompt)
            if spec is None:
                rewards.append(0.0)
                continue

            client = TerminalBenchClient(
                task_specs=[spec],
                command_timeout=command_timeout,
                max_output_length=max_output_length,
            )
            env = TerminalBenchEnv(
                client,
                max_steps=1,  # single-command episode: partial credit counts
                step_penalty=step_penalty,
                w_success=w_success,
                w_eff=w_eff,
                w_quality=w_quality,
            )
            env.reset()
            _obs, reward, _done, _info = env.step(command)
            rewards.append(reward)
        return rewards

    return reward_func
