"""
OpenRLHF-compatible custom reward function for terminalbench2 RLVR.

With GRPO, OpenRLHF generates N responses per prompt and calls this
function on all of them. Advantages are group-normalized:
  A_i = (R_i - mean(G)) / std(G)
"""

import re
from typing import Dict, List
import torch


W_SUCCESS = 1.0
W_EFFICIENCY = 0.1
W_QUALITY = 0.05
STEP_PENALTY = 0.01


def _extract_command(query: str, prompt: str) -> str:
    if query.startswith(prompt):
        response = query[len(prompt):].strip()
    else:
        if "Next command:" in query:
            response = query.split("Next command:")[-1].strip()
        else:
            response = query.strip()

    lines = response.strip().split("\n")
    command = lines[0].strip() if lines else ""

    for prefix in ["$ ", ">>> ", "> ", "RUN: ", "SUBMIT: "]:
        if command.startswith(prefix):
            command = command[len(prefix):]

    return command.strip()


def _extract_task_description(prompt: str) -> str:
    match = re.search(
        r"Task:\s*\n(.*?)(?:\n\nCurrent terminal output:)", prompt, re.DOTALL
    )
    return match.group(1).strip() if match else ""


def _extract_filenames(text: str) -> List[str]:
    return re.findall(r"\b[\w-]+\.\w+\b", text)


def _compute_success_reward(command: str, prompt: str, label: str) -> float:
    task_desc = _extract_task_description(prompt).lower()
    cmd = command.lower()

    if not cmd:
        return 0.0

    score = 0.0

    if "create a file" in task_desc or "write" in task_desc:
        if "echo" in cmd and ">" in cmd:
            score = 0.7
            if any(word in cmd for word in _extract_filenames(task_desc)):
                score = 1.0
    elif "list" in task_desc and ("find" in task_desc or ".py" in task_desc):
        if "find" in cmd and ".py" in cmd:
            score = 0.8
            if "wc" in cmd or "count" in cmd:
                score = 1.0
    elif "fix" in task_desc or "debug" in task_desc:
        if "sed" in cmd or "patch" in cmd or "vim" in cmd:
            score = 0.5
        if "python" in cmd and "test" in cmd:
            score = 0.8
    elif "script" in task_desc or "bash" in task_desc:
        if "bash" in cmd or "chmod" in cmd or "sh " in cmd:
            score = 0.6
    elif "patch" in task_desc or "vulnerability" in task_desc:
        if "sed" in cmd or "patch" in cmd or "git" in cmd:
            score = 0.5

    if label:
        label_lower = label.lower().strip()
        if cmd == label_lower:
            score = 1.0
        elif label_lower in cmd or cmd in label_lower:
            score = max(score, 0.8)

    return min(score, 1.0)


def _compute_quality_reward(command: str) -> float:
    cmd = command.strip()
    if not cmd:
        return -0.5

    penalty = 0.0
    dangerous = [
        "rm -rf /", "rm -rf /*", "mkfs", "dd if=/dev/zero",
        ":(){:|:&};:", "chmod -R 777 /",
    ]
    for d in dangerous:
        if d in cmd.lower():
            penalty -= 2.0

    if len(cmd) > 500:
        penalty -= 0.3
    if cmd.count("\n") > 3:
        penalty -= 0.2

    if penalty == 0.0 and 1 < len(cmd) < 200:
        penalty += 0.01

    return penalty


def reward_func(
    queries: List[str],
    prompts: List[str],
    labels: List[str],
) -> Dict[str, object]:
    batch_size = len(queries)
    rewards = []
    scores = []
    success_rewards = []
    quality_rewards = []

    for i in range(batch_size):
        query = queries[i]
        prompt = prompts[i]
        label = labels[i] if labels and i < len(labels) else ""

        command = _extract_command(query, prompt)
        r_success = _compute_success_reward(command, prompt, label)
        r_efficiency = -STEP_PENALTY
        r_quality = _compute_quality_reward(command)

        total_reward = (
            W_SUCCESS * r_success
            + W_EFFICIENCY * r_efficiency
            + W_QUALITY * r_quality
        )

        rewards.append(total_reward)
        scores.append(r_success)
        success_rewards.append(r_success)
        quality_rewards.append(r_quality)

    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)

    return {
        "rewards": rewards_tensor,
        "scores": scores_tensor,
        "extra_logs": {
            "tb2/mean_success_reward": sum(success_rewards) / batch_size,
            "tb2/mean_quality_reward": sum(quality_rewards) / batch_size,
            "tb2/mean_total_reward": rewards_tensor.mean().item(),
            "tb2/success_rate": sum(
                1.0 for s in scores if s >= 0.99
            ) / batch_size,
        },
    }
