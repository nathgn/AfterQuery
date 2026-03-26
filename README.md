# RLVR for TerminalBench with TRL (PPO / GRPO)

This repo implements a Reinforcement Learning from Verifiable Rewards (RLVR) setup
for improving an LLM's performance on terminal/CLI tasks.

- **Framework:** HuggingFace TRL (PPO + GRPO)
- **Benchmark:** 25 terminal tasks (easy/medium/hard) with real sandbox execution
- **Base model:** TinyLlama-1.1B-Chat (swap to Llama-2-7B-Chat when access is approved)
- **Goal:** Learn a policy that issues efficient, high-quality terminal commands

## Project Overview

The LLM is treated as an RL agent:

- **Observation:** A text prompt encoding the task description, difficulty,
  step count, and terminal command history.
- **Action:** A generated text command, executed in a real sandboxed shell
  (temp directory + bash subprocess).
- **Rewards (verifiable):**
  - Benchmark success score (normalized 0-1) verified by checking filesystem state
  - Per-step efficiency penalty (discourage long trajectories)
  - Command quality penalty (e.g. "command not found", "permission denied")

Two RL algorithms are supported:
- **GRPO** (default, recommended): No critic/value model needed, uses group-relative baselines
- **PPO**: Classic policy gradient with a value model (higher memory, more general)

## Structure

- `envs/tasks.py` - 25 task definitions with setup and verify functions.
- `envs/terminalbench_client.py` - Sandbox client: temp dirs, subprocess execution, scoring.
- `envs/terminalbench_env.py` - Gym-style environment exposing the RLVR MDP.
- `scripts/train_rlvr.py` - PPO/GRPO training with command extraction from model output.
- `scripts/evaluate_terminalbench.py` - Evaluation script for trained or base models.
- `configs/config_terminalbench.yaml` - Hyperparameters and paths.
- `requirements.txt` - Python dependencies.

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Training

```bash
# GRPO (default - recommended for RLVR, no value model needed)
PYTHONUNBUFFERED=1 python -B scripts/train_rlvr.py \
  --config configs/config_terminalbench.yaml \
  --num_prompts 16

# PPO (requires value model, higher memory)
PYTHONUNBUFFERED=1 python -B scripts/train_rlvr.py \
  --config configs/config_terminalbench.yaml \
  --algo ppo \
  --num_prompts 16

# Full training (more prompts, increase total_updates in config)
PYTHONUNBUFFERED=1 python -B scripts/train_rlvr.py \
  --config configs/config_terminalbench.yaml \
  --num_prompts 256
```

The trained model is saved to `models/rlvr/` by default.

### 3. Evaluate

```bash
# Evaluate trained model
python -B scripts/evaluate_terminalbench.py \
  --model_dir models/rlvr \
  --num_episodes 10

# Compare against base model
python -B scripts/evaluate_terminalbench.py --num_episodes 10
```

## GPU Requirements

- **TinyLlama-1.1B:** Works on CPU (~1-3 min/step) or any GPU with 4GB+ VRAM.
- **Llama-2-7B:** ~14GB VRAM for inference. With PPO (policy + value model), ~28GB+.
  GRPO halves this since there's no value model.
- **Cloud options:** Google Colab (free T4), Lambda Labs, RunPod.

## Switching Models

Edit `configs/config_terminalbench.yaml`:

```yaml
# Current (no approval needed)
model_name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# When Llama-2 access is approved
model_name: "meta-llama/Llama-2-7b-chat-hf"

# Alternative (free, 3B, no approval needed)
model_name: "Qwen/Qwen2.5-3B-Instruct"
```

For gated models, log in first: `huggingface-cli login`
