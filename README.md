# AfterQuery — RLVR for Terminal Tasks

A Reinforcement Learning from Verifiable Rewards (RLVR) framework for training language models to perform terminal/CLI tasks. Models learn to issue efficient shell commands by executing them in real sandboxed environments and receiving verifiable rewards based on filesystem state.

- **Framework:** HuggingFace TRL (GRPO + PPO)
- **Benchmark:** 25 terminal tasks (easy / medium / hard) with sandbox execution
- **Models:** TinyLlama-1.1B, Qwen 3B/7B (with LoRA + 4-bit quantization for larger models)
- **Training:** Local GPU/CPU or Google Colab (free T4)

## How It Works

The LLM is treated as an RL agent interacting with a real shell:

| Concept | Description |
|---------|-------------|
| **Observation** | Text prompt encoding the task description, difficulty, step count, and command history |
| **Action** | A generated shell command, executed in an isolated temp directory via bash subprocess |
| **Reward** | Weighted combination of benchmark success score (0–1), step efficiency penalty, and command quality penalty |

Two RL algorithms are supported:
- **GRPO** (default, recommended) — No critic/value model needed, uses group-relative baselines. Memory efficient.
- **PPO** — Classic policy gradient with a separate value model. Higher memory, more general.

## Project Structure

```
configs/
  config_terminalbench.yaml    # Hyperparameters, model selection, reward weights
envs/
  tasks.py                     # 25 task definitions with setup() and verify()
  terminalbench_client.py      # Sandbox client: temp dirs, subprocess, scoring
  terminalbench_env.py         # Gym-style environment for the RLVR MDP
scripts/
  train_rlvr.py                # Main GRPO/PPO training loop
  evaluate_terminalbench.py    # Evaluate trained or base models
  compare_results.py           # Side-by-side baseline vs trained comparison
colab/
  train_colab_3b_200_steps.ipynb   # Qwen 3B, 200 steps (Colab T4)
  train_colab_3b_500_steps.ipynb   # Qwen 3B, 500 steps
  train_colab_7b_200_steps.ipynb   # Qwen 7B + LoRA, 200 steps
  train_colab_7b_500_steps.ipynb   # Qwen 7B + LoRA, 500 steps
  train_colab_7b_save_model.ipynb  # Qwen 7B full pipeline with eval + Drive save
```

## Benchmark Tasks

25 tasks across three difficulty levels:

| Difficulty | Count | Examples |
|------------|-------|---------|
| **Easy** | 10 | Create a file, copy/rename/delete files, list directory, count lines |
| **Medium** | 10 | Find files recursively, grep errors from logs, sort, word frequency, CSV column extraction |
| **Hard** | 5 | Log pipeline (extract + dedup + sort), create & execute a script, tar.gz archive, JSON parsing |

Each task provides a `setup()` function that creates initial filesystem state and a `verify()` function that checks the result, returning a score from 0.0 to 1.0.

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train

```bash
# GRPO 
python scripts/train_rlvr.py \
  --config configs/config_terminalbench.yaml \
  --num_prompts 16

# Full training run
python scripts/train_rlvr.py \
  --config configs/config_terminalbench.yaml \
  --num_prompts 256
```

Checkpoints are saved to `models/rlvr/` every 200 steps.

### 3. Evaluate

```bash
# Evaluate trained model
python scripts/evaluate_terminalbench.py \
  --model_dir models/rlvr \
  --num_episodes 10

# Evaluate base model
python scripts/evaluate_terminalbench.py --num_episodes 10
```

### 4. Compare Results

```bash
python scripts/compare_results.py results/baseline.json results/trained.json
```

Outputs a formatted table with per-difficulty breakdowns and percentage changes.

## Training on Google Colab

The `colab/` directory contains ready-to-run notebooks for training on a free T4 GPU:

| Notebook | Model | Steps | Notes |
|----------|-------|-------|-------|
| `train_colab_3b_200_steps` | Qwen 2.5 3B | 200 | Fastest, good for prototyping |
| `train_colab_3b_500_steps` | Qwen 2.5 3B | 500 | Longer training |
| `train_colab_7b_200_steps` | Qwen 2.5 7B | 200 | LoRA + 4-bit quantization |
| `train_colab_7b_500_steps` | Qwen 2.5 7B | 500 | LoRA + 4-bit quantization |
| `train_colab_7b_save_model` | Qwen 2.5 7B | 500 | Full pipeline: train + eval + save to Drive |

The 7B notebooks use **4-bit quantization + LoRA** to fit within the T4's 16GB VRAM, with gradient checkpointing and reduced batch sizes.

## Switching Models

Edit `configs/config_terminalbench.yaml`:

```yaml
# TinyLlama 1.1B (default, no approval needed)
model_name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Qwen 3B (no approval needed)
model_name: "Qwen/Qwen2.5-3B-Instruct"

# Qwen 7B (no approval needed, use LoRA for <16GB VRAM)
model_name: "Qwen/Qwen2.5-7B-Instruct"

# Llama-2 7B (requires access approval)
model_name: "meta-llama/Llama-2-7b-chat-hf"
```

For gated models, log in first: `huggingface-cli login`

## GPU Requirements

| Model | Inference | PPO Training | GRPO Training |
|-------|-----------|-------------|---------------|
| TinyLlama 1.1B | CPU or 4GB+ GPU | ~8GB | ~4GB |
| Qwen 3B | ~6GB | ~16GB | ~8GB |
| Qwen 7B (4-bit + LoRA) | ~6GB | — | ~14GB |
| Llama-2 7B | ~14GB | ~28GB+ | ~16GB |

## Configuration

Key settings in `configs/config_terminalbench.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `env.max_steps` | 5 | Max commands per episode |
| `env.command_timeout` | 10s | Per-command timeout |
| `env.w_success` | 1.0 | Weight for task completion score |
| `env.w_eff` | 0.1 | Weight for step efficiency penalty |
| `env.w_quality` | 0.05 | Weight for command quality penalty |
| `train.total_updates` | 800 | Total training steps |
| `ppo.learning_rate` | 1.5e-5 | Learning rate |
| `ppo.batch_size` | 16 | Batch size |

## Sandbox Security

Commands are executed in isolated environments with several safety measures:
- Temporary directories with custom `HOME` — no access to real filesystem
- Interactive commands blocked (`bash`, `vi`, `ssh`, `python` REPL, etc.)
- 10-second timeout with hard-kill on runaway processes
- Output truncated to 2000 characters
- Windows support via Git Bash with automatic path conversion
