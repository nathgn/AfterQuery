# scripts/train_rlvr.py
#
# RLVR training for terminalbench using HuggingFace TRL.
# Supports both PPO and GRPO algorithms.
#
# Usage:
#   # GRPO (default, no value model needed - recommended for RLVR)
#   python scripts/train_rlvr.py --config configs/config_terminalbench.yaml
#
#   # PPO (requires value model, higher memory)
#   python scripts/train_rlvr.py --config configs/config_terminalbench.yaml --algo ppo
#
#   # Quick smoke test
#   python scripts/train_rlvr.py --config configs/config_terminalbench.yaml --num_prompts 16

import argparse
import re
import sys

import yaml
from datasets import Dataset

sys.path.insert(0, ".")
from envs.terminalbench_client import TerminalBenchClient
from envs.terminalbench_env import TerminalBenchEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_command(text: str) -> str:
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


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _make_client(cfg: dict) -> TerminalBenchClient:
    return TerminalBenchClient(
        command_timeout=cfg["env"].get("command_timeout", 10),
        max_output_length=cfg["env"].get("max_output_length", 2000),
    )


def _make_env(cfg: dict, tb_client: TerminalBenchClient) -> TerminalBenchEnv:
    return TerminalBenchEnv(
        tb_client,
        max_steps=cfg["env"]["max_steps"],
        step_penalty=cfg["env"]["step_penalty"],
        w_success=cfg["env"]["w_success"],
        w_eff=cfg["env"]["w_eff"],
        w_quality=cfg["env"]["w_quality"],
    )


# ---------------------------------------------------------------------------
# Reward function (shared by PPO and GRPO)
# ---------------------------------------------------------------------------

def make_reward_func(cfg: dict):
    """Create a reward function that runs commands in the terminalbench env."""
    tb_client = _make_client(cfg)
    env = _make_env(cfg, tb_client)

    def reward_func(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        rewards = []
        for prompt, completion in zip(prompts, completions):
            command = _extract_command(completion)
            env.reset()
            _obs, reward, _done, _info = env.step(command)
            rewards.append(reward)
        return rewards

    return reward_func


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_prompt_dataset(cfg: dict, num_prompts: int = 256) -> Dataset:
    """Build a dataset of task prompts for training."""
    tb_client = _make_client(cfg)
    env = _make_env(cfg, tb_client)

    prompts = []
    for _ in range(num_prompts):
        obs = env.reset()
        prompts.append({"prompt": obs})

    return Dataset.from_list(prompts)


# ---------------------------------------------------------------------------
# GRPO training (recommended for RLVR)
# ---------------------------------------------------------------------------

def train_grpo(cfg: dict, args: argparse.Namespace) -> None:
    """Train using GRPO - no value model needed."""
    from trl import GRPOConfig, GRPOTrainer

    train_dataset = build_prompt_dataset(cfg, num_prompts=args.num_prompts)
    reward_func = make_reward_func(cfg)

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=cfg["ppo"]["learning_rate"],
        per_device_train_batch_size=2,
        num_generations=2,
        num_train_epochs=1,
        max_steps=cfg["train"]["total_updates"],
        max_completion_length=cfg["train"]["max_new_tokens"],
        temperature=cfg["train"]["temperature"],
        top_p=cfg["train"]["top_p"],
        beta=cfg["ppo"]["target_kl"],
        logging_steps=1,
        save_strategy="steps",
        save_steps=100,
        report_to="none",
        bf16=False,
        fp16=False,
        use_cpu=True,
        gradient_checkpointing=False,
    )

    trainer = GRPOTrainer(
        model=cfg["model_name"],
        reward_funcs=reward_func,
        args=grpo_config,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


# ---------------------------------------------------------------------------
# PPO training
# ---------------------------------------------------------------------------

def train_ppo(cfg: dict, args: argparse.Namespace) -> None:
    """Train using PPO - requires a value model."""
    from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
    from transformers import AutoTokenizer

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load policy model with value head
    model = AutoModelForCausalLMWithValueHead.from_pretrained(cfg["model_name"])

    # Build prompt dataset
    train_dataset = build_prompt_dataset(cfg, num_prompts=args.num_prompts)

    # PPO configuration
    ppo_config = PPOConfig(
        learning_rate=cfg["ppo"]["learning_rate"],
        batch_size=cfg["ppo"]["batch_size"],
        mini_batch_size=cfg["ppo"]["mini_batch_size"],
        ppo_epochs=cfg["ppo"]["ppo_epochs"],
        target_kl=cfg["ppo"]["target_kl"],
        gamma=cfg["ppo"]["gamma"],
        lam=cfg["ppo"]["gae_lambda"],
        cliprange=cfg["ppo"]["clip_range"],
        log_with=None,
    )

    # Create PPO trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
        dataset=train_dataset,
    )

    # Build reward function
    tb_client = _make_client(cfg)
    env = _make_env(cfg, tb_client)

    max_new_tokens = cfg["train"]["max_new_tokens"]
    total_updates = cfg["train"]["total_updates"]

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": cfg["train"]["temperature"],
        "top_p": cfg["train"]["top_p"],
        "pad_token_id": tokenizer.eos_token_id,
    }

    print(f"Starting PPO training: {total_updates} updates, {args.num_prompts} prompts")

    step = 0
    for epoch in range(1):  # single epoch; control via total_updates
        for batch_idx in range(0, len(train_dataset), ppo_config.batch_size):
            if step >= total_updates:
                break

            # Get batch of prompts
            batch_end = min(batch_idx + ppo_config.batch_size, len(train_dataset))
            batch_prompts = [train_dataset[i]["prompt"] for i in range(batch_idx, batch_end)]

            # Tokenize prompts
            query_tensors = [
                tokenizer.encode(p, return_tensors="pt", truncation=True, max_length=512).squeeze()
                for p in batch_prompts
            ]

            # Generate responses
            response_tensors = ppo_trainer.generate(
                query_tensors,
                **generation_kwargs,
            )

            # Decode responses
            responses_text = [
                tokenizer.decode(r.squeeze(), skip_special_tokens=True)
                for r in response_tensors
            ]

            # Compute rewards
            import torch
            rewards = []
            for prompt_text, response_text in zip(batch_prompts, responses_text):
                command = _extract_command(response_text)
                env.reset()
                _obs, reward, _done, _info = env.step(command)
                rewards.append(torch.tensor(reward, dtype=torch.float32))

            # PPO update step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

            step += 1
            mean_reward = sum(r.item() for r in rewards) / len(rewards)
            print(f"Step {step}/{total_updates} | mean_reward={mean_reward:.4f}")

            if step >= total_updates:
                break

    # Save the trained model
    ppo_trainer.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)

    print(f"Algorithm: {args.algo.upper()}")
    print(f"Model: {cfg['model_name']}")
    print(f"Prompts: {args.num_prompts}")
    print(f"Output: {args.output_dir}")
    print()

    if args.algo == "grpo":
        train_grpo(cfg, args)
    elif args.algo == "ppo":
        train_ppo(cfg, args)
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}. Use 'ppo' or 'grpo'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLVR training for terminalbench")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_terminalbench.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/rlvr",
        help="Where to save the fine-tuned model",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=256,
        help="Number of prompts to generate for training dataset",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="grpo",
        choices=["ppo", "grpo"],
        help="RL algorithm to use: 'ppo' or 'grpo' (default: grpo)",
    )
    main(parser.parse_args())
