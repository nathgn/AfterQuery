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
import sys

import yaml
from datasets import Dataset

sys.path.insert(0, ".")
from envs.rewards import make_reward_func as _make_reward_func
from envs.terminalbench_client import TerminalBenchClient
from envs.terminalbench_env import TerminalBenchEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    """Build the shared task-matched reward function from the config.

    See envs/rewards.py: the prompt is matched back to its TaskSpec (so the
    command is scored against the right task) and partial verify() scores
    flow into the reward.
    """
    return _make_reward_func(
        command_timeout=cfg["env"].get("command_timeout", 10),
        max_output_length=cfg["env"].get("max_output_length", 2000),
        step_penalty=cfg["env"]["step_penalty"],
        w_success=cfg["env"]["w_success"],
        w_eff=cfg["env"]["w_eff"],
        w_quality=cfg["env"]["w_quality"],
    )


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
    import torch
    from trl import GRPOConfig, GRPOTrainer

    train_dataset = build_prompt_dataset(cfg, num_prompts=args.num_prompts)
    reward_func = make_reward_func(cfg)

    # TRL requires the effective batch size (per_device_train_batch_size *
    # num_processes * gradient_accumulation_steps) to be evenly divisible
    # by num_generations.
    batch_size = cfg["train"].get("per_device_batch_size", 4)
    num_generations = cfg["train"].get("num_generations", 4)
    if batch_size % num_generations != 0:
        raise ValueError(
            f"train.per_device_batch_size ({batch_size}) must be divisible "
            f"by train.num_generations ({num_generations})"
        )

    use_cuda = torch.cuda.is_available()
    # Apple Silicon: let the Trainer use Metal instead of falling back to CPU.
    # Stay in fp32 there — fp16 grad scaling and bf16 are both unreliable on MPS.
    use_mps = not use_cuda and torch.backends.mps.is_available()
    bf16 = use_cuda and torch.cuda.is_bf16_supported()

    if args.load_in_4bit and not use_cuda:
        raise SystemExit(
            "--load_in_4bit needs a CUDA GPU: bitsandbytes has no Metal/CPU "
            "4-bit backend. Drop the flag to train in full precision, or run "
            "this on a CUDA machine (Colab T4 is what the colab/ notebooks use)."
        )

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate or cfg["ppo"]["learning_rate"],
        per_device_train_batch_size=batch_size,
        num_generations=num_generations,
        num_train_epochs=1,
        max_steps=args.total_steps or cfg["train"]["total_updates"],
        max_completion_length=args.max_new_tokens or cfg["train"]["max_new_tokens"],
        temperature=cfg["train"]["temperature"],
        top_p=cfg["train"]["top_p"],
        beta=cfg["ppo"]["target_kl"],
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        report_to="none",
        bf16=bf16,
        fp16=use_cuda and not bf16,
        use_cpu=not (use_cuda or use_mps),
        gradient_checkpointing=use_cuda,
    )

    model_name = args.model or cfg["model_name"]

    # 4-bit quantization + LoRA: what the colab/ notebooks use to fit a 7B
    # policy on a 16GB T4. Both are opt-in so the default path is unchanged.
    model_arg = model_name
    if args.load_in_4bit:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        model_arg = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
                bnb_4bit_quant_type="nf4",
            ),
            device_map="auto",
        )
    elif use_mps:
        # Passing a model *name* lets transformers lazy-init on the meta device,
        # which blows up in backward ("expected device meta but got mps:0").
        # Materialize the weights on Metal up front instead.
        from transformers import AutoModelForCausalLM
        model_arg = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32
        ).to("mps")

    peft_config = None
    if args.lora:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )

    device = "cuda" if use_cuda else ("mps" if use_mps else "cpu")
    print(f"Model: {model_name} | 4bit={args.load_in_4bit} | lora={args.lora} "
          f"| steps={grpo_config.max_steps} | device={device}")

    trainer = GRPOTrainer(
        model=model_arg,
        reward_funcs=[reward_func],
        args=grpo_config,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"\nModel saved to {args.output_dir}")


# ---------------------------------------------------------------------------
# PPO training
# ---------------------------------------------------------------------------

def train_ppo(cfg: dict, args: argparse.Namespace) -> None:
    """Train using PPO - requires a value model.

    NOTE: this path targets the legacy TRL PPO API (trl < 0.9:
    PPOConfig(ppo_epochs=..., log_with=...), ppo_trainer.step(...)) and will
    not run against the trl >= 0.14 required by the GRPO path. Kept for
    reference; pin trl==0.8.x in a separate environment to use it.
    """
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

    # Build reward function (task-matched, shared with GRPO)
    reward_func = make_reward_func(cfg)

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

            # Compute rewards against the task each prompt actually posed
            import torch
            rewards = [
                torch.tensor(r, dtype=torch.float32)
                for r in reward_func(batch_prompts, responses_text)
            ]

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

    print("\nTraining complete. Exiting.")
    sys.exit(0)


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
        default=512,
        help="Number of prompts to generate for training dataset",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="grpo",
        choices=["ppo", "grpo"],
        help="RL algorithm to use: 'ppo' or 'grpo' (default: grpo)",
    )
    # Overrides so each colab/ notebook config is reproducible headlessly.
    parser.add_argument("--model", type=str, default=None,
                        help="Override config model_name")
    parser.add_argument("--total_steps", type=int, default=None,
                        help="Override train.total_updates")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Override ppo.learning_rate")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Override train.max_new_tokens")
    parser.add_argument("--save_steps", type=int, default=200,
                        help="Checkpoint interval")
    parser.add_argument("--lora", action="store_true",
                        help="Train LoRA adapters instead of full fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank (alpha is set to 2x this)")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="4-bit NF4 quantization (requires CUDA)")
    main(parser.parse_args())
