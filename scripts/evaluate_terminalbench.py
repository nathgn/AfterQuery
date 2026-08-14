# scripts/evaluate_terminalbench.py
#
# Evaluate a TRL-trained (or base) model on terminalbench tasks.
#
# Usage:
#   # Evaluate base model, save results
#   python scripts/evaluate_terminalbench.py --seed 42 --output results/baseline.json --num_episodes 50
#
#   # Evaluate trained model, save results
#   python scripts/evaluate_terminalbench.py --model_dir models/rlvr --seed 42 --output results/trained.json --num_episodes 50

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from statistics import mean, stdev

import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, ".")
from envs.rewards import extract_command
from envs.terminalbench_client import TerminalBenchClient
from envs.terminalbench_env import TerminalBenchEnv


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)

    # Seed RNGs: `random` drives task sampling, torch drives sampling-based
    # generation (do_sample=True). Both must be seeded for reproducible runs.
    if args.seed is not None:
        import torch
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"Random seed: {args.seed}")

    model_name = args.model_dir or cfg["model_name"]
    print(f"Loading model: {model_name}")

    # A LoRA/PEFT checkpoint is an adapter dir, not a full model - detect it and
    # load through peft so trained runs from train_rlvr.py --lora can be scored.
    is_peft = os.path.isfile(os.path.join(model_name, "adapter_config.json"))
    tokenizer_src = model_name
    if is_peft:
        import json as _json
        with open(os.path.join(model_name, "adapter_config.json")) as f:
            base = _json.load(f).get("base_model_name_or_path")
        if base and not os.path.isdir(os.path.join(model_name, "tokenizer_config.json")):
            tokenizer_src = model_name if os.path.isfile(
                os.path.join(model_name, "tokenizer_config.json")) else base
        print(f"Detected LoRA adapter (base: {base})")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_peft:
        from peft import AutoPeftModelForCausalLM
        model = AutoPeftModelForCausalLM.from_pretrained(model_name).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name).eval()

    # Use the best available accelerator; otherwise generation crawls on CPU.
    import torch as _torch
    if _torch.cuda.is_available():
        model = model.to("cuda")
    elif _torch.backends.mps.is_available():
        model = model.to("mps")
    print(f"Device: {model.device}")

    max_ctx = getattr(model.config, "max_position_embeddings", 2048)

    tb_client = TerminalBenchClient(
        command_timeout=cfg["env"].get("command_timeout", 10),
        max_output_length=cfg["env"].get("max_output_length", 2000),
    )
    env = TerminalBenchEnv(
        tb_client,
        max_steps=cfg["env"]["max_steps"],
        step_penalty=cfg["env"]["step_penalty"],
        w_success=cfg["env"]["w_success"],
        w_eff=cfg["env"]["w_eff"],
        w_quality=cfg["env"]["w_quality"],
    )

    max_new = args.max_new_tokens or cfg["train"]["max_new_tokens"]
    num_episodes = args.num_episodes

    print(f"\nRunning {num_episodes} evaluation episodes...")
    results = []

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        last_score = 0.0
        task_id = env.task.task_id if env.task else "unknown"
        difficulty = env.task.difficulty if env.task else "unknown"
        commands = []

        while not done:
            # Truncate prompt to fit within model context window
            inputs = tokenizer(
                obs,
                return_tensors="pt",
                truncation=True,
                max_length=max_ctx - max_new,
            ).to(model.device)

            out_ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new,
                do_sample=True,
                temperature=cfg["train"]["temperature"],
                top_p=cfg["train"]["top_p"],
                pad_token_id=tokenizer.eos_token_id,
            )

            # Decode only the newly generated tokens
            new_ids = out_ids[0][inputs["input_ids"].shape[1]:]
            raw_text = tokenizer.decode(new_ids, skip_special_tokens=True)
            action_text = extract_command(raw_text)

            commands.append(action_text)
            print(f"    step {env.step_count + 1}: {action_text[:80]}")
            obs, reward, done, info = env.step(action_text)
            episode_reward += reward
            last_score = info["success_score"]

        results.append({
            "episode": ep + 1,
            "task_id": task_id,
            "difficulty": difficulty,
            "success_score": last_score,
            "episode_reward": episode_reward,
            "step_count": info["step_count"],
            "commands": commands,
        })
        print(f"  Episode {ep + 1}: task={task_id} score={last_score:.2f} steps={info['step_count']}")

    # Summary
    scores = [r["success_score"] for r in results]
    rewards = [r["episode_reward"] for r in results]
    steps = [r["step_count"] for r in results]

    summary = {
        "model": model_name,
        "num_episodes": num_episodes,
        "seed": args.seed,
        "mean_score": round(mean(scores), 4),
        "std_score": round(stdev(scores), 4) if len(scores) > 1 else 0.0,
        "success_rate": round(mean(1.0 if s >= 1.0 else 0.0 for s in scores), 4),
        "mean_reward": round(mean(rewards), 4),
        "mean_steps": round(mean(steps), 4),
    }

    by_difficulty = defaultdict(list)
    for r in results:
        by_difficulty[r["difficulty"]].append(r["success_score"])

    difficulty_breakdown = {}
    for diff in ["easy", "medium", "hard"]:
        if diff in by_difficulty:
            d = by_difficulty[diff]
            difficulty_breakdown[diff] = {
                "mean_score": round(mean(d), 4),
                "success_rate": round(mean(1.0 if s >= 1.0 else 0.0 for s in d), 4),
                "count": len(d),
            }

    summary["by_difficulty"] = difficulty_breakdown

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Episodes: {num_episodes}")
    print(f"Mean score: {summary['mean_score']:.3f} (std: {summary['std_score']:.3f})")
    print(f"Success rate (>=1.0): {summary['success_rate']:.3f}")
    print(f"Mean reward: {summary['mean_reward']:.3f}")
    print(f"Mean steps: {summary['mean_steps']:.2f}")

    if difficulty_breakdown:
        print("\nBy difficulty:")
        for diff in ["easy", "medium", "hard"]:
            if diff in difficulty_breakdown:
                d = difficulty_breakdown[diff]
                print(
                    f"  {diff:>8s}: score={d['mean_score']:.3f}, "
                    f"success_rate={d['success_rate']:.3f}, n={d['count']}"
                )

    # Save results to JSON
    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        output_data = {
            "summary": summary,
            "episodes": results,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/config_terminalbench.yaml",
    )
    parser.add_argument(
        "--model_dir", default=None,
        help="Path or HF ID of the RLVR-fine-tuned model; defaults to base model.",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=50,
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible task sampling.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save results as JSON (e.g. results/baseline.json).",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=None,
        help="Override config train.max_new_tokens.",
    )
    main(parser.parse_args())
