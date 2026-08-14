# scripts/evaluate_mlx.py
#
# Evaluate a model on terminalbench using MLX (Apple Silicon / Metal).
#
# Why this exists: the transformers path in evaluate_terminalbench.py needs
# bitsandbytes 4-bit quantization to fit a 7B model, and bitsandbytes is
# CUDA-only. On a Mac, MLX is the only way to run a 4-bit 7B locally. Output
# JSON matches evaluate_terminalbench.py exactly, so compare_results.py works
# across both paths.
#
# Usage:
#   python scripts/evaluate_mlx.py \
#     --model mlx-community/Qwen2.5-7B-Instruct-4bit \
#     --num_episodes 25 --seed 42 --output results/qwen7b_mlx_baseline.json

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from statistics import mean, stdev

import yaml

sys.path.insert(0, ".")
from envs.rewards import extract_command
from envs.terminalbench_client import TerminalBenchClient
from envs.terminalbench_env import TerminalBenchEnv


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(args: argparse.Namespace) -> None:
    import mlx.core as mx
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    cfg = load_config(args.config)

    if args.seed is not None:
        random.seed(args.seed)
        mx.random.seed(args.seed)
        print(f"Random seed: {args.seed}")

    print(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)

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
    sampler = make_sampler(temp=cfg["train"]["temperature"], top_p=cfg["train"]["top_p"])

    print(f"\nRunning {args.num_episodes} evaluation episodes (max_new_tokens={max_new})...")
    results = []

    for ep in range(args.num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        last_score = 0.0
        task_id = env.task.task_id if env.task else "unknown"
        difficulty = env.task.difficulty if env.task else "unknown"
        commands = []

        while not done:
            prompt = obs
            if args.chat_template and tokenizer.chat_template is not None:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": obs}],
                    add_generation_prompt=True,
                    tokenize=False,
                )

            raw_text = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_new,
                sampler=sampler,
                verbose=False,
            )
            action_text = extract_command(raw_text)

            commands.append(action_text)
            print(f"    ep{ep + 1} step {env.step_count + 1}: {action_text[:80]}")
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
        status = "PASS" if last_score >= 1.0 else "FAIL"
        print(f"  Episode {ep + 1}: [{status}] task={task_id} "
              f"score={last_score:.2f} steps={info['step_count']}")

    scores = [r["success_score"] for r in results]
    rewards = [r["episode_reward"] for r in results]
    steps = [r["step_count"] for r in results]

    summary = {
        "model": args.model,
        "num_episodes": args.num_episodes,
        "seed": args.seed,
        "mean_score": round(mean(scores), 4),
        "std_score": round(stdev(scores), 4) if len(scores) > 1 else 0.0,
        "success_rate": round(mean(1.0 if s >= 1.0 else 0.0 for s in scores), 4),
        "mean_reward": round(mean(rewards), 4),
        "mean_steps": round(mean(steps), 4),
        "backend": "mlx",
        "chat_template": bool(args.chat_template),
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

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (MLX)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Mean score: {summary['mean_score']:.3f} (std: {summary['std_score']:.3f})")
    print(f"Success rate (>=1.0): {summary['success_rate']:.3f}")
    print(f"Mean reward: {summary['mean_reward']:.3f}")
    print(f"Mean steps: {summary['mean_steps']:.2f}")

    if difficulty_breakdown:
        print("\nBy difficulty:")
        for diff in ["easy", "medium", "hard"]:
            if diff in difficulty_breakdown:
                d = difficulty_breakdown[diff]
                print(f"  {diff:>8s}: score={d['mean_score']:.3f}, "
                      f"success_rate={d['success_rate']:.3f}, n={d['count']}")

    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"summary": summary, "episodes": results}, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model on terminalbench via MLX")
    parser.add_argument("--config", default="configs/config_terminalbench.yaml")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit",
                        help="MLX model repo or local path")
    parser.add_argument("--num_episodes", type=int, default=25)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Override config train.max_new_tokens")
    parser.add_argument("--chat_template", action="store_true",
                        help="Wrap the prompt in the model's chat template "
                             "(off by default to match evaluate_terminalbench.py)")
    main(parser.parse_args())
