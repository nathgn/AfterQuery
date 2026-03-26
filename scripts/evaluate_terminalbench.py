# scripts/evaluate_terminalbench.py
#
# Evaluate a TRL-trained (or base) model on terminalbench tasks.
#
# Usage:
#   # Evaluate base model
#   python scripts/evaluate_terminalbench.py --num_episodes 10
#
#   # Evaluate trained model
#   python scripts/evaluate_terminalbench.py --model_dir models/rlvr --num_episodes 10

import argparse
import sys
from collections import defaultdict
from statistics import mean, stdev

import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, ".")
from envs.terminalbench_client import TerminalBenchClient
from envs.terminalbench_env import TerminalBenchEnv
from scripts.train_rlvr import _extract_command


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)

    model_name = args.model_dir or cfg["model_name"]
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name).eval()
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

    max_new = cfg["train"]["max_new_tokens"]
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
            action_text = _extract_command(raw_text)

            print(f"    step {env.step_count + 1}: {action_text[:80]}")
            obs, reward, done, info = env.step(action_text)
            episode_reward += reward
            last_score = info["success_score"]

        results.append({
            "success_score": last_score,
            "episode_reward": episode_reward,
            "step_count": info["step_count"],
            "task_id": task_id,
            "difficulty": difficulty,
        })
        print(f"  Episode {ep + 1}: task={task_id} score={last_score:.2f} steps={info['step_count']}")

    # Summary
    scores = [r["success_score"] for r in results]
    rewards = [r["episode_reward"] for r in results]
    steps = [r["step_count"] for r in results]

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Episodes: {num_episodes}")
    print(f"Mean score: {mean(scores):.3f}" +
          (f" (std: {stdev(scores):.3f})" if len(scores) > 1 else ""))
    print(f"Success rate (>=1.0): "
          f"{mean(1.0 if s >= 1.0 else 0.0 for s in scores):.3f}")
    print(f"Mean reward: {mean(rewards):.3f}")
    print(f"Mean steps: {mean(steps):.2f}")

    # Breakdown by difficulty
    by_difficulty = defaultdict(list)
    for r in results:
        by_difficulty[r["difficulty"]].append(r["success_score"])

    if by_difficulty:
        print("\nBy difficulty:")
        for diff in ["easy", "medium", "hard"]:
            if diff in by_difficulty:
                d = by_difficulty[diff]
                sr = mean(1.0 if s >= 1.0 else 0.0 for s in d)
                print(
                    f"  {diff:>8s}: score={mean(d):.3f}, "
                    f"success_rate={sr:.3f}, n={len(d)}"
                )


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
    main(parser.parse_args())
