"""
Evaluate RLVR-tuned model on terminalbench2.

Usage:
    python scripts/evaluate_terminalbench.py \
        --config configs/config_terminalbench.yaml \
        --model_dir ./models/rlvr-merged
"""

import argparse
import sys
from collections import defaultdict
from statistics import mean, stdev

import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

sys.path.insert(0, ".")
from envs.terminalbench_client import TerminalBenchClient
from envs.terminalbench_env import TerminalBenchEnv


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(model_name, lora_path=None, load_in_4bit=False):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto",
        )

    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def run_episode(
    model, tokenizer, env, max_new_tokens=1024, temperature=0.7, top_p=0.9
):
    obs = env.reset()
    done = False
    episode_reward = 0.0
    last_info = {}

    while not done:
        inputs = tokenizer(
            obs, return_tensors="pt", truncation=True, max_length=2048
        ).to(model.device)

        with torch.no_grad():
            out_ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = out_ids[0][inputs["input_ids"].shape[1]:]
        action_text = tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()
        action_text = action_text.split("\n")[0].strip()

        obs, reward, done, info = env.step(action_text)
        episode_reward += reward
        last_info = info

    return {
        "success_score": last_info.get("success_score", 0.0),
        "episode_reward": episode_reward,
        "step_count": last_info.get("step_count", 0),
        "task_id": last_info.get("task_id", ""),
        "difficulty": last_info.get("difficulty", ""),
        "domain": last_info.get("domain", ""),
    }


def main(args):
    cfg = load_config(args.config)
    model_name = args.model_dir or cfg.get("model_name")
    load_in_4bit = args.load_in_4bit or (args.lora_path is not None)

    print(f"Loading model: {model_name}")
    model, tokenizer = load_model(model_name, args.lora_path, load_in_4bit)

    env_cfg = cfg.get("env", {})
    env = TerminalBenchEnv(
        TerminalBenchClient(),
        max_steps=env_cfg.get("max_steps", 20),
        step_penalty=env_cfg.get("step_penalty", 0.01),
        w_success=env_cfg.get("w_success", 1.0),
        w_eff=env_cfg.get("w_eff", 0.1),
        w_quality=env_cfg.get("w_quality", 0.05),
    )

    train_cfg = cfg.get("train", {})

    print(f"\nRunning {args.num_episodes} evaluation episodes...")
    results = []
    for i in range(args.num_episodes):
        result = run_episode(
            model, tokenizer, env,
            max_new_tokens=train_cfg.get("generate_max_len", 1024),
            temperature=train_cfg.get("temperature", 0.7),
            top_p=train_cfg.get("top_p", 0.9),
        )
        results.append(result)
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{args.num_episodes}")

    scores = [r["success_score"] for r in results]
    rewards = [r["episode_reward"] for r in results]
    steps = [r["step_count"] for r in results]

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Episodes: {args.num_episodes}")
    print(f"Mean score: {mean(scores):.3f}" +
          (f" (std: {stdev(scores):.3f})" if len(scores) > 1 else ""))
    print(f"Success rate (>=0.99): "
          f"{mean(1.0 if s >= 0.99 else 0.0 for s in scores):.3f}")
    print(f"Mean reward: {mean(rewards):.3f}")
    print(f"Mean steps: {mean(steps):.2f}")

    by_difficulty = defaultdict(list)
    for r in results:
        by_difficulty[r["difficulty"]].append(r["success_score"])

    if by_difficulty:
        print("\nBy difficulty:")
        for diff in ["easy", "medium", "hard"]:
            if diff in by_difficulty:
                d = by_difficulty[diff]
                sr = mean(1.0 if s >= 0.99 else 0.0 for s in d)
                print(
                    f"  {diff:>8s}: score={mean(d):.3f}, "
                    f"success_rate={sr:.3f}, n={len(d)}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/config_terminalbench.yaml"
    )
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--num_episodes", type=int, default=50)
    main(parser.parse_args())
