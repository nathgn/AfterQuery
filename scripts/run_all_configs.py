# scripts/run_all_configs.py
#
# Queue every training configuration from colab/ as a headless run.
#
# The notebooks in colab/ are interactive and Colab-specific (they import
# google.colab, mount Drive, and need manual runtime restarts), so they can't
# be batched. This reproduces all five configs through the repo's CLI instead:
# each one trains, evaluates, and writes results/<name>.json, unattended.
#
# Requires a CUDA GPU: every config uses 4-bit NF4 quantization via
# bitsandbytes, which has no Metal or CPU backend.
#
# Usage:
#   python scripts/run_all_configs.py                  # run all five
#   python scripts/run_all_configs.py --only qwen3b_200 qwen7b_500
#   python scripts/run_all_configs.py --dry_run        # print commands only
#   python scripts/run_all_configs.py --eval_only      # skip training

import argparse
import os
import subprocess
import sys
import time

# Mirrors colab/*.ipynb one-for-one. Keep in sync if a notebook changes.
CONFIGS = [
    {
        "name": "qwen3b_200",
        "notebook": "colab/train_colab_3b_200_steps.ipynb",
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "total_steps": 200, "learning_rate": 5e-6, "max_new_tokens": 64,
        "num_prompts": 256, "eval_episodes": 20,
    },
    {
        "name": "qwen3b_500",
        "notebook": "colab/train_colab_3b_500_steps.ipynb",
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "total_steps": 500, "learning_rate": 5e-6, "max_new_tokens": 64,
        "num_prompts": 256, "eval_episodes": 20,
    },
    {
        "name": "qwen7b_200",
        "notebook": "colab/train_colab_7b_200_steps.ipynb",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "total_steps": 200, "learning_rate": 5e-6, "max_new_tokens": 64,
        "num_prompts": 256, "eval_episodes": 20,
    },
    {
        "name": "qwen7b_500",
        "notebook": "colab/train_colab_7b_500_steps.ipynb",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "total_steps": 500, "learning_rate": 5e-6, "max_new_tokens": 64,
        "num_prompts": 256, "eval_episodes": 20,
    },
    {
        "name": "qwen7b_save",
        "notebook": "colab/train_colab_7b_save_model.ipynb",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "total_steps": 500, "learning_rate": 1e-5, "max_new_tokens": 128,
        "num_prompts": 256, "eval_episodes": 30,
    },
]


def check_cuda() -> bool:
    try:
        import torch
    except ImportError:
        print("torch is not installed. pip install -r requirements.txt", file=sys.stderr)
        return False
    if not torch.cuda.is_available():
        print(
            "No CUDA GPU detected.\n\n"
            "Every config here uses 4-bit NF4 quantization via bitsandbytes,\n"
            "which has no Metal or CPU backend, and each is a 200-500 step GRPO\n"
            "run on a 3B-7B policy (1-2 hours per config on a T4).\n\n"
            "Run this on a CUDA machine. On Apple Silicon you can still get\n"
            "base-model numbers with:\n"
            "    python scripts/evaluate_mlx.py --model mlx-community/Qwen2.5-7B-Instruct-4bit \\\n"
            "        --num_episodes 50 --seed 42 --output results/qwen7b_mlx_baseline.json",
            file=sys.stderr,
        )
        return False
    print(f"CUDA: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    return True


def run(cmd: list[str], dry: bool) -> int:
    print("  $ " + " ".join(cmd), flush=True)
    if dry:
        return 0
    return subprocess.call(cmd)


def main(args: argparse.Namespace) -> int:
    configs = CONFIGS
    if args.only:
        names = set(args.only)
        configs = [c for c in CONFIGS if c["name"] in names]
        unknown = names - {c["name"] for c in CONFIGS}
        if unknown:
            print(f"Unknown config(s): {sorted(unknown)}", file=sys.stderr)
            print(f"Available: {[c['name'] for c in CONFIGS]}", file=sys.stderr)
            return 2

    if not args.dry_run and not check_cuda():
        return 1

    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print(f"\nQueued {len(configs)} config(s): {[c['name'] for c in configs]}\n")
    outcomes = []

    for i, c in enumerate(configs, 1):
        out_dir = f"models/{c['name']}"
        result = f"results/{c['name']}_trained.json"
        print("=" * 70)
        print(f"[{i}/{len(configs)}] {c['name']}  ({c['notebook']})")
        print("=" * 70)
        started = time.time()

        if not args.eval_only:
            rc = run([
                sys.executable, "scripts/train_rlvr.py",
                "--config", args.config,
                "--model", c["model"],
                "--output_dir", out_dir,
                "--num_prompts", str(c["num_prompts"]),
                "--total_steps", str(c["total_steps"]),
                "--learning_rate", str(c["learning_rate"]),
                "--max_new_tokens", str(c["max_new_tokens"]),
                "--lora", "--load_in_4bit",
            ], args.dry_run)
            if rc != 0:
                print(f"  TRAIN FAILED (exit {rc}) - skipping eval\n")
                outcomes.append((c["name"], "train failed", 0.0))
                continue

        rc = run([
            sys.executable, "scripts/evaluate_terminalbench.py",
            "--config", args.config,
            "--model_dir", out_dir,
            "--num_episodes", str(c["eval_episodes"]),
            "--seed", str(args.seed),
            "--output", result,
        ], args.dry_run)

        mins = (time.time() - started) / 60
        if rc != 0:
            print(f"  EVAL FAILED (exit {rc})\n")
            outcomes.append((c["name"], "eval failed", mins))
        else:
            print(f"  done in {mins:.1f} min -> {result}\n")
            outcomes.append((c["name"], "ok", mins))

    print("=" * 70)
    print("QUEUE SUMMARY")
    print("=" * 70)
    for name, status, mins in outcomes:
        print(f"  {name:<14s}  {status:<12s}  {mins:>6.1f} min")
    if not args.dry_run:
        print("\nTabulate everything with:")
        print("  python scripts/summarize_results.py")
    return 0 if all(s == "ok" for _, s, _ in outcomes) else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Queue all colab/ training configs headlessly")
    p.add_argument("--config", default="configs/config_terminalbench.yaml")
    p.add_argument("--only", nargs="+", default=None,
                   help=f"Subset to run. Choices: {[c['name'] for c in CONFIGS]}")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry_run", action="store_true",
                   help="Print the commands without running them")
    p.add_argument("--eval_only", action="store_true",
                   help="Skip training, evaluate existing checkpoints")
    sys.exit(main(p.parse_args()))
