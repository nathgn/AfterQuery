# scripts/compare_results.py
#
# Compare baseline vs trained model evaluation results.
#
# Usage:
#   python scripts/compare_results.py results/baseline.json results/trained.json

import argparse
import json
import sys
from collections import defaultdict


def load_results(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def pct_change(baseline: float, trained: float) -> str:
    """Format percentage change with arrow."""
    if baseline == 0:
        if trained == 0:
            return "  --"
        return "  +inf"
    change = ((trained - baseline) / abs(baseline)) * 100
    arrow = "^" if change > 0 else "v" if change < 0 else "="
    return f"{arrow} {abs(change):+.1f}%"


def print_header(title: str, width: int = 70):
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_row(label: str, baseline_val, trained_val, fmt: str = ".3f", show_change: bool = True):
    b = f"{baseline_val:{fmt}}"
    t = f"{trained_val:{fmt}}"
    change = ""
    if show_change and isinstance(baseline_val, (int, float)) and isinstance(trained_val, (int, float)):
        diff = trained_val - baseline_val
        if abs(diff) < 0.0001:
            change = "  (no change)"
        else:
            sign = "+" if diff > 0 else ""
            change = f"  ({sign}{diff:{fmt}})"
    print(f"  {label:<28s}  {b:>10s}  {t:>10s}{change}")


def main(args: argparse.Namespace) -> None:
    baseline_data = load_results(args.baseline)
    trained_data = load_results(args.trained)

    b = baseline_data["summary"]
    t = trained_data["summary"]

    # -- Header --
    print_header("MODEL COMPARISON: Baseline vs RLVR-Trained")
    print(f"  Baseline model:  {b['model']}")
    print(f"  Trained model:   {t['model']}")
    print(f"  Episodes:        {b['num_episodes']} (baseline), {t['num_episodes']} (trained)")
    print(f"  Seed:            {b.get('seed', 'N/A')} (baseline), {t.get('seed', 'N/A')} (trained)")

    # -- Overall metrics --
    print_header("OVERALL METRICS")
    print(f"  {'Metric':<28s}  {'Baseline':>10s}  {'Trained':>10s}  {'Delta':>10s}")
    print(f"  {'-'*28}  {'-'*10}  {'-'*10}  {'-'*10}")
    print_row("Mean Score", b["mean_score"], t["mean_score"])
    print_row("Success Rate (>=1.0)", b["success_rate"], t["success_rate"])
    print_row("Mean Reward", b["mean_reward"], t["mean_reward"])
    print_row("Mean Steps", b["mean_steps"], t["mean_steps"], fmt=".2f")

    # -- Difficulty breakdown --
    b_diff = b.get("by_difficulty", {})
    t_diff = t.get("by_difficulty", {})
    all_diffs = sorted(set(list(b_diff.keys()) + list(t_diff.keys())),
                       key=lambda x: {"easy": 0, "medium": 1, "hard": 2}.get(x, 3))

    if all_diffs:
        print_header("BY DIFFICULTY")
        print(f"  {'Difficulty':<12s}  {'Metric':<18s}  {'Baseline':>10s}  {'Trained':>10s}  {'Delta':>10s}")
        print(f"  {'-'*12}  {'-'*18}  {'-'*10}  {'-'*10}  {'-'*10}")
        for diff in all_diffs:
            bd = b_diff.get(diff, {"mean_score": 0, "success_rate": 0, "count": 0})
            td = t_diff.get(diff, {"mean_score": 0, "success_rate": 0, "count": 0})

            b_score = bd["mean_score"]
            t_score = td["mean_score"]
            b_sr = bd["success_rate"]
            t_sr = td["success_rate"]
            b_n = bd["count"]
            t_n = td["count"]

            score_delta = t_score - b_score
            sr_delta = t_sr - b_sr
            sign_s = "+" if score_delta > 0 else ""
            sign_sr = "+" if sr_delta > 0 else ""

            print(f"  {diff:<12s}  {'Score':<18s}  {b_score:>10.3f}  {t_score:>10.3f}  {sign_s}{score_delta:>+9.3f}")
            print(f"  {'':<12s}  {'Success Rate':<18s}  {b_sr:>10.3f}  {t_sr:>10.3f}  {sign_sr}{sr_delta:>+9.3f}")
            print(f"  {'':<12s}  {'Episodes':<18s}  {b_n:>10d}  {t_n:>10d}")
            print()

    # -- Per-task comparison --
    b_tasks = defaultdict(list)
    t_tasks = defaultdict(list)
    for ep in baseline_data.get("episodes", []):
        b_tasks[ep["task_id"]].append(ep["success_score"])
    for ep in trained_data.get("episodes", []):
        t_tasks[ep["task_id"]].append(ep["success_score"])

    all_task_ids = sorted(set(list(b_tasks.keys()) + list(t_tasks.keys())))

    if all_task_ids:
        print_header("PER-TASK BREAKDOWN")
        print(f"  {'Task ID':<24s}  {'Base Score':>10s}  {'Train Score':>10s}  {'Base n':>6s}  {'Train n':>7s}  {'Change':>8s}")
        print(f"  {'-'*24}  {'-'*10}  {'-'*11}  {'-'*6}  {'-'*7}  {'-'*8}")

        improved = 0
        regressed = 0
        unchanged = 0

        for tid in all_task_ids:
            b_scores = b_tasks.get(tid, [])
            t_scores = t_tasks.get(tid, [])
            b_avg = sum(b_scores) / len(b_scores) if b_scores else 0
            t_avg = sum(t_scores) / len(t_scores) if t_scores else 0
            diff = t_avg - b_avg

            if diff > 0.01:
                marker = "  +"
                improved += 1
            elif diff < -0.01:
                marker = "  -"
                regressed += 1
            else:
                marker = "  ="
                unchanged += 1

            print(f"  {tid:<24s}  {b_avg:>10.3f}  {t_avg:>11.3f}  {len(b_scores):>6d}  {len(t_scores):>7d}{marker}")

        print(f"\n  Tasks improved:  {improved}")
        print(f"  Tasks regressed: {regressed}")
        print(f"  Tasks unchanged: {unchanged}")

    # -- Final verdict --
    print_header("SUMMARY")
    score_delta = t["mean_score"] - b["mean_score"]
    sr_delta = t["success_rate"] - b["success_rate"]

    if score_delta > 0.01:
        verdict = "RLVR training IMPROVED model performance"
    elif score_delta < -0.01:
        verdict = "RLVR training DECREASED model performance"
    else:
        verdict = "RLVR training had NO SIGNIFICANT EFFECT on performance"

    print(f"  {verdict}")
    print(f"  Score:        {b['mean_score']:.3f} -> {t['mean_score']:.3f}  ({'+' if score_delta >= 0 else ''}{score_delta:.3f})")
    print(f"  Success rate: {b['success_rate']:.3f} -> {t['success_rate']:.3f}  ({'+' if sr_delta >= 0 else ''}{sr_delta:.3f})")
    print(f"  Steps:        {b['mean_steps']:.2f} -> {t['mean_steps']:.2f}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare baseline vs trained model results")
    parser.add_argument(
        "baseline",
        type=str,
        help="Path to baseline results JSON (e.g. results/baseline.json)",
    )
    parser.add_argument(
        "trained",
        type=str,
        help="Path to trained results JSON (e.g. results/trained.json)",
    )
    main(parser.parse_args())
