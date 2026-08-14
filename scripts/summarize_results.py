# scripts/summarize_results.py
#
# Tabulate every evaluation result in a directory. compare_results.py handles
# exactly two runs; this gives a single table across all of them, which is what
# you want when comparing several model sizes.
#
# Usage:
#   python scripts/summarize_results.py
#   python scripts/summarize_results.py results/*.json

import argparse
import glob
import json
import os
import sys


def load(path):
    with open(path, "r") as f:
        data = json.load(f)
    if "summary" not in data:
        raise ValueError(f"{path}: not an evaluation result (no 'summary' key)")
    return data


def main(argv):
    parser = argparse.ArgumentParser(description="Summarize terminalbench evaluation results")
    parser.add_argument("paths", nargs="*", help="Result JSONs (default: results/*.json)")
    args = parser.parse_args(argv)

    paths = args.paths or sorted(glob.glob("results/*.json"))
    if not paths:
        print("No result files found in results/", file=sys.stderr)
        return 1

    rows = []
    for p in paths:
        try:
            d = load(p)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"  skipping {p}: {e}", file=sys.stderr)
            continue
        s = d["summary"]
        bd = s.get("by_difficulty", {})
        rows.append({
            "file": os.path.basename(p),
            "model": s.get("model", "?"),
            "backend": s.get("backend", "transformers"),
            "n": s.get("num_episodes", 0),
            "score": s.get("mean_score", 0.0),
            "sr": s.get("success_rate", 0.0),
            "steps": s.get("mean_steps", 0.0),
            "easy": bd.get("easy", {}).get("mean_score"),
            "medium": bd.get("medium", {}).get("mean_score"),
            "hard": bd.get("hard", {}).get("mean_score"),
        })

    if not rows:
        print("No valid result files.", file=sys.stderr)
        return 1

    rows.sort(key=lambda r: r["score"], reverse=True)

    def fmt(v):
        return "  --  " if v is None else f"{v:.3f}"

    w = max(len(r["model"]) for r in rows)
    w = max(w, 5)

    print()
    print("=" * (w + 62))
    print("  TERMINALBENCH EVALUATION SUMMARY")
    print("=" * (w + 62))
    print(f"  {'Model':<{w}}  {'Backend':<12}  {'n':>3}  {'Score':>6}  {'Success':>7}  {'Steps':>5}")
    print(f"  {'-'*w}  {'-'*12}  {'-'*3}  {'-'*6}  {'-'*7}  {'-'*5}")
    for r in rows:
        print(f"  {r['model']:<{w}}  {r['backend']:<12}  {r['n']:>3}  "
              f"{r['score']:>6.3f}  {r['sr']:>7.3f}  {r['steps']:>5.2f}")

    print()
    print(f"  {'Model':<{w}}  {'easy':>6}  {'medium':>6}  {'hard':>6}")
    print(f"  {'-'*w}  {'-'*6}  {'-'*6}  {'-'*6}")
    for r in rows:
        print(f"  {r['model']:<{w}}  {fmt(r['easy']):>6}  {fmt(r['medium']):>6}  {fmt(r['hard']):>6}")

    best = rows[0]
    print()
    print(f"  Best mean score: {best['model']} at {best['score']:.3f} "
          f"(success rate {best['sr']:.3f}, n={best['n']})")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
