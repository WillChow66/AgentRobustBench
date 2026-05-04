#!/usr/bin/env python3
"""
Print the overall + per-benchmark scores for every model after the reparse.
Also produces a CSV written to paper/analysis/score_cvs/ if that directory
exists, otherwise prints to stdout only.
"""

import csv
import json
import sys
from pathlib import Path

RESULTS_ROOT = Path("<repo-root>/results/robustbench/results")


def load_scores(mdir):
    p = mdir / "scores.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def main():
    # Collect per-model overall summaries
    rows = []
    for mdir in sorted(RESULTS_ROOT.iterdir()):
        if not mdir.is_dir():
            continue
        data = load_scores(mdir)
        if not data:
            continue
        ov = data["overall"]["overall"]
        by_bm = data["overall"]["by_benchmark"]
        by_pert = data["overall"]["by_perturbation"]

        name = mdir.name
        # Shorten SFT names
        if "SFT-Clean-4k" in name:
            if "checkpoint" in name:
                short = "SFT-Clean-4k-ckpt100"
            else:
                short = "SFT-Clean-4k"
        elif "SFT-Robust" in name:
            short = "SFT-Robust"
        else:
            short = name.split("__", 1)[-1]  # drop org prefix

        row = {
            "model": short,
            "n": ov["n"],
            "overall_acc": ov["accuracy"],
        }
        for bm in ("bfcl_v3", "apibank", "rotbench", "toolalpaca", "tooleyes"):
            v = by_bm.get(bm, {})
            row[f"{bm}_n"] = v.get("n", 0)
            row[f"{bm}_acc"] = v.get("accuracy", None)
        row["clean_acc"] = by_pert.get("clean", {}).get("accuracy", None)
        # Avg transient acc
        tr_keys = [k for k in by_pert if k.startswith("transient_")]
        if tr_keys:
            row["transient_avg"] = sum(
                by_pert[k]["accuracy"] for k in tr_keys
            ) / len(tr_keys)
        else:
            row["transient_avg"] = None
        rows.append(row)

    # Sort by overall accuracy descending
    rows.sort(key=lambda r: -(r["overall_acc"] or 0))

    # Print table
    print(f"\n{'Model':<34} {'n':>5} {'Overall':>8} {'Clean':>7} "
          f"{'BFCL':>6} {'APIBk':>6} {'RoT':>6} {'TAlp':>6} {'TEye':>6} "
          f"{'Trans':>7}")
    print("-" * 108)
    for r in rows:
        print(f"{r['model']:<34} {r['n']:>5} "
              f"{r['overall_acc']:>8.4f} "
              f"{_fmt(r['clean_acc']):>7} "
              f"{_fmt(r['bfcl_v3_acc']):>6} "
              f"{_fmt(r['apibank_acc']):>6} "
              f"{_fmt(r['rotbench_acc']):>6} "
              f"{_fmt(r['toolalpaca_acc']):>6} "
              f"{_fmt(r['tooleyes_acc']):>6} "
              f"{_fmt(r['transient_avg']):>7}")

    # CSV export
    csv_dir = Path("<repo-root>/paper/analysis/score_cvs")
    if not csv_dir.exists():
        csv_dir = Path("<repo-root>/results_csvs")
        csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "robustbench_tc_all_models.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV written: {csv_path}")


def _fmt(v):
    if v is None:
        return "-"
    return f"{v:.4f}"


if __name__ == "__main__":
    main()
