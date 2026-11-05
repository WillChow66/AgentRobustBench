"""Recompute the paper's Transition gap reduction from the released scores."""

import csv
from pathlib import Path


def main():
    leaderboard = Path(__file__).resolve().parents[1] / "leaderboard" / "leaderboard.csv"
    with leaderboard.open(encoding="utf-8", newline="") as scores:
        by_model = {row["Model"]: row for row in csv.DictReader(scores)}

    baseline = by_model["ToolRL-Qwen2.5-3B"]
    trained = by_model["ToolRL-DR-Full (ours)"]
    baseline_gap = float(baseline["Clean"]) - float(baseline["Transition Acc."])
    trained_gap = float(trained["Clean"]) - float(trained["Transition Acc."])
    reduction = baseline_gap - trained_gap

    print(f"ToolRL-Clean Transition gap:    {100 * baseline_gap:.2f} pp")
    print(f"ToolRL-DR-Full Transition gap:  {100 * trained_gap:.2f} pp")
    print(f"Gap reduction:                  {100 * reduction:.2f} pp")
    print(f"Baseline gap closed:            {100 * reduction / baseline_gap:.1f}%")


if __name__ == "__main__":
    main()
