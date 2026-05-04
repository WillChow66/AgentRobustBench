#!/usr/bin/env python3
"""
Re-parse tool_calls from existing raw_outputs in *.predictions.jsonl.

Does NOT re-run inference — reads each prediction record, feeds raw_output
through the updated scripts/run_eval.py::parse_tool_calls, writes back with
the refreshed tool_calls. Backs up the original to ``*.predictions.jsonl.bak``
on first run.

Usage:
    python scripts/reparse_predictions.py \
        --results-root <repo-root>/results/robustbench/results \
        --models-glob '*'          # all models, or filter
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_eval import parse_tool_calls  # noqa: E402


def reparse_file(path: Path, make_backup: bool = True) -> Dict[str, int]:
    """Re-parse tool_calls in a .predictions.jsonl file in place.

    Returns counts: {total, had_calls_before, had_calls_after, changed}.
    """
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    had_before = 0
    had_after = 0
    changed = 0
    for rec in records:
        pred = rec.get("prediction") or {}
        raw = pred.get("raw_output") or ""
        old_calls = pred.get("tool_calls") or []
        bm = rec.get("benchmark") or ""

        if old_calls:
            had_before += 1

        if not raw.strip():
            had_after += 1 if old_calls else 0
            continue

        try:
            new_calls = parse_tool_calls(raw, bm)
        except Exception as e:
            # Don't blow up on a single bad record; preserve the existing parse.
            print(f"  parse error on {path.name}/{rec.get('sample_id')}: {e}",
                  file=sys.stderr)
            new_calls = old_calls

        if new_calls:
            had_after += 1
        if new_calls != old_calls:
            pred["tool_calls"] = new_calls
            rec["prediction"] = pred
            changed += 1

    if changed == 0:
        return {
            "total": len(records),
            "had_calls_before": had_before,
            "had_calls_after": had_after,
            "changed": 0,
        }

    if make_backup:
        bak = path.with_suffix(path.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(path, bak)

    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(path)

    return {
        "total": len(records),
        "had_calls_before": had_before,
        "had_calls_after": had_after,
        "changed": changed,
    }


def iter_prediction_files(model_dir: Path):
    """Yield every .predictions.jsonl under a model result directory,
    including transition_* subdirs but excluding .bak files.
    """
    for pred in sorted(model_dir.rglob("*.predictions.jsonl")):
        if pred.suffix == ".bak":
            continue
        yield pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path,
                    default=Path("<repo-root>/results/robustbench/results"))
    ap.add_argument("--models", nargs="*", default=None,
                    help="Filter to specific model directories (substring match)")
    ap.add_argument("--no-backup", action="store_true",
                    help="Skip creating .bak files (not recommended)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report counts only, do not modify files")
    args = ap.parse_args()

    model_dirs = [d for d in args.results_root.iterdir() if d.is_dir()]
    if args.models:
        model_dirs = [d for d in model_dirs
                      if any(m in d.name for m in args.models)]

    grand_total = {"total": 0, "had_calls_before": 0,
                   "had_calls_after": 0, "changed": 0}

    for mdir in sorted(model_dirs):
        print(f"\n=== {mdir.name} ===")
        m_total = {"total": 0, "had_calls_before": 0,
                   "had_calls_after": 0, "changed": 0}
        for pred in iter_prediction_files(mdir):
            if args.dry_run:
                # Simulate without writing
                counts = reparse_file_dry(pred)
            else:
                counts = reparse_file(pred, make_backup=not args.no_backup)
            for k in m_total:
                m_total[k] += counts[k]
            rel = pred.relative_to(mdir)
            delta = counts["had_calls_after"] - counts["had_calls_before"]
            if counts["changed"] > 0 or delta != 0:
                print(f"  {rel}: total={counts['total']:4d}  "
                      f"before={counts['had_calls_before']:4d}  "
                      f"after={counts['had_calls_after']:4d}  "
                      f"delta={delta:+d}  changed={counts['changed']}")
        print(f"  MODEL TOTAL: total={m_total['total']}  "
              f"before={m_total['had_calls_before']}  "
              f"after={m_total['had_calls_after']}  "
              f"delta={m_total['had_calls_after']-m_total['had_calls_before']:+d}")
        for k in grand_total:
            grand_total[k] += m_total[k]

    print(f"\n\nGRAND TOTAL across {len(model_dirs)} models:")
    print(f"  total samples:    {grand_total['total']}")
    print(f"  had_calls before: {grand_total['had_calls_before']}")
    print(f"  had_calls after:  {grand_total['had_calls_after']}")
    print(f"  delta:            {grand_total['had_calls_after'] - grand_total['had_calls_before']:+d}")
    print(f"  records changed:  {grand_total['changed']}")


def reparse_file_dry(path: Path) -> Dict[str, int]:
    had_before = had_after = changed = total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            rec = json.loads(line)
            pred = rec.get("prediction") or {}
            raw = pred.get("raw_output") or ""
            old_calls = pred.get("tool_calls") or []
            bm = rec.get("benchmark") or ""
            if old_calls:
                had_before += 1
            if not raw.strip():
                if old_calls:
                    had_after += 1
                continue
            try:
                new_calls = parse_tool_calls(raw, bm)
            except Exception:
                new_calls = old_calls
            if new_calls:
                had_after += 1
            if new_calls != old_calls:
                changed += 1
    return {"total": total, "had_calls_before": had_before,
            "had_calls_after": had_after, "changed": changed}


if __name__ == "__main__":
    main()
