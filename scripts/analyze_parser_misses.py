#!/usr/bin/env python3
"""
Sample raw_outputs from (model, benchmark) pairs where tool_calls=[] but
raw_output is non-empty. This surfaces parser bugs — cases where the model
emitted something that looks tool-call-shaped but our parser didn't catch it.

Groups examples by benchmark and shows 3 representative samples per
(model, benchmark, perturbation) bucket with high miss rate.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


RESULTS_ROOT = Path("<repo-root>/results/robustbench/results")


def iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def collect_misses(model_dir: Path) -> List[Dict[str, Any]]:
    """Collect (benchmark, perturbation, sample_id, raw_output) for all samples
    where raw_output != "" but tool_calls == [].
    """
    misses = []
    for pred_file in sorted(model_dir.glob("*.predictions.jsonl")):
        pert = pred_file.stem.replace(".predictions", "")
        for rec in iter_jsonl(pred_file):
            pred = rec.get("prediction") or {}
            raw = pred.get("raw_output") or ""
            tcs = pred.get("tool_calls") or []
            if raw.strip() and not tcs:
                misses.append({
                    "perturbation": pert,
                    "benchmark": rec.get("benchmark", ""),
                    "category": rec.get("category", ""),
                    "sample_id": rec.get("sample_id") or rec.get("id"),
                    "raw_output": raw,
                    "golden": rec.get("golden_answers"),
                })
    # transition subdirs also
    for t_dir in sorted(model_dir.glob("transition_*")):
        pert = t_dir.name
        for pred_file in t_dir.glob("*.predictions.jsonl"):
            for rec in iter_jsonl(pred_file):
                pred = rec.get("prediction") or {}
                raw = pred.get("raw_output") or ""
                tcs = pred.get("tool_calls") or []
                if raw.strip() and not tcs:
                    misses.append({
                        "perturbation": pert,
                        "benchmark": rec.get("benchmark", ""),
                        "category": rec.get("category", ""),
                        "sample_id": rec.get("sample_id") or rec.get("id"),
                        "raw_output": raw,
                        "golden": rec.get("golden_answers"),
                    })
    return misses


def summarize_patterns(raw: str) -> List[str]:
    """Rough heuristics to classify the shape of the raw output."""
    patterns = []
    if "<think>" in raw or "</think>" in raw:
        patterns.append("HAS_THINK_TAG")
    if re.search(r"<tool_call>", raw):
        patterns.append("HAS_TOOL_CALL_TAG")
    if re.search(r"```(?:json|python)?", raw):
        patterns.append("HAS_CODE_FENCE")
    if re.search(r"```tool_code", raw):
        patterns.append("HAS_TOOL_CODE_FENCE")
    if re.search(r"Action\s*:", raw, re.I):
        patterns.append("HAS_ACTION")
    if re.search(r"\[\s*\{", raw):
        patterns.append("HAS_JSON_LIST")
    if re.search(r"\{\s*\"(?:name|function|tool|action)\"", raw):
        patterns.append("HAS_NAMED_JSON")
    if re.search(r"^[A-Za-z_][\w.]*\s*\(", raw.strip()):
        patterns.append("STARTS_WITH_CALL")
    if re.search(r"\[\s*[A-Za-z_][\w.]*\s*\(", raw):
        patterns.append("HAS_BFCL_LIST")
    if re.search(r"<tool\b", raw) and "<tool_call>" not in raw:
        patterns.append("HAS_OTHER_XML")
    if re.search(r"print\(", raw):
        patterns.append("HAS_PYTHON_PRINT")
    if re.search(r"^(I |Let me|Sure|Here|Based on|To )", raw.strip()[:50], re.I):
        patterns.append("STARTS_WITH_PROSE")
    if re.search(r"Thought\s*:", raw):
        patterns.append("HAS_THOUGHT")
    if re.search(r"Final Answer", raw, re.I):
        patterns.append("HAS_FINAL_ANSWER")
    if len(raw) < 30:
        patterns.append("SHORT")
    return patterns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                    help="Filter to specific model dirs (basename match)")
    ap.add_argument("--benchmarks", nargs="*", default=None,
                    help="Filter to specific benchmarks: bfcl_v3, apibank, rotbench, toolalpaca, tooleyes")
    ap.add_argument("--samples-per-bucket", type=int, default=3,
                    help="How many raw samples to print per (model, benchmark) bucket")
    ap.add_argument("--min-miss-rate", type=float, default=0.1,
                    help="Only show (model, benchmark) buckets with at least this many misses")
    ap.add_argument("--output", type=Path, default=None,
                    help="Optional: save full JSON dump to this path")
    args = ap.parse_args()

    all_model_dirs = [d for d in RESULTS_ROOT.iterdir() if d.is_dir()]
    if args.models:
        all_model_dirs = [d for d in all_model_dirs
                          if any(m in d.name for m in args.models)]

    # model -> benchmark -> list of miss records
    bucketed = defaultdict(lambda: defaultdict(list))
    pattern_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for mdir in all_model_dirs:
        print(f"\n=== Scanning {mdir.name} ===")
        misses = collect_misses(mdir)
        print(f"    total misses (raw≠∅, tool_calls=[]): {len(misses)}")
        for m in misses:
            bm = m["benchmark"]
            if args.benchmarks and bm not in args.benchmarks:
                continue
            bucketed[mdir.name][bm].append(m)
            for p in summarize_patterns(m["raw_output"]):
                pattern_counts[mdir.name][bm][p] += 1

    # Print per-bucket samples
    for model_name, bm_map in bucketed.items():
        print(f"\n\n========================================")
        print(f"  {model_name}")
        print(f"========================================")
        for bm, rows in sorted(bm_map.items(), key=lambda kv: -len(kv[1])):
            if len(rows) < 5:
                continue
            print(f"\n  [{bm}] {len(rows)} misses")
            # Pattern histogram
            counts = pattern_counts[model_name][bm]
            top = sorted(counts.items(), key=lambda kv: -kv[1])[:10]
            print(f"    Patterns: {top}")
            # Show samples, prefer diverse perturbations
            by_pert = defaultdict(list)
            for r in rows:
                by_pert[r["perturbation"]].append(r)
            shown = 0
            for pert, group in sorted(by_pert.items()):
                if shown >= args.samples_per_bucket:
                    break
                r = group[0]
                raw = r["raw_output"][:500].replace("\n", "\\n")
                print(f"\n    --- perturbation={pert}, sample={r['sample_id']} ---")
                print(f"    raw_output[:500]: {raw}")
                if r["golden"]:
                    golden = json.dumps(r["golden"])[:200]
                    print(f"    golden: {golden}")
                shown += 1

    # Optional full dump
    if args.output:
        dump = {
            m: {bm: rows for bm, rows in bms.items()}
            for m, bms in bucketed.items()
        }
        with open(args.output, "w") as f:
            json.dump(dump, f, indent=2, default=str)
        print(f"\nFull dump written to {args.output}")


if __name__ == "__main__":
    main()
