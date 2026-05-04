#!/usr/bin/env python3
"""
RobustBench-TC results evaluator.

Consumes the `.predictions.jsonl` files produced by `run_eval.py`, dispatches
scoring by `sample.eval_config.method`, and aggregates metrics by
(benchmark, perturbation_type, mdp_category).

Scoring is fully deterministic — no external LLM judge is used:
  - exact_match  -> BFCL-style name + numeric-tolerant parameter equality
  - rule_based   -> RoTBench TS / PI / CF sub-metrics
  - gpt_judge    -> replaced deterministically:
        metric="relevance"          -> name-only match (ToolEyes)
        metric="complete_accuracy"  -> exact_match    (ToolAlpaca)

Usage:
    python scripts/eval_results.py \
        --predictions results/my-model \
        --output results/my-model/scores.json

    # Score a subset:
    python scripts/eval_results.py \
        --predictions results/my-model \
        --files clean.predictions.jsonl realistic_typos.predictions.jsonl \
        --output results/my-model/scores.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


# ────────────────────────────────────────────────────────────────
# Exact match (bfcl_v3, apibank)
# ────────────────────────────────────────────────────────────────

def _normalize_params(params: Any) -> Dict[str, Any]:
    """Canonicalize parameter dicts for comparison."""
    if isinstance(params, dict):
        out: Dict[str, Any] = {}
        for k, v in params.items():
            if isinstance(v, str):
                out[k] = v.strip()
            elif isinstance(v, list):
                # Represent lists in a comparable way; keep elements coerced
                out[k] = [x.strip() if isinstance(x, str) else x for x in v]
            else:
                out[k] = v
        return out
    return {}


def _try_number(x: Any) -> Optional[float]:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x.strip())
        except (ValueError, TypeError):
            return None
    return None


def _is_empty_default(value: Any) -> bool:
    """Return True if `value` represents a BFCL 'optional/omittable' default.

    BFCL ground-truth lists (e.g. ``"unit": [""]``) are collapsed to scalars
    by the unified-sample builder, so a GT of ``""`` or ``None`` signals
    'this param is optional and may be omitted by the model'. Empty
    containers count too.
    """
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, (list, tuple, dict)) and len(value) == 0:
        return True
    return False


def _params_equal(a: Any, b: Any) -> bool:
    """Compare parameter dicts with BFCL-compatible tolerance.

    Rules:
      - Numeric-tolerant: '1500', '1500.0', 1500, 1500.0 all equal.
      - Optional-param omission: keys present in one side with an
        "empty default" value (``""``, ``None``, ``[]``, ``{}``) may be
        absent from the other side. Matches BFCL's ground-truth convention
        of listing ``""`` as an acceptable omission for optional params.

    ``a`` is the predicted dict, ``b`` is the ground-truth dict.
    """
    na = _normalize_params(a)
    nb = _normalize_params(b)

    # Key union, but drop keys whose value is an empty-default on either side
    # when the other side doesn't have that key.
    all_keys = set(na) | set(nb)
    for k in all_keys:
        if k in na and k in nb:
            va, vb = na[k], nb[k]
            nva = _try_number(va)
            nvb = _try_number(vb)
            if nva is not None and nvb is not None:
                if nva != nvb:
                    return False
                continue
            if va != vb:
                # Allow matching an empty-default on one side to an absent
                # key on the other — but if both are present and unequal,
                # and at least one is empty-default, we treat that as
                # "GT says optional and model provided a value", accept it.
                if _is_empty_default(va) or _is_empty_default(vb):
                    continue
                return False
        else:
            # Key only in one side. Allowed iff its value is an empty default.
            only_val = na.get(k, nb.get(k))
            if not _is_empty_default(only_val):
                return False
    return True


def _ga_name(ga: Dict[str, Any]) -> str:
    return ga.get("name") or ga.get("Action") or ""


def _ga_params(ga: Dict[str, Any]) -> Dict[str, Any]:
    return ga.get("parameters") or ga.get("arguments") or ga.get("Action_Input") or {}


def _params_equal_bfcl(pred_params: Dict[str, Any],
                       gt_params: Dict[str, Any]) -> bool:
    """BFCL-flavored param comparison.

    BFCL's native ground-truth lists alternatives per param
    (``"unit": ["metric", ""]`` = "either 'metric' or omit it entirely").
    The unified HF schema collapses those lists to a single scalar, so
    we can no longer distinguish "mandatory" from "optional with default".
    To match BFCL's official checker's leniency, allow the prediction to
    OMIT any key that GT has. For keys present in both, require numeric-
    tolerant equality (same as the strict matcher).
    """
    nref = _normalize_params(gt_params)
    nprd = _normalize_params(pred_params)
    for k, vp in nprd.items():
        if k not in nref:
            return False          # pred hallucinated an arg
        vg = nref[k]
        nvp = _try_number(vp)
        nvg = _try_number(vg)
        if nvp is not None and nvg is not None:
            if nvp != nvg:
                return False
            continue
        if vp != vg:
            return False
    return True


def score_exact_match(pred_calls: List[Dict[str, Any]],
                      golden_answers: List[Dict[str, Any]],
                      benchmark: str = "") -> Dict[str, Any]:
    """A prediction is correct if for every GA, at least one predicted call
    matches both name and parameters.

    For BFCL single-turn 'multiple': GA list has one entry; at least one pred
    must match it. For longer GA lists: every GA must be covered.

    ``benchmark="bfcl_v3"`` triggers BFCL-lenient param matching (allow
    omitting GT keys, per BFCL's optional-param convention); otherwise
    strict matching with empty-default leniency only.
    """
    if not golden_answers:
        correct = (len(pred_calls) == 0)
        return {"correct": correct, "score": 1.0 if correct else 0.0,
                "method": "exact_match", "matched": 0, "expected": 0}

    eq = _params_equal_bfcl if benchmark == "bfcl_v3" else _params_equal

    matched = 0
    for ga in golden_answers:
        gname = _ga_name(ga)
        gparams = _ga_params(ga)
        hit = False
        for pc in pred_calls:
            pname = pc.get("name", "")
            pparams = pc.get("parameters") or pc.get("arguments") or {}
            if pname == gname and eq(pparams, gparams):
                hit = True
                break
        if hit:
            matched += 1

    total = len(golden_answers)
    correct = (matched == total)
    return {
        "correct": correct,
        "score": matched / total if total else 0.0,
        "method": "exact_match",
        "matched": matched,
        "expected": total,
    }


# ────────────────────────────────────────────────────────────────
# Rule-based (rotbench: TS/PI/CF)
# ────────────────────────────────────────────────────────────────

def score_rule_based(pred_calls: List[Dict[str, Any]],
                     golden_answers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """RoTBench TS/PI/CF metrics, evaluated against first GA."""
    if not golden_answers:
        no_calls = (len(pred_calls) == 0)
        return {"correct": no_calls, "score": 1.0 if no_calls else 0.0,
                "method": "rule_based",
                "tool_selection": no_calls,
                "parameter_identification": no_calls,
                "content_filling": no_calls}

    ga = golden_answers[0]
    gname = _ga_name(ga)
    gparams = _ga_params(ga)

    ts = False
    pi = False
    cf = False
    for pc in pred_calls:
        pname = pc.get("name", "")
        pparams = pc.get("parameters") or pc.get("arguments") or {}
        if pname == gname:
            ts = True
            if set((pparams or {}).keys()) == set((gparams or {}).keys()):
                pi = True
                if _params_equal(pparams, gparams):
                    cf = True
            # Don't break — different pred call might satisfy more sub-metrics
    score = (int(ts) + int(pi) + int(cf)) / 3.0
    return {
        "correct": cf,
        "score": score,
        "method": "rule_based",
        "tool_selection": ts,
        "parameter_identification": pi,
        "content_filling": cf,
    }


# ────────────────────────────────────────────────────────────────
# Name-only relevance (ToolEyes) — replaces GPT judge deterministically
# ────────────────────────────────────────────────────────────────

def score_relevance(pred_calls: List[Dict[str, Any]],
                    golden_answers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Name-only match for the ToolEyes 'relevance' metric.

    ToolEyes GAs typically carry empty params (the benchmark only asks
    whether the model picked an appropriate tool). Anything name-equal to a
    GA is considered correct.
    """
    if not golden_answers:
        correct = (len(pred_calls) == 0)
        return {"correct": correct, "score": 1.0 if correct else 0.0,
                "method": "name_match"}
    gt_names = {_ga_name(ga) for ga in golden_answers}
    pred_names = {pc.get("name", "") for pc in pred_calls}
    hit = bool(gt_names & pred_names)
    return {"correct": hit, "score": 1.0 if hit else 0.0,
            "method": "name_match"}


# ────────────────────────────────────────────────────────────────
# Score one prediction record
# ────────────────────────────────────────────────────────────────

def score_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch by sample eval_config. All branches deterministic.

    Samples whose original eval_config.method is "gpt_judge" are scored
    without an LLM:
      - metric == "relevance"          -> name-only match (ToolEyes)
      - metric == "complete_accuracy"  -> exact match    (ToolAlpaca)
    """
    eval_cfg = record.get("eval_config") or {}
    method = eval_cfg.get("method", "exact_match")
    metrics = eval_cfg.get("metrics") or ["complete_accuracy"]
    metric = metrics[0] if metrics else "complete_accuracy"

    pred = record.get("prediction") or {}
    pred_calls = pred.get("tool_calls") or []
    gas = record.get("golden_answers") or []

    if method == "exact_match":
        return score_exact_match(pred_calls, gas, benchmark=record.get("benchmark", ""))

    if method == "rule_based":
        return score_rule_based(pred_calls, gas)

    if method == "gpt_judge":
        if metric == "relevance":
            return score_relevance(pred_calls, gas)
        return score_exact_match(pred_calls, gas)

    # Unknown method → exact match as safest default
    return score_exact_match(pred_calls, gas)


# ────────────────────────────────────────────────────────────────
# File- and corpus-level scoring + aggregation
# ────────────────────────────────────────────────────────────────

def score_file(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    out = []
    for rec in records:
        sc = score_record(rec)
        out.append({
            "id": rec.get("id"),
            "benchmark": rec.get("benchmark"),
            "category": rec.get("category"),
            "perturbation": rec.get("perturbation"),
            "score": sc,
        })
    return out


def aggregate(scored_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate into (benchmark, perturbation.type, mdp_category) buckets."""
    # Per-bucket running totals
    def _mk():
        return {"n": 0, "correct": 0, "score_sum": 0.0}

    overall = _mk()
    by_benchmark = defaultdict(_mk)
    by_perturbation = defaultdict(_mk)  # keyed by perturbation.type (or "clean")
    by_mdp = defaultdict(_mk)
    by_bench_pert = defaultdict(_mk)    # (benchmark, perturbation.type)

    for rec in scored_records:
        sc = rec["score"]
        n = 1
        c = 1 if sc.get("correct") else 0
        s = float(sc.get("score", 0.0))

        overall["n"] += n
        overall["correct"] += c
        overall["score_sum"] += s

        b = rec.get("benchmark", "?")
        by_benchmark[b]["n"] += n
        by_benchmark[b]["correct"] += c
        by_benchmark[b]["score_sum"] += s

        pinfo = rec.get("perturbation") or {}
        ptype = pinfo.get("type") or "clean"
        mdp = pinfo.get("mdp_category") or "clean"

        by_perturbation[ptype]["n"] += n
        by_perturbation[ptype]["correct"] += c
        by_perturbation[ptype]["score_sum"] += s

        by_mdp[mdp]["n"] += n
        by_mdp[mdp]["correct"] += c
        by_mdp[mdp]["score_sum"] += s

        k = f"{b}::{ptype}"
        by_bench_pert[k]["n"] += n
        by_bench_pert[k]["correct"] += c
        by_bench_pert[k]["score_sum"] += s

    def _finalize(d):
        out = {}
        for k, v in d.items():
            n = v["n"]
            out[k] = {
                "n": n,
                "correct": v["correct"],
                "accuracy": v["correct"] / n if n else 0.0,
                "mean_score": v["score_sum"] / n if n else 0.0,
            }
        return out

    return {
        "overall": {
            "n": overall["n"],
            "correct": overall["correct"],
            "accuracy": (overall["correct"] / overall["n"])
                         if overall["n"] else 0.0,
            "mean_score": (overall["score_sum"] / overall["n"])
                           if overall["n"] else 0.0,
        },
        "by_benchmark": _finalize(by_benchmark),
        "by_perturbation": _finalize(by_perturbation),
        "by_mdp_category": _finalize(by_mdp),
        "by_benchmark_x_perturbation": _finalize(by_bench_pert),
    }


# ────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RobustBench-TC scorer.")
    parser.add_argument("--predictions", required=True,
                        help="Directory containing *.predictions.jsonl files.")
    parser.add_argument("--files", nargs="*", default=None,
                        help="Subset of predictions filenames to score.")
    parser.add_argument("--output", required=True,
                        help="Path for aggregated scores JSON.")
    parser.add_argument("--per-sample-output", default=None,
                        help="Optional: path for per-sample scored records jsonl.")
    args = parser.parse_args()

    pred_dir = Path(args.predictions)
    if args.files:
        files = [pred_dir / f for f in args.files]
    else:
        # Recursive so transition_{type}/ subdirectories are included.
        files = sorted(pred_dir.rglob("*.predictions.jsonl"))

    if not files:
        print(f"ERROR: no *.predictions.jsonl files in {pred_dir}",
              file=sys.stderr)
        sys.exit(1)

    all_scored: List[Dict[str, Any]] = []
    per_file_agg: Dict[str, Any] = {}
    for f in files:
        if not f.exists():
            print(f"WARN: missing {f}, skipping", file=sys.stderr)
            continue
        scored = score_file(f)
        all_scored.extend(scored)
        per_file_agg[f.name] = aggregate(scored)

    overall = aggregate(all_scored)
    summary = {
        "overall": overall,
        "per_file": per_file_agg,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    if args.per_sample_output:
        psp = Path(args.per_sample_output)
        psp.parent.mkdir(parents=True, exist_ok=True)
        with open(psp, "w", encoding="utf-8") as f:
            for rec in all_scored:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Console summary
    ov = overall["overall"]
    print(f"\n=== eval_results.py summary ===")
    print(f"n={ov['n']}  accuracy={ov['accuracy']:.4f}  "
          f"mean_score={ov['mean_score']:.4f}")
    print("\nBy benchmark:")
    for k, v in sorted(overall["by_benchmark"].items()):
        print(f"  {k:12s} n={v['n']:4d} acc={v['accuracy']:.4f} "
              f"score={v['mean_score']:.4f}")
    print("\nBy MDP category:")
    for k, v in sorted(overall["by_mdp_category"].items()):
        print(f"  {k:12s} n={v['n']:4d} acc={v['accuracy']:.4f} "
              f"score={v['mean_score']:.4f}")
    print("\nBy perturbation type:")
    for k, v in sorted(overall["by_perturbation"].items()):
        print(f"  {k:32s} n={v['n']:4d} acc={v['accuracy']:.4f} "
              f"score={v['mean_score']:.4f}")


if __name__ == "__main__":
    main()
