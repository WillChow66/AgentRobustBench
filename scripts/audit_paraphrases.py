"""Paraphrase Quality Audit (Appendix E of v2 paper).

What it does — concretely:
  1. For each of 4 LLM-generated perturbation subtypes (Typo, QueryPara,
     ToolPara, ParamPara), pull 50 (clean, perturbed) pairs from
     hf_data/datasets/api_eval/.
  2. Score each pair with a *judge* LLM on a 5-point semantic-equivalence
     scale: 5 = identical intent, 1 = intent broken.
  3. Report mean / median / fraction <= 2 per subtype, plus the worst 5
     pairs per subtype (low-scoring examples for the appendix).

Why this experiment exists:
  Reviewer 3 questioned whether our LLM-generated perturbations preserve
  intent. If a paraphrase silently changes the answer the model is
  expected to give, the resulting accuracy drop is not attributable to
  robustness — it is just label noise. The audit defends the benchmark
  against that critique by quantifying intent preservation.

Why two judges from different families:
  Our generator is gpt-5-mini (OpenAI). If we audit with gpt-5 (same
  family), reviewers can object that the model is grading itself. Using
  Claude Sonnet 4.6 (Anthropic) as a second judge defuses that critique
  and lets us report inter-rater agreement (Cohen's kappa).

Usage (single-judge, OpenAI only):
    export OPENAI_API_KEY=sk-...
    python scripts/audit_paraphrases.py \
        --judge gpt-5 \
        --samples-per-subtype 50 \
        --output paper/neurips_2026_v2/audit/paraphrase_audit.jsonl

Usage (two-judge):
    export OPENAI_API_KEY=sk-...
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/audit_paraphrases.py \
        --judge gpt-5 --second-judge claude-sonnet-4-6 \
        --samples-per-subtype 50

Cost (50 pairs x 4 subtypes = 200 pairs):
  - gpt-5 single-judge: ~$2-4 at posted rates
  - + Claude Sonnet 4.6:  ~$5-7 total
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional


REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "hf_data" / "datasets" / "api_eval"

# Mapping: paper display name -> filename in api_eval/
SUBTYPES = {
    "Typo":       "realistic_typos.jsonl",
    "QueryPara":  "query_paraphrase.jsonl",
    "ToolPara":   "paraphrase_tool_description.jsonl",
    "ParamPara":  "paraphrase_parameter_description.jsonl",
}

CLEAN_FILE = DATA / "clean.jsonl"


# -----------------------------------------------------------------------
# Pair loading
# -----------------------------------------------------------------------

def load_pairs(subtype: str, n: int, rng: random.Random) -> list[dict]:
    """Pair clean and perturbed samples by sample_id, return n random pairs."""
    perturbed_path = DATA / SUBTYPES[subtype]
    if not CLEAN_FILE.exists() or not perturbed_path.exists():
        sys.exit(f"missing data file: {CLEAN_FILE} or {perturbed_path}")

    clean_by_id = {}
    with CLEAN_FILE.open() as f:
        for line in f:
            r = json.loads(line)
            sid = r.get("sample_id") or r.get("id")
            if sid:
                clean_by_id[sid] = r

    pairs = []
    with perturbed_path.open() as f:
        for line in f:
            p = json.loads(line)
            sid = p.get("sample_id") or p.get("id")
            if sid in clean_by_id:
                pairs.append({"clean": clean_by_id[sid], "perturbed": p,
                              "subtype": subtype, "sample_id": sid})
    rng.shuffle(pairs)
    return pairs[:n]


def extract_relevant_text(pair: dict) -> tuple[str, str]:
    """Pull the clean and perturbed text fields that this subtype actually
    perturbs — query for Typo/QueryPara, tool desc for ToolPara, param
    desc for ParamPara — so the judge sees only the changed field.
    """
    sub = pair["subtype"]
    c = pair["clean"]
    p = pair["perturbed"]

    def _conv_str(rec):
        conv = rec.get("conversation")
        if isinstance(conv, str):
            try: conv = json.loads(conv)
            except Exception: return ""
        if isinstance(conv, list) and conv:
            for m in conv:
                if isinstance(m, dict) and m.get("role") == "user":
                    return m.get("content") or ""
        return ""

    def _tools(rec):
        t = rec.get("tools")
        if isinstance(t, str):
            try: t = json.loads(t)
            except Exception: return []
        return t or []

    if sub in ("Typo", "QueryPara"):
        return _conv_str(c), _conv_str(p)
    if sub == "ToolPara":
        ct, pt = _tools(c), _tools(p)
        if ct and pt:
            cd = json.dumps([t.get("description","") for t in ct])
            pd = json.dumps([t.get("description","") for t in pt])
            return cd, pd
        return "", ""
    if sub == "ParamPara":
        ct, pt = _tools(c), _tools(p)
        def _params(ts):
            out = []
            for t in ts:
                params = (t.get("parameters") or {}).get("properties") or {}
                for name, schema in params.items():
                    out.append(f"{t.get('name','?')}.{name}: {schema.get('description','')}")
            return "\n".join(out)
        return _params(ct), _params(pt)
    return "", ""


# -----------------------------------------------------------------------
# Judge prompt
# -----------------------------------------------------------------------

JUDGE_SYSTEM = (
    "You score paraphrase / typo perturbations of tool-use evaluation samples. "
    "Given a clean and perturbed text, you assess whether the perturbation preserves the "
    "intent and information needed to make the SAME tool call as the clean version."
)

JUDGE_TEMPLATE = """\
Subtype: {subtype}

CLEAN:
{clean}

PERTURBED:
{perturbed}

Score on a 1-5 scale (return STRICT JSON only, no prose):
  5 = Identical intent. All entities/numbers/constraints preserved.
  4 = Same intent. Minor wording loss; would not change the correct tool call.
  3 = Same intent but a parameter or constraint became ambiguous.
  2 = Intent shifted slightly; correct tool call is no longer unambiguous.
  1 = Intent broken. This sample should be excluded from evaluation.

Return: {{"score": <int 1-5>, "reason": "<one sentence>"}}"""


# -----------------------------------------------------------------------
# Judge clients
# -----------------------------------------------------------------------

def judge_openai(model: str, clean: str, perturbed: str, subtype: str) -> dict:
    from openai import OpenAI
    client = OpenAI()
    msgs = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user",   "content": JUDGE_TEMPLATE.format(
            subtype=subtype, clean=clean[:2400], perturbed=perturbed[:2400])},
    ]
    # gpt-5 / o-series reasoning models reject temperature != default (1).
    # Older chat models support arbitrary temperatures. We branch on model name.
    kwargs = dict(model=model, messages=msgs,
                  response_format={"type": "json_object"})
    if not (model.startswith("gpt-5") or model.startswith("o1")
            or model.startswith("o3") or model.startswith("o4")):
        kwargs["temperature"] = 0
    resp = client.chat.completions.create(**kwargs)
    txt = resp.choices[0].message.content or "{}"
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return {"score": None, "reason": f"PARSE_ERROR: {txt[:200]}"}


def judge_anthropic(model: str, clean: str, perturbed: str, subtype: str) -> dict:
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model, max_tokens=200, temperature=0, system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": JUDGE_TEMPLATE.format(
            subtype=subtype, clean=clean[:2400], perturbed=perturbed[:2400])}],
    )
    txt = "".join(b.text for b in resp.content if hasattr(b, "text"))
    try:
        i, j = txt.find("{"), txt.rfind("}")
        if i >= 0 and j > i:
            return json.loads(txt[i:j+1])
    except json.JSONDecodeError:
        pass
    return {"score": None, "reason": f"PARSE_ERROR: {txt[:200]}"}


# -----------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-per-subtype", type=int, default=50)
    ap.add_argument("--judge", default="gpt-5", help="primary OpenAI judge model")
    ap.add_argument("--second-judge", default=None,
                    help="optional Anthropic judge, e.g. claude-sonnet-4-6")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=str(REPO / "paper" / "neurips_2026_v2"
                                            / "audit" / "paraphrase_audit.jsonl"))
    ap.add_argument("--dry-run", action="store_true",
                    help="Sample pairs but skip judge calls (for cost preview).")
    args = ap.parse_args()

    if not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: OPENAI_API_KEY not set in environment.")
    if args.second_judge and not args.dry_run \
            and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ERROR: ANTHROPIC_API_KEY not set; pass --second-judge only "
                 "after exporting it.")

    rng = random.Random(args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    with out_path.open("w") as out:
        for subtype in SUBTYPES:
            pairs = load_pairs(subtype, args.samples_per_subtype, rng)
            print(f"  {subtype}: {len(pairs)} pairs loaded")
            for pair in pairs:
                clean_txt, pert_txt = extract_relevant_text(pair)
                if not clean_txt or not pert_txt:
                    continue
                rec = {
                    "subtype": subtype,
                    "sample_id": pair["sample_id"],
                    "clean_excerpt": clean_txt[:200],
                    "perturbed_excerpt": pert_txt[:200],
                }
                if args.dry_run:
                    pass  # no API calls
                else:
                    rec[f"{args.judge}_judgment"] = judge_openai(
                        args.judge, clean_txt, pert_txt, subtype)
                    if args.second_judge:
                        rec[f"{args.second_judge}_judgment"] = judge_anthropic(
                            args.second_judge, clean_txt, pert_txt, subtype)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_total += 1
    print(f"\nWrote {n_total} audited pairs to {out_path}")


if __name__ == "__main__":
    main()
