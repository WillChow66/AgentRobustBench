#!/usr/bin/env python3
"""
Convert RobustBench-TC training JSONLs to the parquet format expected by
ToolRL's public training code (veRL-based GRPO trainer).

Each output sample becomes:
  prompt   = ToolRL system prompt (with tools listed) + dialogue history
  response = <tool_call>{json}</tool_call>  (one per line)

Output:
  <output_dir>/train.parquet
  <output_dir>/val.parquet
  <output_dir>/raw_jsonl_for_inspection.jsonl

Usage:
  python training/convert_to_parquet.py \
      --input  training/data/perturbed.jsonl \
      --output-dir training/parquet/dr_full
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


# Verbatim copy of ToolRL's standard system prompt template. We do NOT add
# synthetic <think> reasoning since we don't have golden reasoning per sample.
TOOLRL_SYSTEM_TEMPLATE = """You are a helpful multi-turn dialogue assistant capable of leveraging tool calls to solve user tasks and provide structured chat responses.

**Available Tools**
In your response, you can use the following tools:
{tools_block}

**Steps for Each Turn**
1. **Think:** Recall relevant context and analyze the current user goal.
2. **Decide on Tool Usage:** If a tool is needed, specify the tool and its parameters.
3. **Respond Appropriately:** If a response is needed, generate one while maintaining consistency across user queries.

**Output Format**
```plaintext
<think> Your thoughts and reasoning </think>
<tool_call>
{{"name": "Tool name", "parameters": {{"Parameter name": "Parameter content", "... ...": "... ..."}}}}
{{"name": "... ...", "parameters": {{"... ...": "... ...", "... ...": "... ..."}}}}
...
</tool_call>
<response> AI's final response </response>
```

**Important Notes**
1. You must always include the `<think>` field to outline your reasoning. Provide at least one of `<tool_call>` or `<response>`. Decide whether to use `<tool_call>` (possibly multiple times), `<response>`, or both.
2. You can invoke multiple tool calls simultaneously in the `<tool_call>` fields. Each tool call should be a JSON object with a "name" field and an "parameters" field containing a dictionary of parameters. If no parameters are needed, leave the "parameters" field an empty dictionary.
3. Refer to the previous dialogue records in the history, including the user's queries, previous `<tool_call>`, `<response>`, and any tool feedback noted as `<obs>` (if exists).
"""


def _format_tools_block(tools: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for i, tool in enumerate(tools, 1):
        name = tool.get("name", "")
        desc = tool.get("description", "")
        params = tool.get("parameters", {})
        if isinstance(params, dict):
            properties = params.get("properties", params)
        else:
            properties = {}
        lines.append(f"{i}. Name: {name}")
        lines.append(f"Description: {desc}")
        lines.append(f"Parameters: {json.dumps(properties, ensure_ascii=False)}")
    return "\n".join(lines)


def _format_dialogue(conversation: List[Dict[str, str]]) -> str:
    parts: List[str] = ["**Dialogue Records History**"]
    for msg in conversation:
        role = msg.get("role", "")
        content = (msg.get("content") or "").strip()
        if role == "system":
            continue
        elif role == "user":
            parts.append(f"<user> {content} </user>")
        elif role == "assistant":
            parts.append(f"{content}")
        elif role == "tool":
            parts.append(f"<obs> {content} </obs>")
    return "\n".join(parts)


def _format_response(golden_answers: List[Dict[str, Any]]) -> str:
    if not golden_answers:
        return "<response> No tool call needed. </response>"
    lines: List[str] = ["<tool_call>"]
    for ga in golden_answers:
        name = ga.get("name") or ga.get("Action") or ""
        params = ga.get("parameters") or ga.get("arguments") or ga.get("Action_Input") or {}
        lines.append(json.dumps({"name": name, "parameters": params},
                                ensure_ascii=False))
    lines.append("</tool_call>")
    return "\n".join(lines)


def convert_one(sample: Dict[str, Any]) -> Dict[str, str]:
    def _maybe_load(v):
        return json.loads(v) if isinstance(v, str) else v

    conv = _maybe_load(sample.get("conversation", []))
    tools = _maybe_load(sample.get("tools", []))
    gas = _maybe_load(sample.get("golden_answers", []))

    instruction = TOOLRL_SYSTEM_TEMPLATE.format(
        tools_block=_format_tools_block(tools)
    )
    user_input = _format_dialogue(conv)
    response = _format_response(gas)

    prompt = instruction + "\n\n" + user_input
    return {"prompt": prompt, "response": response}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="Path to one of training/data/{clean,perturbed,mixed}.jsonl")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--val-frac", type=float, default=0.02,
                   help="Validation fraction (default 0.02 -> matches ToolRL).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    samples: List[Dict[str, Any]] = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    print(f"Loaded {len(samples)} samples from {args.input}")

    converted: List[Dict[str, str]] = [convert_one(s) for s in samples]
    keep = [r for r in converted if r["response"].strip()]
    if len(keep) < len(converted):
        print(f"  Dropped {len(converted) - len(keep)} samples with empty response")

    random.seed(args.seed)
    random.shuffle(keep)
    n_val = max(1, int(len(keep) * args.val_frac))
    val = keep[:n_val]
    train = keep[n_val:]
    print(f"Split: {len(train)} train / {len(val)} val")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "raw_jsonl_for_inspection.jsonl", "w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote raw jsonl: {out_dir / 'raw_jsonl_for_inspection.jsonl'}")

    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas required (pip install pandas pyarrow)")
        return 1
    pd.DataFrame(train).to_parquet(out_dir / "train.parquet")
    pd.DataFrame(val).to_parquet(out_dir / "val.parquet")
    print(f"Wrote: {out_dir / 'train.parquet'} ({len(train)} rows)")
    print(f"Wrote: {out_dir / 'val.parquet'} ({len(val)} rows)")

    print("\n=== Sample 0 (visual check) ===")
    print("--- prompt (first 800 chars) ---")
    print(keep[0]["prompt"][:800])
    print("\n--- response ---")
    print(keep[0]["response"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
