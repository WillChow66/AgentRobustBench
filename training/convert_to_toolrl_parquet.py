#!/usr/bin/env python3
"""
Convert our perturbed/mixed train data to ToolRL-compatible parquet format.

ToolRL parquet schema:
  - data_source: str ("rlla")
  - prompt: list[dict] = [{"role":"system","content":instruction}, {"role":"user","content":input}]
  - ability: str ("math")
  - reward_model: dict = {"style":"rule", "ground_truth": "<think>...<tool_call>..."}
  - extra_info: dict = {"split":..., "index":..., "instruction":..., "input":..., "output":...}

Key conversion logic:
  1. Tools (JSON) → plain text instruction (ToolRL format)
  2. golden_answers → <think>template</think>\n<tool_call>\n{json}\n</tool_call>
  3. response_only (GA=[]) → use original ToolRL response text
  4. conversation → input text with <user> tags
"""

import json
import copy
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

np.random.seed(42)

# ═══════════════════════════════════════════════════════════════
# Paths (override via env vars; defaults assume the layout below)
# ═══════════════════════════════════════════════════════════════
import os as _os
# Directory holding our source JSONLs (perturbed.jsonl, mixed.jsonl) in
# the RobustBench-TC training schema. These intermediate JSONLs are not
# redistributed in the anonymous release; the final parquets shipped in
# `training/data/` were produced by running this script on them.
TRAIN_DIR = Path(_os.environ.get("TRAIN_DIR", "<your-jsonl-source-dir>"))
# Path to the original public ToolRL rlla_4k raw json (clone the public
# ToolRL repo and point this at dataset/rlla_4k_raw/rlla_rl.json).
TOOLRL_RAW = Path(_os.environ.get(
    "TOOLRL_RAW",
    "<path-to-public-toolrl>/dataset/rlla_4k_raw/rlla_rl.json",
))
# Where parquets are written.
OUTPUT_DIR = Path(_os.environ.get("OUTPUT_DIR", "training/data"))
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


# ═══════════════════════════════════════════════════════════════
# Load original ToolRL data (for response_only text + instruction template)
# ═══════════════════════════════════════════════════════════════
print("Loading original ToolRL data...")
with open(TOOLRL_RAW) as f:
    toolrl_raw = json.load(f)

# Build index: original_index → raw sample
toolrl_by_idx = {i: s for i, s in enumerate(toolrl_raw)}

# Extract instruction template (everything before tool definitions)
# The system prompt has a fixed header + tool list + fixed footer
SYSTEM_HEADER = """You are a helpful multi-turn dialogue assistant capable of leveraging tool calls to solve user tasks and provide structured chat responses.

**Available Tools**
In your response, you can use the following tools:
"""

SYSTEM_FOOTER = """
You should respond with a structured format including:
1. A thinking section `<think>...</think>` where you analyze the user's request.
2. If a tool call is needed, use `<tool_call>...</tool_call>` with valid JSON for each tool call.
3. If no tool is needed or after receiving tool results, provide a direct response in `<response>...</response>`.

**Output Structure**
```
<think> [Your reasoning process] </think>
<tool_call>
{"name": "tool_name", "parameters": {"param1": "value1"}}
</tool_call>
```
or
```
<think> [Your reasoning process] </think>
<response> [Your response to the user] </response>
```"""

# Extract the actual footer from ToolRL's instruction
# It varies slightly per sample, let's get the canonical one
_sample_instr = toolrl_raw[1]["instruction"]
# Find where "You should respond" starts
_footer_start = _sample_instr.find("You should respond")
if _footer_start > 0:
    SYSTEM_FOOTER = "\n" + _sample_instr[_footer_start:]
    print(f"  Extracted footer ({len(SYSTEM_FOOTER)} chars)")


# ═══════════════════════════════════════════════════════════════
# Helper: Convert our JSON tools → ToolRL plain text format
# ═══════════════════════════════════════════════════════════════
def tools_to_instruction(tools):
    """Convert our JSON tools list to ToolRL's instruction text format."""
    tool_lines = []
    for i, tool in enumerate(tools, 1):
        name = tool["name"]
        desc = tool.get("description", "")

        # Convert parameters to ToolRL format
        # ToolRL uses: {"param_name": {"description": "...", "type": "str", "default": "..."}}
        params = tool.get("parameters", {})
        props = params.get("properties", {})
        required = set(params.get("required", []))

        param_dict = {}
        for pname, pinfo in props.items():
            entry = {}
            if "description" in pinfo:
                entry["description"] = pinfo["description"]

            ptype = pinfo.get("type", "string")
            if pname not in required:
                entry["type"] = f"{ptype}, optional"
                if "default" in pinfo:
                    entry["default"] = str(pinfo["default"])
            else:
                entry["type"] = ptype

            param_dict[pname] = entry

        params_str = json.dumps(param_dict, ensure_ascii=False)
        tool_lines.append(f"{i}. Name: {name}\nDescription: {desc}\nParameters: {params_str}")

    return SYSTEM_HEADER + "\n".join(tool_lines) + SYSTEM_FOOTER


# ═══════════════════════════════════════════════════════════════
# Helper: Build ground_truth output string
# ═══════════════════════════════════════════════════════════════
THINK_TOOL_CALL = "I should use the appropriate tool with proper parameters to respond to the user's need."
THINK_RESPONSE = "I should directly respond to the user's need."


def build_ground_truth(sample, orig_idx):
    """
    Build ToolRL-format ground_truth string from our sample.

    For tool_call samples: <think>template</think>\n<tool_call>\n{json lines}\n</tool_call>
    For response_only: use original ToolRL output (has actual response text)
    """
    ga = sample.get("golden_answers", [])
    if isinstance(ga, str):
        ga = json.loads(ga)

    if not ga:
        # Response-only: use original ToolRL output
        if orig_idx is not None and orig_idx in toolrl_by_idx:
            return toolrl_raw[orig_idx]["output"]
        else:
            # Fallback: generic response
            return f"<think> {THINK_RESPONSE} </think>\n<response> I cannot fulfill this request with the available tools. </response>"

    # Tool call: build from our golden_answers
    tc_lines = []
    for call in ga:
        tc_lines.append(json.dumps({"name": call["name"], "parameters": call.get("parameters", {})}, ensure_ascii=False))

    tc_block = "\n".join(tc_lines)
    return f"<think> {THINK_TOOL_CALL} </think>\n<tool_call>\n{tc_block}\n</tool_call>"


# ═══════════════════════════════════════════════════════════════
# Helper: Build input text (conversation → ToolRL input format)
# ═══════════════════════════════════════════════════════════════
def build_input(sample, orig_idx):
    """
    Build ToolRL-format input text from our conversation.

    For perturbed data with query_paraphrase, use our modified conversation.
    For others, we can use our conversation (same query, just formatted differently).
    For multi-turn originals, preserve the full history from ToolRL.
    """
    conv = sample.get("conversation", [])
    if isinstance(conv, str):
        conv = json.loads(conv)

    pert = sample.get("perturbation", {})
    if isinstance(pert, str):
        pert = json.loads(pert)
    pert_type = pert.get("type", "none")

    # For query_paraphrase: our conversation has the modified query
    # For all others: the query is the same as original

    # Check if original is multi-turn
    if orig_idx is not None and orig_idx in toolrl_by_idx:
        orig_input = toolrl_raw[orig_idx]["input"]
        is_multi_turn = orig_input.count("<user>") > 1

        if is_multi_turn:
            if pert_type == "query_paraphrase":
                # Replace last <user> block with our paraphrased query
                last_user_start = orig_input.rfind("<user>")
                last_user_end = orig_input.find("</user>", last_user_start) + len("</user>")
                our_query = conv[-1]["content"] if conv else ""
                return orig_input[:last_user_start] + f"<user> {our_query} </user>" + orig_input[last_user_end:]
            else:
                # Use original multi-turn input (tools are perturbed in instruction, not input)
                return orig_input

    # Single-turn: wrap our query in ToolRL format
    if conv:
        query = conv[-1]["content"] if conv[-1]["role"] == "user" else conv[0]["content"]
    else:
        query = ""

    return f"**Dialogue Records History**\n<user> {query} </user>"


# ═══════════════════════════════════════════════════════════════
# Convert one file
# ═══════════════════════════════════════════════════════════════
def convert_file(input_path, output_path, split="train"):
    """Convert a JSONL file to ToolRL parquet format."""
    print(f"\nConverting {input_path.name} → {output_path.name}")

    data = []
    with open(input_path) as f:
        samples = [json.loads(l) for l in f if l.strip()]

    stats = Counter()

    for idx, sample in enumerate(samples):
        # Parse fields
        tools = sample.get("tools", [])
        if isinstance(tools, str):
            tools = json.loads(tools)

        ga = sample.get("golden_answers", [])
        if isinstance(ga, str):
            ga = json.loads(ga)

        pert = sample.get("perturbation", {})
        if isinstance(pert, str):
            pert = json.loads(pert)

        src = sample.get("source", {})
        if isinstance(src, str):
            src = json.loads(src)

        orig_idx = src.get("original_index")

        # Build instruction (system prompt with perturbed tools)
        instruction = tools_to_instruction(tools)

        # Build input (user message / dialogue history)
        input_text = build_input(sample, orig_idx)

        # Build ground_truth output
        output = build_ground_truth(sample, orig_idx)

        # Track stats
        if ga:
            stats["tool_call"] += 1
        else:
            stats["response_only"] += 1
        stats[pert.get("type", "none")] += 1

        row = {
            "data_source": "rlla",
            "prompt": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": input_text},
            ],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": output,
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "instruction": instruction,
                "input": input_text,
                "output": output,
                "perturbation": pert,
                "original_id": sample.get("id", ""),
            },
        }
        data.append(row)

    df = pd.DataFrame(data)
    df.to_parquet(output_path)

    print(f"  Saved {len(df)} samples")
    print(f"  Stats: {dict(stats)}")

    # Verify format
    verify_parquet(output_path)

    return df


def verify_parquet(path):
    """Verify parquet matches ToolRL expected format."""
    df = pd.read_parquet(path)

    # Check columns
    expected_cols = {"data_source", "prompt", "ability", "reward_model", "extra_info"}
    assert set(df.columns) == expected_cols, f"Columns mismatch: {set(df.columns)} vs {expected_cols}"

    # Check a few rows
    errors = 0
    for i in range(min(5, len(df))):
        row = df.iloc[i]

        # data_source must contain "rlla"
        assert "rlla" in row["data_source"], f"Row {i}: data_source missing 'rlla'"

        # prompt must be list of 2 dicts
        prompt = row["prompt"]
        assert isinstance(prompt, (list, np.ndarray)), f"Row {i}: prompt not list"
        assert len(prompt) == 2, f"Row {i}: prompt has {len(prompt)} messages, expected 2"
        assert prompt[0]["role"] == "system", f"Row {i}: first message not system"
        assert prompt[1]["role"] == "user", f"Row {i}: second message not user"

        # reward_model must have ground_truth
        rm = row["reward_model"]
        assert "ground_truth" in rm, f"Row {i}: missing ground_truth"
        gt = rm["ground_truth"]
        assert "<think>" in gt, f"Row {i}: ground_truth missing <think>"
        assert "</think>" in gt, f"Row {i}: ground_truth missing </think>"
        assert "<tool_call>" in gt or "<response>" in gt, f"Row {i}: ground_truth missing tool_call or response"

    # Compare with original ToolRL parquet structure (optional sanity check;
    # set TOOLRL_PARQUET to the path of public ToolRL's preprocessed parquet
    # to enable, otherwise this comparison is skipped).
    _orig_pq = _os.environ.get("TOOLRL_PARQUET")
    if _orig_pq is None or not Path(_orig_pq).exists():
        print("  (skipping dtype comparison; set TOOLRL_PARQUET to enable)")
        return
    orig_df = pd.read_parquet(_orig_pq)

    # Check dtypes match
    for col in expected_cols:
        our_dtype = str(df[col].dtype)
        orig_dtype = str(orig_df[col].dtype)
        if our_dtype != orig_dtype:
            print(f"  ⚠ dtype mismatch for {col}: ours={our_dtype}, orig={orig_dtype}")

    print(f"  ✓ Format verification passed ({len(df)} rows)")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Convert perturbed.jsonl
    convert_file(
        TRAIN_DIR / "perturbed.jsonl",
        OUTPUT_DIR / "train.parquet",
        split="train",
    )

    # Create a small test split (2% of perturbed, matching ToolRL's split ratio)
    df_train = pd.read_parquet(OUTPUT_DIR / "train.parquet")
    n_test = max(1, int(len(df_train) * 0.02))

    # Shuffle and split
    indices = np.random.permutation(len(df_train))
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    df_test = df_train.iloc[test_indices].copy()
    df_train_final = df_train.iloc[train_indices].copy()

    # Update split markers in extra_info
    test_rows = df_test.to_dict("records")
    for row in test_rows:
        row["extra_info"] = {**row["extra_info"], "split": "test"}
    df_test = pd.DataFrame(test_rows)

    df_train_final.to_parquet(OUTPUT_DIR / "train.parquet")
    df_test.to_parquet(OUTPUT_DIR / "test.parquet")

    print(f"\n{'='*60}")
    print(f"Final output:")
    print(f"  {OUTPUT_DIR / 'train.parquet'}: {len(df_train_final)} samples")
    print(f"  {OUTPUT_DIR / 'test.parquet'}: {len(df_test)} samples")

    # Also convert mixed.jsonl as alternative
    convert_file(
        TRAIN_DIR / "mixed.jsonl",
        OUTPUT_DIR / "mixed_train.parquet",
        split="train",
    )

    # Quick sanity check: load back and compare with original ToolRL
    # (optional; only runs if TOOLRL_PARQUET env var is set).
    print(f"\n{'='*60}")
    _orig_pq = _os.environ.get("TOOLRL_PARQUET")
    if _orig_pq is None or not Path(_orig_pq).exists():
        print("Sanity check skipped (set TOOLRL_PARQUET to enable).")
        raise SystemExit(0)
    print("Sanity check: comparing with original ToolRL parquet...")
    orig = pd.read_parquet(_orig_pq)
    ours = pd.read_parquet(OUTPUT_DIR / "train.parquet")

    print(f"  Original: {len(orig)} rows, columns={list(orig.columns)}")
    print(f"  Ours:     {len(ours)} rows, columns={list(ours.columns)}")

    # Compare prompt structure
    orig_prompt = orig.iloc[0]["prompt"]
    ours_prompt = ours.iloc[0]["prompt"]
    print(f"  Original prompt[0] role: {orig_prompt[0]['role']}, content[:80]: {orig_prompt[0]['content'][:80]}")
    print(f"  Ours prompt[0] role:     {ours_prompt[0]['role']}, content[:80]: {ours_prompt[0]['content'][:80]}")

    # Compare ground_truth structure
    orig_gt = orig.iloc[0]["reward_model"]["ground_truth"]
    ours_gt = ours.iloc[0]["reward_model"]["ground_truth"]
    print(f"  Original GT[:100]: {orig_gt[:100]}")
    print(f"  Ours GT[:100]:     {ours_gt[:100]}")
