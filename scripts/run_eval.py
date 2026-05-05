#!/usr/bin/env python3
"""
RobustBench-TC inference runner.

Reads perturbation JSONL files from the HuggingFace `api_eval` directory
(each sample carries its own benchmark / category / eval_config / golden_answers
in the unified schema), dispatches per-benchmark prompt construction, and
records raw model outputs + parsed tool calls for downstream scoring by
`eval_results.py`.

Two modes:
  - ``--mode single`` (default): one model call per sample. Used for all 16
    data-driven perturbation files (observation / action / reward) plus
    the clean baseline.
  - ``--mode transition``: 2-pass runtime injection applied to clean.jsonl.
    Pass 1 gets the model's first tool call, then we inject a transient
    error message (``--transition-type``, one of timeout, rate_limit,
    auth_error, server_error, malformed_response, schema_drift) and score
    the pass-2 retry output.

Usage:
    # 1. Start vLLM server with the ckpt to be evaluated:
    vllm serve /path/to/ckpt --port 8000

    # 2a. Run inference across all 16 perturbation files + clean baseline:
    python scripts/run_eval.py \
        --model-name my-model \
        --endpoint http://localhost:8000/v1 \
        --data-dir <repo-root>/hf_data/datasets/api_eval \
        --output-dir results/my-model

    # 2b. Run transition perturbation (timeout) on clean baseline:
    python scripts/run_eval.py \
        --model-name my-model \
        --mode transition --transition-type timeout \
        --output-dir results/my-model

    # 2c. Single file:
    python scripts/run_eval.py --model-name my-model \
        --files clean.jsonl realistic_typos.jsonl \
        --output-dir results/my-model

Output: one .predictions.jsonl per input file, where each line has the full
sample plus {"prediction": {"raw_output": "...", "tool_calls": [...]}}.
Transition outputs are placed under ``{output-dir}/transition_{type}/``.
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm


# ────────────────────────────────────────────────────────────────
# BFCL V3 default system prompt (only prepended for BFCL single-turn
# samples; API-Bank / RoTBench / ToolAlpaca / ToolEyes already carry
# their own system prompt inside sample["conversation"][0]).
# ────────────────────────────────────────────────────────────────

BFCL_SYSTEM_PROMPT_TEMPLATE = (
    "You are an expert in composing functions. You are given a question "
    "and a set of possible functions. Based on the question, you will need "
    "to make one or more function/tool calls to achieve the purpose.\n"
    "If none of the functions can be used, point it out. If the given "
    "question lacks the parameters required by the function, also point it out.\n"
    "You should only return the function calls in your response.\n\n"
    "If you decide to invoke any of the function(s), you MUST put it in the "
    "format of [func_name1(params_name1=params_value1, params_name2=params_value2...), "
    "func_name2(params)]\n"
    "You SHOULD NOT include any other text in the response.\n\n"
    "At each turn, you should try your best to complete the tasks requested "
    "by the user within the current turn. Continue to output functions to "
    "call until you have fulfilled the user's request to the best of your ability. "
    "Once you have no more functions to call, the system will consider the "
    "current turn complete and proceed to the next turn or task.\n"
    "\nHere is a list of functions in JSON format that you can invoke.\n"
    "{functions}\n"
)


# ────────────────────────────────────────────────────────────────
# Transition perturbation: 6 transient-error message variants.
# Six transient-error string templates (one per Transition variant). The
# scorer also keys on these, so do not edit without updating the scorer.
# ────────────────────────────────────────────────────────────────

TRANSITION_ERROR_MESSAGES: Dict[str, str] = {
    "timeout":
        "ERROR: Timeout occurred while executing the tool call. "
        "The operation took too long to respond. Please retry your request.",
    "rate_limit":
        "ERROR: HTTP 429 Too Many Requests. Rate limit exceeded for this API. "
        "You have sent too many requests in a given time window. "
        "Please wait a moment and retry your request.",
    "auth_error":
        "ERROR: HTTP 401 Unauthorized. Invalid or expired authentication token. "
        "The tool could not verify your credentials. Please re-authenticate and "
        "retry your request.",
    "server_error":
        "ERROR: HTTP 500 Internal Server Error. The tool's backend service "
        "encountered an unexpected condition and could not fulfill the request. "
        "This is a temporary issue. Please retry your request.",
    "malformed_response":
        "ERROR: Malformed response from tool execution. The response could not "
        "be parsed. Expected valid JSON output but received corrupted data: "
        "'{\\x00\\xef...truncated'. Please retry your request.",
    "schema_drift":
        "ERROR: Schema validation failed. One or more parameters in your request "
        "are no longer recognized by the current API version. The API may have "
        "been updated. Please check the tool documentation and retry with valid "
        "parameters.",
}


# ────────────────────────────────────────────────────────────────
# Sample loading + normalization
# ────────────────────────────────────────────────────────────────

def _maybe_json_loads(value: Any) -> Any:
    """HF api_eval stores conversation/tools/golden_answers as JSON strings."""
    if isinstance(value, str):
        return json.loads(value)
    return value


_NESTED_FIELDS = (
    "conversation", "tools", "golden_answers",
    "eval_config", "perturbation", "source",
)


def load_samples(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file from the HF api_eval directory.

    The HF publication encodes nested structures (conversation, tools,
    golden_answers, eval_config, perturbation, source) as JSON-encoded
    *strings* on every line. Parse them back to real Python objects.
    """
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            for fld in _NESTED_FIELDS:
                if fld in raw:
                    raw[fld] = _maybe_json_loads(raw[fld])
            samples.append(raw)
    return samples


_OPENAI_FC_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")


def _sanitize_tool_name(name: str) -> str:
    """OpenAI Chat Completions FC requires tool names to match ``^[a-zA-Z0-9_-]+$``.

    Many BFCL / API-Bank tools are named e.g. ``country_info.capital`` (a dot)
    or ``**Think`` (asterisks) which the API rejects. Replace any disallowed
    char with an underscore. Caller is responsible for reverse-mapping the
    sanitized name back to the original when parsing tool_calls (see
    ``_unsanitize_tool_call``).
    """
    return _OPENAI_FC_NAME_RE.sub("_", name)


def _build_fc_tools(sample_tools: List[Dict[str, Any]]) -> Tuple[
    List[Dict[str, Any]], Dict[str, str]
]:
    """Return (tools= payload, sanitized->original name map)."""
    name_map: Dict[str, str] = {}
    payload: List[Dict[str, Any]] = []
    for t in sample_tools:
        original = t.get("name", "")
        sanitized = _sanitize_tool_name(original)
        name_map[sanitized] = original
        payload.append(
            {
                "type": "function",
                "function": {
                    "name": sanitized,
                    "description": t.get("description", ""),
                    "parameters": _normalize_tool_schema(t.get("parameters", {})),
                },
            }
        )
    return payload, name_map


def _normalize_tool_schema(schema: Any) -> Dict[str, Any]:
    """Normalize a BFCL/API-Bank parameter schema to strict JSON Schema for OpenAI FC.

    BFCL stores parameters as ``{"type": "dict", "properties": {...}}`` which
    OpenAI's tools= API rejects (only "object" is valid at the root). It also
    stores per-property types as e.g. "dict" / "tuple" — we map those to the
    nearest JSON-Schema equivalent recursively.
    """
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}

    type_remap = {
        "dict": "object",
        "tuple": "array",
        "list": "array",
        "any": "string",
        # Python primitive names that BFCL / API-Bank tool schemas sometimes
        # use instead of the JSON-Schema names. OpenAI FC validates strictly.
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
    }

    def fix(node: Any) -> Any:
        if isinstance(node, dict):
            out: Dict[str, Any] = {}
            for k, v in node.items():
                if k == "type" and isinstance(v, str):
                    out[k] = type_remap.get(v.lower(), v)
                else:
                    out[k] = fix(v)
            return out
        if isinstance(node, list):
            return [fix(x) for x in node]
        return node

    fixed = fix(schema)
    # Root must be type=object for OpenAI FC.
    if fixed.get("type") != "object":
        fixed["type"] = "object"
    if "properties" not in fixed:
        fixed["properties"] = {}
    return fixed


def is_multi_turn(sample: Dict[str, Any]) -> bool:
    cat = sample.get("category") or ""
    return "multi_turn" in cat


# ────────────────────────────────────────────────────────────────
# Per-benchmark message builder
# ────────────────────────────────────────────────────────────────

BFCL_MINIMAL_FC_SYSTEM = (
    "You are a helpful assistant with access to the provided tools. "
    "Use them when appropriate to answer the user's request."
)


def build_messages(
    sample: Dict[str, Any],
    no_system_prompt: bool = False,
    use_fc: bool = False,
) -> List[Dict[str, str]]:
    """Construct OpenAI chat-completions `messages` for this sample.

    Delegates per-benchmark protocol:
      - bfcl_v3 single-turn:
          * prompt mode (use_fc=False): prepend BFCL's default system prompt
            with tools embedded and AST format instructions.
          * FC mode (use_fc=True): use a minimal system prompt; tools are
            passed separately via the OpenAI ``tools=`` API field (by the
            caller).
      - apibank / rotbench / toolalpaca / tooleyes: conversation[0] already
        contains the benchmark's full system prompt (with tool descriptions
        in text form). Return it unchanged in either mode; FC mode simply
        adds the structured ``tools=`` API field alongside.

    ``no_system_prompt=True`` (e.g. for DeepSeek-R1-Distill which explicitly
    requires "all instructions in user message, no system prompt"): merges
    any system content into the first user message.
    """
    conv = sample["conversation"]
    tools = sample["tools"]
    benchmark = sample["benchmark"]

    # Defensive: always return list of {role, content} dicts
    def _coerce(messages):
        cleaned = []
        for m in messages:
            if isinstance(m, dict) and "role" in m:
                cleaned.append({"role": m["role"], "content": m.get("content") or ""})
        return cleaned

    if benchmark == "bfcl_v3":
        if use_fc:
            # FC mode: minimal system; tools go via the API field
            system = BFCL_MINIMAL_FC_SYSTEM
        else:
            # Prompt mode: embed tools in system + BFCL AST format instruction
            funcs_json = json.dumps(tools, indent=2)
            system = BFCL_SYSTEM_PROMPT_TEMPLATE.format(functions=funcs_json)
        msgs = [{"role": "system", "content": system}] + _coerce(conv)
    else:
        # Other benchmarks already embed system prompt + tools in conversation[0]
        msgs = _coerce(conv)

    if no_system_prompt:
        msgs = _fold_system_into_user(msgs)
    return msgs


def _fold_system_into_user(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Merge a leading system message into the first user message.

    Used by models (e.g. DeepSeek-R1-Distill) whose authors recommend
    against system prompts ("all instructions should be contained within
    the user prompt"). The system content is prepended to the user content
    with a separator, preserving instruction visibility but in user role.
    """
    if not messages:
        return messages
    sys_chunks: List[str] = []
    rest: List[Dict[str, str]] = []
    for m in messages:
        if m["role"] == "system":
            if m["content"]:
                sys_chunks.append(m["content"])
        else:
            rest.append(m)
    if not sys_chunks:
        return messages
    # Prepend system content to the first user message
    folded = "\n\n".join(sys_chunks)
    out: List[Dict[str, str]] = []
    inserted = False
    for m in rest:
        if not inserted and m["role"] == "user":
            out.append({"role": "user", "content": folded + "\n\n" + m["content"]})
            inserted = True
        else:
            out.append(m)
    if not inserted:
        # Edge case: no user message at all; create one
        out.insert(0, {"role": "user", "content": folded})
    return out


# ────────────────────────────────────────────────────────────────
# Per-benchmark tool-call parser
# ────────────────────────────────────────────────────────────────

def _parse_bfcl_ast(output: str) -> List[Dict[str, Any]]:
    """Parse BFCL-style function call syntax.

    Accepts both the canonical bracketed form
        ``[func(a=1), other(b=2)]``
    and bare single-call forms
        ``func(a=1)``
        ``Result = CountryAvailableCountries()``
    The latter two turn up in ToolAlpaca outputs where the model doesn't
    follow the BFCL list convention.

    Also accepts the **colon-separator variant** some Qwen-family RL models
    (Qwen2.5-1.5B ToolRL) emit:
        ``[solarFarm.potential(coordinates: [43.6, -79.4], month: "December")]``
    We detect the `:` form and rewrite to `=` before AST parsing.
    """
    import ast
    import re

    if not output:
        return []

    # Try the canonical [ ... ] list first; if not present, try to find
    # bare ``name(args)`` calls line-by-line. Iterate over candidates so
    # we don't bail on the first regex match that happens to span prose
    # (e.g. "needed (like width and height if required)" happens to look
    # like a call but isn't valid Python).
    m = re.search(r"\[[^\[\]]*\([^\[\]]*\)[^\[\]]*\]", output, re.DOTALL)
    candidates: List[str] = []
    if m:
        candidates.append(m.group(0))
    for bare in re.finditer(
        r"((?:[A-Za-z_][\w.]*)\s*\((?:[^()]|\([^()]*\))*\))",
        output,
    ):
        candidates.append("[" + bare.group(1) + "]")

    # Colon-separator variant: `[fn(a: 1, b: "x")]` — rewrite to `=` form
    # so the AST parser can handle it. We only rewrite inside call argument
    # lists (between the first `(` and matching `)`), not inside list/dict
    # literals where `:` is semantically significant.
    colon_candidates: List[str] = []
    for cand in list(candidates):
        if ":" in cand and "=" not in cand:
            rewritten = _rewrite_colon_kwargs(cand)
            if rewritten != cand:
                colon_candidates.append(rewritten)
    candidates.extend(colon_candidates)

    tree = None
    for cand in candidates:
        try:
            tree = ast.parse(cand.strip(), mode="eval")
            break
        except SyntaxError:
            continue
    if tree is None:
        return []

    calls = []
    body = tree.body
    expressions = body.elts if isinstance(body, ast.List) else [body]
    for expr in expressions:
        if not isinstance(expr, ast.Call):
            continue
        # BFCL uses both plain (foo) and dotted (module.submodule.foo) names.
        # ast parses dotted names as chained ast.Attribute nodes — reconstruct.
        name = _extract_dotted_name(expr.func)
        if not name:
            continue
        params: Dict[str, Any] = {}
        for kw in expr.keywords:
            try:
                params[kw.arg] = ast.literal_eval(kw.value)
            except (ValueError, SyntaxError):
                # Fallback: use unparse for non-literal expressions
                try:
                    params[kw.arg] = ast.unparse(kw.value)
                except Exception:
                    pass
        calls.append({"name": name, "parameters": params})
    return calls


def _rewrite_colon_kwargs(cand: str) -> str:
    """Rewrite ``fn(a: 1, b: "x")`` to ``fn(a=1, b="x")``.

    Only rewrites `:` that sits between a call's first `(` and matching `)`,
    and only when it's acting as a keyword separator (`<ident>\\s*:\\s*`).
    Bracket-depth tracking keeps list/dict literals unchanged.
    """
    import re

    # Walk char-by-char so we respect (), [], {} nesting. Track whether we
    # are inside a call-argument list (paren depth > 0, but list/dict depth
    # at their baseline for the current paren scope).
    out: List[str] = []
    # Stack of (kind, list_base, dict_base) triples so nested calls work too.
    stack: List[Tuple[str, int, int]] = []
    list_depth = 0
    dict_depth = 0
    in_str: str = ""  # current string quote char or ""
    i = 0
    n = len(cand)
    while i < n:
        ch = cand[i]
        if in_str:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(cand[i + 1])
                i += 2
                continue
            if ch == in_str:
                in_str = ""
            i += 1
            continue
        if ch in ('"', "'"):
            in_str = ch
            out.append(ch)
            i += 1
            continue
        if ch == "(":
            stack.append(("(", list_depth, dict_depth))
            out.append(ch)
            i += 1
            continue
        if ch == "[":
            list_depth += 1
            out.append(ch)
            i += 1
            continue
        if ch == "]":
            list_depth -= 1
            out.append(ch)
            i += 1
            continue
        if ch == "{":
            dict_depth += 1
            out.append(ch)
            i += 1
            continue
        if ch == "}":
            dict_depth -= 1
            out.append(ch)
            i += 1
            continue
        if ch == ")" and stack:
            stack.pop()
            out.append(ch)
            i += 1
            continue
        if (
            ch == ":"
            and stack
            and stack[-1][0] == "("
            and list_depth == stack[-1][1]
            and dict_depth == stack[-1][2]
        ):
            # Look backward: is this preceded by `<ident>\s*`?
            j = len(out) - 1
            while j >= 0 and out[j] == " ":
                j -= 1
            end = j + 1
            while j >= 0 and (out[j].isalnum() or out[j] == "_"):
                j -= 1
            ident = "".join(out[j + 1:end])
            # j now points at the char before the identifier; verify that
            # what precedes the identifier is `(`, `,`, or start — i.e.
            # this really is a kwarg slot, not a dict literal `key: value`.
            k = j
            while k >= 0 and out[k] == " ":
                k -= 1
            prev_nonspace = out[k] if k >= 0 else ""
            if ident and ident[0].isalpha() and prev_nonspace in ("(", ","):
                out.append("=")
                i += 1
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def _extract_dotted_name(node) -> str:
    """Return the dotted name of an ast.Name / ast.Attribute node.

    ``foo(x)``       -> func is ast.Name("foo")        -> "foo"
    ``a.b.foo(x)``   -> func is ast.Attribute chain    -> "a.b.foo"
    Anything else    -> ""
    """
    import ast
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parts = []
        cur = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
            return ".".join(reversed(parts))
    return ""


def _parse_tool_call_xml(output: str) -> List[Dict[str, Any]]:
    """Parse `<tool_call>{"name": ..., "parameters": ...}</tool_call>` format.

    Used by API-Bank (and many Qwen-style prompt protocols). Multiple tool
    calls may appear back-to-back inside one <tool_call>...</tool_call> block.
    Also handles **unterminated** <tool_call> blocks (model stopped before
    emitting the closing tag, common when max_tokens truncates): we extract
    everything after the final ``<tool_call>`` up to ``</response>``,
    ``</tool_call>``, or end of output.
    """
    import re

    if not output or "<tool_call>" not in output:
        return []
    # Closed blocks first
    blocks = re.findall(r"<tool_call>(.*?)</tool_call>", output, re.DOTALL)
    # Plus any trailing unterminated block
    tail_match = re.search(r"<tool_call>((?:(?!</tool_call>).)*)\Z",
                           output, re.DOTALL)
    if tail_match:
        blocks.append(tail_match.group(1))
    calls = []
    for blk in blocks:
        # Each block may contain one or more JSON objects, separated by newlines.
        text = blk.strip()
        if not text:
            continue
        decoder = json.JSONDecoder()
        pos = 0
        while pos < len(text):
            while pos < len(text) and text[pos] in " \t\r\n":
                pos += 1
            if pos >= len(text):
                break
            try:
                obj, end = decoder.raw_decode(text, pos)
                pos = end
            except json.JSONDecodeError:
                break
            if isinstance(obj, dict) and "name" in obj:
                params = obj.get("parameters", obj.get("arguments", {}))
                calls.append({"name": obj["name"], "parameters": params or {}})
    return calls


_NAME_KEYS = ("name", "function", "tool", "func_name", "tool_name", "action")
_PARAM_KEYS = ("parameters", "arguments", "params", "args", "action_input")


def _parse_json_blob(output: str) -> List[Dict[str, Any]]:
    """Parse JSON tool-call blobs, with or without code fences.

    Handles ToolAlpaca-style outputs where the model emits JSON like
        {"function": "foo", "parameters": {...}}
        {"name": "foo", "arguments": {...}}
        {"func_name": "foo", "params": {...}}       # SFT-Clean variants
    optionally inside ``` ... ``` code fences.
    """
    import re

    if not output:
        return []

    calls: List[Dict[str, Any]] = []

    # Find all {...} JSON candidates (simple brace matching; non-greedy).
    # This catches top-level JSON and JSON inside ```json blocks equally.
    candidates = re.findall(r"\{(?:[^{}]|\{[^{}]*\})*\}", output, re.DOTALL)
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        name = None
        for k in _NAME_KEYS:
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                name = v
                break
        if not name:
            continue
        params: Dict[str, Any] = {}
        for k in _PARAM_KEYS:
            if k in obj:
                v = obj[k]
                if isinstance(v, dict):
                    params = v
                    break
        calls.append({"name": name, "parameters": params})
    return calls


def _parse_bfcl_keyword_list(output: str) -> List[Dict[str, Any]]:
    """Parse ``[func_name="X", params={...}]`` form (SFT-Clean).

    Some SFT models emit the BFCL list shape but with quoted keyword args
    instead of a real call, e.g.
        [func_name="sol_qua_equ", params={"a": 2, "b": 6, "c": 5}]
    which is not valid Python (bare keyword args outside a call). Detect
    this with a regex and pull the name/params out directly.
    """
    import re

    if not output:
        return []
    calls: List[Dict[str, Any]] = []
    # Match [ ... func_name="X" ... params={...} ... ]
    for m in re.finditer(
        r"\[\s*"
        r"(?:name|func_name|function|tool|tool_name)\s*=\s*[\"']([^\"']+)[\"']"
        r"\s*,\s*"
        r"(?:parameters|params|arguments|args)\s*=\s*(\{(?:[^{}]|\{[^{}]*\})*\})"
        r"\s*\]",
        output,
        re.DOTALL,
    ):
        name = m.group(1)
        try:
            params = json.loads(m.group(2))
        except json.JSONDecodeError:
            params = {}
        if not isinstance(params, dict):
            params = {}
        calls.append({"name": name, "parameters": params})
    return calls


def _parse_xml_tool_variants(output: str) -> List[Dict[str, Any]]:
    """Parse miscellaneous `<tool*>` XML variants emitted by Qwen2.5-1.5B etc.

    Accepts many spellings the base `_parse_tool_call_xml` misses, e.g.:
        <tool>RecordHealthData</tool>\n{...}
        <tool>RecordHealthData\n{...}
        <tool name="RecordHealthData">{...}</tool>
        <tool name="X" parameters={...}/>
        <toolcall tool="X">\n{...}
        <tool_call tool_name="X">\n{...}
        <tool_call tool="X">{...}
        <tool Call="X" parameters={...}/>

    Strategy: find every `<tool*>` open tag, extract a candidate name (from
    body-text or any `name=/tool=/tool_name=/Call=` attribute), then find
    the nearest JSON object after it.
    """
    import re

    if not output:
        return []

    calls: List[Dict[str, Any]] = []
    # Walk through all <tool...> open tags. We accept tool, toolcall, tool_call.
    tag_re = re.compile(
        r"<(tool(?:_call|call)?)\b([^>]*)>",
        re.IGNORECASE,
    )
    attr_name_re = re.compile(
        r'(?:name|tool|tool_name|function|func_name|call)\s*=\s*'
        r'[\"\']([^\"\']+)[\"\']',
        re.IGNORECASE,
    )
    attr_params_re = re.compile(
        r'(?:parameters|params|arguments|args|parameters)\s*=\s*'
        r'(\{(?:[^{}]|\{[^{}]*\})*\})',
        re.IGNORECASE,
    )
    json_obj_re = re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\}", re.DOTALL)

    seen_spans: List[Tuple[int, int]] = []
    for m in tag_re.finditer(output):
        tag_body = m.group(2) or ""
        tag_end = m.end()

        # 1) Try to extract name from attribute (e.g. `tool="X"`, `name="X"`)
        name = None
        name_attr = attr_name_re.search(tag_body)
        if name_attr:
            name = name_attr.group(1).strip()

        # 2) Check for parameters attribute (self-closing form)
        params: Optional[Dict[str, Any]] = None
        param_attr = attr_params_re.search(tag_body)
        if param_attr:
            try:
                params = json.loads(param_attr.group(1))
                if not isinstance(params, dict):
                    params = None
            except json.JSONDecodeError:
                # Try a Python-literal fallback (dict with single quotes)
                try:
                    import ast
                    loaded = ast.literal_eval(param_attr.group(1))
                    if isinstance(loaded, dict):
                        params = loaded
                except Exception:
                    params = None

        # 3) Scan forward for body text / JSON object
        # Boundary: next open tag or closing `</tool*>` or end of string.
        close_re = re.compile(r"</tool(?:_call|call)?\s*>", re.IGNORECASE)
        next_tag = tag_re.search(output, tag_end)
        close_match = close_re.search(output, tag_end)
        cutoff = len(output)
        if next_tag:
            cutoff = min(cutoff, next_tag.start())
        if close_match:
            cutoff = min(cutoff, close_match.end())
        scope = output[tag_end:cutoff]

        # If name not from attribute, try body text (first line after tag)
        if not name:
            # Strip any closing </tool*> tags
            scope_for_name = re.sub(r"</tool(?:_call|call)?\s*>", "\n", scope,
                                    flags=re.IGNORECASE)
            # First non-empty line up to a newline or "{"
            line_match = re.match(r"\s*([A-Za-z_][\w]*)\b", scope_for_name)
            if line_match:
                name = line_match.group(1)

        if not name:
            continue

        # If params not from attribute, find first JSON object in scope
        if params is None:
            j = json_obj_re.search(scope)
            if j:
                try:
                    loaded = json.loads(j.group(0))
                    if isinstance(loaded, dict):
                        params = loaded
                except json.JSONDecodeError:
                    params = None

        if params is None:
            params = {}
        # De-duplicate: if this (name, params) pair was already captured
        # via _parse_tool_call_xml on the same span, skip.
        calls.append({"name": name, "parameters": params})
        seen_spans.append((m.start(), cutoff))

    return calls


def _parse_toolalpaca_named_blob(output: str) -> List[Dict[str, Any]]:
    """Parse ToolAlpaca ad-hoc formats that name the function outside the JSON.

    Handles three variants that show up in Qwen2.5/3.5 outputs:
        findCatWithText: {"text": "Happy Birthday!"}                  # name: {json}
        Function: AirportApi_getAirport\\nParameters: {"icao_code": "EDDF"}
        `result_get?appid=X&i=...&units=metric`                         # inline query-string
    Returns at most one call (the first match).
    """
    import re

    if not output:
        return []

    calls: List[Dict[str, Any]] = []

    # Variant 1: "func_name: {json}" — the identifier is followed by `:`
    # then (optionally on a new line) a JSON object.
    for m in re.finditer(
        r"(?:^|\n|```[a-zA-Z]*\n)\s*([A-Za-z_][\w]{2,})\s*:\s*(\{(?:[^{}]|\{[^{}]*\})*\})",
        output,
        re.DOTALL,
    ):
        name = m.group(1)
        # Skip common non-function identifiers that precede a JSON-ish blob.
        if name.lower() in {
            "thought", "action", "observation", "result", "response",
            "json", "parameters", "params", "arguments", "args",
            "output", "input", "note", "example", "format", "final",
            "think", "user", "system", "assistant", "tool",
        }:
            continue
        try:
            params = json.loads(m.group(2))
        except json.JSONDecodeError:
            try:
                import ast
                parsed = ast.literal_eval(m.group(2))
                if isinstance(parsed, dict):
                    params = parsed
                else:
                    continue
            except Exception:
                continue
        if not isinstance(params, dict):
            continue
        calls.append({"name": name, "parameters": params})

    if calls:
        return calls

    # Variant 2: "Function: X\nParameters: {json}"
    fn_match = re.search(
        r"Function\s*:\s*([A-Za-z_][\w.]*)\s*(?:\n|\r\n?)\s*"
        r"Parameters?\s*:\s*(\{(?:[^{}]|\{[^{}]*\})*\})",
        output,
        re.DOTALL | re.IGNORECASE,
    )
    if fn_match:
        try:
            params = json.loads(fn_match.group(2))
            if isinstance(params, dict):
                calls.append({
                    "name": fn_match.group(1),
                    "parameters": params,
                })
                return calls
        except json.JSONDecodeError:
            pass

    # Variant 3: inline query-string form — `fn_name?p1=v1&p2=v2`
    qs_match = re.search(
        r"`([A-Za-z_][\w]*)\?([^`]+)`",
        output,
    )
    if qs_match:
        name = qs_match.group(1)
        qs = qs_match.group(2)
        params: Dict[str, Any] = {}
        for pair in qs.split("&"):
            if "=" in pair:
                k, v = pair.split("=", 1)
                # Drop URL-encoded + signs etc., don't try to be clever.
                params[k.strip()] = v.strip()
        calls.append({"name": name, "parameters": params})
        return calls

    return calls


def _parse_react_action(output: str) -> List[Dict[str, Any]]:
    """Parse ReAct `Action: name\\nAction Input: {...}` format.

    Used by RoTBench and ToolEyes. Only captures the first Action/Input pair
    per step; callers evaluate the next-step action.

    Also accepts the ``ActionCode:``/``Action Code:`` alias some SFT models
    (SFT-Robust-Qwen2.5-3B) emit instead of ``Action:``.
    """
    import re

    if not output:
        return []
    # Non-greedy: match Action up to newline, then Action Input up to next
    # blank line / Observation / Thought / end.
    action_match = re.search(
        r"Action(?:\s*Code)?\s*:\s*([^\n]+)", output, re.IGNORECASE
    )
    if not action_match:
        return []
    name = action_match.group(1).strip().strip("`'\"")
    # Guard against cases where the "Action:" line doesn't have a real
    # identifier after it (e.g. "Action: **Think**" meta-text).
    name = re.sub(r"[*`_\s]+$", "", name)
    if not re.match(r"^[A-Za-z_][\w]*$", name):
        # Try extracting just a leading identifier from the tail
        ident = re.match(r"^([A-Za-z_][\w]*)", name)
        if not ident:
            return []
        name = ident.group(1)

    params: Dict[str, Any] = {}
    input_match = re.search(
        r"Action(?:\s*Code)?\s*Input\s*:\s*(\{.*?\})"
        r"(?:\s*(?:Observation|Thought|Action|$))",
        output,
        re.DOTALL | re.IGNORECASE,
    )
    if not input_match:
        # Greedy fallback: from "Action Input:" / "ActionCode Input:" to end
        input_match = re.search(
            r"Action(?:\s*Code)?\s*Input\s*:\s*(\{.*\})",
            output,
            re.DOTALL | re.IGNORECASE,
        )
    if input_match:
        raw_input = input_match.group(1).strip()
        try:
            params = json.loads(raw_input)
        except json.JSONDecodeError:
            # Try Python-literal fallback for single-quoted dicts
            import ast
            try:
                parsed = ast.literal_eval(raw_input)
                if isinstance(parsed, dict):
                    params = parsed
            except Exception:
                params = {}
    return [{"name": name, "parameters": params}]


_THINK_TAG_RE = __import__("re").compile(r"<think>.*?</think>", __import__("re").DOTALL)


def _strip_thinking(output: str) -> str:
    """Strip reasoning-model preambles so downstream parsers see clean output.

    Handles:
      - ``<think>...</think>``: balanced tags stripped (DeepSeek-R1-Distill
        normal mode, MUA-RL in BFCL context, Qwen3 thinking).
      - **Bare ``</think>``** (no open tag): DeepSeek-R1-Distill default
        behavior — reasoning ends with ``</think>``. Strip up to and
        including the close tag.
      - **Unterminated ``<think>``**: the model (e.g. MUA-RL on apibank)
        sometimes forgets to close the tag but still emits a valid
        ``<tool_call>...</tool_call>`` *inside* the think block. Strip
        only the open tag itself and let downstream parsers extract any
        tool-call markers from the remaining content.
    """
    if not output:
        return output
    if "<think>" not in output and "</think>" not in output:
        return output
    cleaned = _THINK_TAG_RE.sub("", output)
    if "<think>" in cleaned:
        # Unterminated: remove just the tag text so downstream parsers
        # (XML tool_call, AST, ReAct) can still find any tool-call markers
        # that appeared inside the think block.
        cleaned = cleaned.replace("<think>", "")
    if "</think>" in cleaned:
        # Bare close tag: treat everything before it as reasoning preamble.
        cleaned = cleaned.split("</think>", 1)[1]
    return cleaned.strip()


def parse_tool_calls(output: str, benchmark: str) -> List[Dict[str, Any]]:
    """Dispatch output parsing by benchmark protocol."""
    if not output:
        return []

    # Reasoning models (DeepSeek-R1-Distill, MUA-RL, etc.) emit a <think>
    # block before the actual tool call. Strip it so AST / XML / ReAct
    # parsers see the final answer only.
    output = _strip_thinking(output)

    if benchmark == "bfcl_v3":
        calls = _parse_bfcl_ast(output)
        if calls:
            return calls
        # Fallback 1: <tool_call>...</tool_call> format
        calls = _parse_tool_call_xml(output)
        if calls:
            return calls
        # Fallback 2: SFT-Clean "[func_name=..., params=...]" form
        calls = _parse_bfcl_keyword_list(output)
        if calls:
            return calls
        # Fallback 3: JSON blob format — some SFT models emit
        # [{"name": "...", "parameters": {...}}] directly (list of JSON
        # objects inside brackets). This is a valid tool-call structure
        # even though it doesn't match BFCL's native "[func(a=1)]" AST form.
        return _parse_json_blob(output)

    if benchmark == "apibank":
        calls = _parse_tool_call_xml(output)
        if calls:
            return calls
        # Qwen2.5-1.5B etc. emit many miscellaneous <tool*> variants
        calls = _parse_xml_tool_variants(output)
        if calls:
            return calls
        # SFT models sometimes emit JSON without <tool_call> wrapper
        return _parse_json_blob(output)

    if benchmark in ("rotbench", "tooleyes"):
        calls = _parse_react_action(output)
        if calls:
            return calls
        # Some models (MUA-RL etc.) emit <tool_call> even in ReAct contexts
        calls = _parse_tool_call_xml(output)
        if calls:
            return calls
        # TL-CodeLLaMA sometimes emits bare Python calls on RoTBench
        calls = _parse_bfcl_ast(output)
        if calls:
            return calls
        # Last resort: JSON object with a recognisable name key
        return _parse_json_blob(output)

    if benchmark == "toolalpaca":
        # ToolAlpaca models emit highly varied formats: ReAct,
        # <tool_call> XML, bare Python calls, or JSON blobs in ```code```
        # fences. Try each until one lands.
        for fn in (_parse_react_action, _parse_tool_call_xml,
                   _parse_toolalpaca_named_blob,
                   _parse_json_blob, _parse_bfcl_ast):
            calls = fn(output)
            if calls:
                return calls
        return []

    # Unknown benchmark: try all
    for fn in (_parse_bfcl_ast, _parse_tool_call_xml,
               _parse_react_action, _parse_json_blob):
        calls = fn(output)
        if calls:
            return calls
    return []


# ────────────────────────────────────────────────────────────────
# Inference
# ────────────────────────────────────────────────────────────────

def run_single_pass(
    sample: Dict[str, Any],
    client: OpenAI,
    model_name: str,
    temperature: float,
    max_tokens: int,
    use_fc: bool,
    no_system_prompt: bool = False,
    enable_thinking: Optional[bool] = None,
) -> Dict[str, Any]:
    """Run one round of inference for a sample. Returns prediction dict.

    Kept deliberately small and functional so that a future transition-mode
    wrapper can call it twice and inject a timeout message between calls.

    ``no_system_prompt`` (e.g. for DeepSeek-R1-Distill): merge any system
    message into the first user message before sending.

    ``enable_thinking`` (Qwen3 family, MUA-RL, LoopTool): pass through as
    ``extra_body.chat_template_kwargs.enable_thinking`` so vLLM applies the
    template in non-thinking mode (models trained/evaluated under that
    setting per their papers — MUA-RL uses non-thinking, Qwen3 recommended
    default for tool calling is non-thinking).
    """
    messages = build_messages(sample, no_system_prompt=no_system_prompt, use_fc=use_fc)

    # OpenAI o-series reasoning models (o1, o3, o4-mini, etc.) reject the
    # legacy `max_tokens` and `temperature` parameters — they require
    # `max_completion_tokens` and only accept the default temperature.
    is_openai_reasoning = (
        model_name.lower().startswith(("o1", "o3", "o4", "gpt-5"))
        or "/o4-mini" in model_name.lower()
        or "/o1" in model_name.lower()
        or "/o3" in model_name.lower()
    )

    kwargs: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
    }
    if is_openai_reasoning:
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["temperature"] = temperature
        kwargs["max_tokens"] = max_tokens

    if enable_thinking is not None:
        kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": enable_thinking}
        }

    fc_name_map: Dict[str, str] = {}
    if use_fc and sample["tools"]:
        kwargs["tools"], fc_name_map = _build_fc_tools(sample["tools"])

    try:
        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        raw_output = choice.message.content or ""

        tool_calls: List[Dict[str, Any]] = []
        if use_fc and choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    params = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    params = {}
                # Reverse-map sanitized name back to the original tool name so
                # downstream scoring can match against ground-truth (which uses
                # the original BFCL/API-Bank name with dots).
                returned_name = tc.function.name
                original_name = fc_name_map.get(returned_name, returned_name)
                tool_calls.append({"name": original_name, "parameters": params})

        if not tool_calls:
            tool_calls = parse_tool_calls(raw_output, sample["benchmark"])

        return {
            "raw_output": raw_output,
            "tool_calls": tool_calls,
            "error": None,
        }
    except Exception as e:
        return {
            "raw_output": "",
            "tool_calls": [],
            "error": f"{type(e).__name__}: {e}",
        }


# ────────────────────────────────────────────────────────────────
# Transition perturbation: 2-pass inference with runtime error injection
# ────────────────────────────────────────────────────────────────

def _format_tool_call_for_injection(tool_calls: List[Dict[str, Any]]) -> str:
    """Get a display name for the tool call that triggered the failure."""
    if not tool_calls:
        return "unknown_tool"
    first = tool_calls[0]
    if isinstance(first, dict):
        return str(first.get("name") or first.get("tool") or "unknown_tool")
    return str(first)


def run_transition_pass(
    sample: Dict[str, Any],
    client: OpenAI,
    model_name: str,
    temperature: float,
    max_tokens: int,
    use_fc: bool,
    error_message: str,
    no_system_prompt: bool = False,
    enable_thinking: Optional[bool] = None,
) -> Dict[str, Any]:
    """2-pass transient-failure evaluation.

    Pass 1: generate model's initial tool call.
    If the model produced at least one tool call, inject a runtime error
    message as a user-role turn (``Tool execution result for <name>: <error>``)
    and run pass 2. The pass-2 output is what the evaluator scores.

    If the model emitted no tool calls in pass 1, there is no call to
    interrupt; we return the pass-1 output unchanged and flag
    ``transition_injected=False`` so the evaluator can segregate those cases.
    """
    pass1 = run_single_pass(
        sample, client, model_name, temperature, max_tokens, use_fc,
        no_system_prompt=no_system_prompt, enable_thinking=enable_thinking,
    )

    if pass1.get("error") or not pass1.get("tool_calls"):
        return {
            "raw_output": pass1.get("raw_output", ""),
            "tool_calls": pass1.get("tool_calls") or [],
            "error": pass1.get("error"),
            "transition_injected": False,
            "pass1_raw_output": pass1.get("raw_output", ""),
            "pass1_tool_calls": pass1.get("tool_calls") or [],
        }

    tool_name = _format_tool_call_for_injection(pass1["tool_calls"])
    injection_text = f"Tool execution result for {tool_name}: {error_message}"

    messages = build_messages(sample, no_system_prompt=no_system_prompt, use_fc=use_fc)
    messages.append({"role": "assistant", "content": pass1["raw_output"]})
    messages.append({"role": "user", "content": injection_text})

    is_openai_reasoning = (
        model_name.lower().startswith(("o1", "o3", "o4", "gpt-5"))
        or "/o4-mini" in model_name.lower()
        or "/o1" in model_name.lower()
        or "/o3" in model_name.lower()
    )
    kwargs: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
    }
    if is_openai_reasoning:
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["temperature"] = temperature
        kwargs["max_tokens"] = max_tokens
    if enable_thinking is not None:
        kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": enable_thinking}
        }
    fc_name_map: Dict[str, str] = {}
    if use_fc and sample["tools"]:
        kwargs["tools"], fc_name_map = _build_fc_tools(sample["tools"])

    try:
        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        raw_output = choice.message.content or ""
        tool_calls: List[Dict[str, Any]] = []
        if use_fc and choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    params = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    params = {}
                # Reverse-map sanitized name back to the original tool name so
                # downstream scoring can match against ground-truth (which uses
                # the original BFCL/API-Bank name with dots).
                returned_name = tc.function.name
                original_name = fc_name_map.get(returned_name, returned_name)
                tool_calls.append({"name": original_name, "parameters": params})
        if not tool_calls:
            tool_calls = parse_tool_calls(raw_output, sample["benchmark"])

        return {
            "raw_output": raw_output,
            "tool_calls": tool_calls,
            "error": None,
            "transition_injected": True,
            "pass1_raw_output": pass1["raw_output"],
            "pass1_tool_calls": pass1["tool_calls"],
        }
    except Exception as e:
        return {
            "raw_output": "",
            "tool_calls": [],
            "error": f"{type(e).__name__}: {e}",
            "transition_injected": True,
            "pass1_raw_output": pass1["raw_output"],
            "pass1_tool_calls": pass1["tool_calls"],
        }


# ────────────────────────────────────────────────────────────────
# File-level runner
# ────────────────────────────────────────────────────────────────

def run_file(
    input_path: Path,
    output_path: Path,
    client: OpenAI,
    model_name: str,
    temperature: float,
    max_tokens: int,
    use_fc: bool,
    max_workers: int,
    skip_multi_turn: bool,
    resume: bool,
    mode: str = "single",
    transition_error_message: Optional[str] = None,
    limit: Optional[int] = None,
    no_system_prompt: bool = False,
    enable_thinking: Optional[bool] = None,
) -> Tuple[int, int, int]:
    """Run inference for all samples in one jsonl file.

    Args:
        mode: "single" → one model call per sample.
              "transition" → 2-pass: inject ``transition_error_message`` after
                  the first tool call, re-query, and record pass-2 output.
        transition_error_message: required when mode=="transition".

    Returns (n_processed, n_skipped_multi_turn, n_errors).
    """
    samples = load_samples(input_path)

    # Determine already-completed IDs if resuming
    done_ids = set()
    if resume and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_ids.add(rec.get("id"))
                except json.JSONDecodeError:
                    pass

    # Filter samples to run
    pending: List[Tuple[int, Dict[str, Any]]] = []
    n_skipped_mt = 0
    for idx, s in enumerate(samples):
        if skip_multi_turn and is_multi_turn(s):
            n_skipped_mt += 1
            continue
        if s.get("id") in done_ids:
            continue
        pending.append((idx, s))
        if limit is not None and len(pending) >= limit:
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    open_mode = "a" if resume and output_path.exists() else "w"

    n_errors = 0
    n_processed = 0

    # Extract the transition type from the env-message dict at entry (if any),
    # so it can tag the per-sample output records.
    transition_type_tag: Optional[str] = None
    if mode == "transition" and transition_error_message is not None:
        for k, v in TRANSITION_ERROR_MESSAGES.items():
            if v == transition_error_message:
                transition_type_tag = k
                break

    def _work(item):
        idx, s = item
        if mode == "transition":
            pred = run_transition_pass(
                s, client, model_name, temperature, max_tokens, use_fc,
                error_message=transition_error_message,
                no_system_prompt=no_system_prompt,
                enable_thinking=enable_thinking,
            )
        else:
            pred = run_single_pass(
                s, client, model_name, temperature, max_tokens, use_fc,
                no_system_prompt=no_system_prompt,
                enable_thinking=enable_thinking,
            )
        return idx, s, pred

    with open(output_path, open_mode, encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_work, item) for item in pending]
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=input_path.name,
                unit="sample",
                leave=False,
            ):
                idx, s, pred = fut.result()
                if pred.get("error"):
                    n_errors += 1

                # In transition mode the input file is clean.jsonl, so the
                # sample's perturbation field is null. Re-label it so the
                # evaluator's MDP aggregation attributes these predictions
                # to the 'transition' category.
                if mode == "transition":
                    perturbation = {
                        "type": f"transient_{transition_type_tag or 'timeout'}",
                        "mdp_category": "transition",
                        "variant": None,
                        "rate": None,
                        "metadata": {"pass1_had_tool_calls":
                                     pred.get("transition_injected", False)},
                    }
                else:
                    perturbation = s.get("perturbation")

                out_record = {
                    "id": s["id"],
                    "benchmark": s["benchmark"],
                    "category": s.get("category"),
                    "perturbation": perturbation,
                    "eval_config": s.get("eval_config"),
                    "golden_answers": s["golden_answers"],
                    "prediction": pred,
                }
                fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                fout.flush()
                n_processed += 1

    return n_processed, n_skipped_mt, n_errors


# ────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RobustBench-TC inference runner.")
    parser.add_argument("--model-name", required=True,
                        help="Model identifier as registered on the vLLM server.")
    parser.add_argument("--endpoint", default="http://localhost:8000/v1",
                        help="vLLM OpenAI-compatible endpoint.")
    parser.add_argument("--api-key", default="EMPTY",
                        help="API key placeholder for the OpenAI SDK; local "
                             "vLLM does not verify it. No real key needed.")
    parser.add_argument(
        "--data-dir",
        default="<repo-root>/hf_data/datasets/api_eval",
        help="Directory containing the 17 HF api_eval .jsonl files.",
    )
    parser.add_argument("--files", nargs="*", default=None,
                        help="Subset of filenames to run (e.g. clean.jsonl). "
                             "Default: all *.jsonl in --data-dir.")
    parser.add_argument("--output-dir", required=True,
                        help="Where to write {filename}.predictions.jsonl.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N samples per file (for quick "
                             "sanity checks / smoke tests). Default: unlimited.")
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-workers", type=int, default=8,
                        help="Parallel inflight requests to vLLM.")
    parser.add_argument("--use-fc", action="store_true",
                        help="Use OpenAI native tools=[...] mode. Default off: "
                             "we rely on each benchmark's prompt protocol so the "
                             "parser/evaluator pipeline stays consistent.")
    parser.add_argument("--skip-multi-turn", action="store_true", default=True,
                        help="Skip BFCL multi-turn samples (default on).")
    parser.add_argument("--include-multi-turn", dest="skip_multi_turn",
                        action="store_false",
                        help="Disable the multi-turn skip (NOT supported by "
                             "single-pass runner — will produce bad scores).")
    parser.add_argument("--resume", action="store_true",
                        help="Skip samples whose id is already in the output file.")
    parser.add_argument("--no-system-prompt", action="store_true",
                        help="Fold system messages into the first user message. "
                             "Required by some models (e.g. DeepSeek-R1-Distill) "
                             "whose authors recommend against system prompts.")
    # Mutually-exclusive thinking-mode flags. Default (both off) leaves the
    # template kwarg unset, so models use their own default.
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--enable-thinking", dest="enable_thinking",
        action="store_const", const=True, default=None,
        help="Pass chat_template_kwargs.enable_thinking=True "
             "(Qwen3 thinking mode; rarely used for tool calling).",
    )
    thinking_group.add_argument(
        "--disable-thinking", dest="enable_thinking",
        action="store_const", const=False,
        help="Pass chat_template_kwargs.enable_thinking=False. "
             "REQUIRED for Qwen3 family (MUA-RL-8B/14B/32B, LoopTool-8B/32B, "
             "Qwen3-*-Instruct) to match the non-thinking eval setting used "
             "in their papers.",
    )
    parser.add_argument(
        "--mode", choices=("single", "transition"), default="single",
        help="'single' (default): one model call per sample, used for all 16 "
             "data-driven perturbations (observation / action / reward) plus "
             "clean. 'transition': 2-pass runtime injection — applies to the "
             "clean.jsonl baseline only; inject a transient-error after the "
             "first tool call, then score pass-2 output.",
    )
    parser.add_argument(
        "--transition-type",
        choices=tuple(TRANSITION_ERROR_MESSAGES.keys()), default="timeout",
        help="Which transient-error message to inject when --mode transition. "
             "Default 'timeout' matches the published paper; other 5 types "
             "are ablations.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Default file selection differs by mode:
    #  - single mode: run everything in data_dir
    #  - transition mode: run on clean.jsonl only (transition is a runtime
    #    injection applied to the baseline data)
    if args.files:
        files = [data_dir / f for f in args.files]
    elif args.mode == "transition":
        files = [data_dir / "clean.jsonl"]
    else:
        files = sorted(data_dir.glob("*.jsonl"))

    missing = [f for f in files if not f.exists()]
    if missing:
        print(f"ERROR: missing files: {missing}", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=args.endpoint, api_key=args.api_key)

    # Sanity-ping the server
    try:
        _ = client.models.list()
    except Exception as e:
        print(f"ERROR: cannot reach vLLM endpoint {args.endpoint}: {e}",
              file=sys.stderr)
        sys.exit(1)

    transition_error_message = None
    if args.mode == "transition":
        transition_error_message = TRANSITION_ERROR_MESSAGES[args.transition_type]

    summary: List[Dict[str, Any]] = []
    t0 = time.time()
    for f in files:
        if args.mode == "transition":
            # Segregate transition outputs by error type, e.g.
            # results/model/transition_timeout/clean.predictions.jsonl
            out = output_dir / f"transition_{args.transition_type}" / f"{f.stem}.predictions.jsonl"
        else:
            out = output_dir / f"{f.stem}.predictions.jsonl"
        tqdm.write(f"[run] {f.name} -> {out}  (mode={args.mode})")
        n_proc, n_skip_mt, n_err = run_file(
            f, out, client,
            model_name=args.model_name,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            use_fc=args.use_fc,
            max_workers=args.max_workers,
            skip_multi_turn=args.skip_multi_turn,
            resume=args.resume,
            mode=args.mode,
            transition_error_message=transition_error_message,
            limit=args.limit,
            no_system_prompt=args.no_system_prompt,
            enable_thinking=args.enable_thinking,
        )
        summary.append({
            "file": f.name,
            "processed": n_proc,
            "skipped_multi_turn": n_skip_mt,
            "errors": n_err,
        })

    elapsed = time.time() - t0
    tqdm.write("\n=== run_eval.py summary ===")
    tqdm.write(f"Model: {args.model_name}")
    tqdm.write(f"Endpoint: {args.endpoint}")
    tqdm.write(f"Elapsed: {elapsed:.1f}s")
    for row in summary:
        tqdm.write(
            f"  {row['file']:45s} "
            f"processed={row['processed']:4d} "
            f"skipped_mt={row['skipped_multi_turn']:3d} "
            f"errors={row['errors']:3d}"
        )

    # Persist summary
    (output_dir / "_run_summary.json").write_text(
        json.dumps({"model": args.model_name, "files": summary,
                    "elapsed_sec": elapsed}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
