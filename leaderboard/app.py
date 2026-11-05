"""AgentRobustBench leaderboard on Hugging Face Spaces.

Run locally:
    pip install -r requirements.txt
    python app.py

Hosted Space: https://huggingface.co/spaces/willchow66/AgentRobustBench

Submission flow (BFCL-style):
    1. User downloads the benchmark data from https://huggingface.co/datasets/robustbench-tc/RobustBench-TC.
    2. User runs their model on their own GPU to produce predictions
       (with the released inference helper).
    3. User uploads predictions JSON to this Space.
    4. The Space scores it server-side with eval_results.py and updates
       the leaderboard.

We deliberately do NOT run inference on HF Space hardware: 32B models
exceed Space ZeroGPU limits, and BFCL/Gorilla and most leaderboards
follow this same submit-predictions pattern.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Monkey-patch gradio_client.utils to fix
#   "TypeError: argument of type 'bool' is not iterable"
# inside get_type when a JSON schema's `additionalProperties` is a bool
# instead of a dict. Affects gradio 4.44.0 + bundled gradio_client 1.3.0,
# triggered during app.api_info schema generation at startup. Must run
# BEFORE `import gradio` so the patched function is the one gradio sees.
# Upstream fix: gradio_client>=1.5.x. Until we upgrade gradio to 5.x,
# this in-process patch is the smallest safe workaround.
# ---------------------------------------------------------------------------
import gradio_client.utils as _gc_utils

_gc_orig_get_type = _gc_utils.get_type

def _gc_get_type_safe(schema):
    if not isinstance(schema, dict):
        return "Any"
    return _gc_orig_get_type(schema)

_gc_utils.get_type = _gc_get_type_safe

_gc_orig_json_to_python = _gc_utils._json_schema_to_python_type

def _gc_json_to_python_safe(schema, defs=None):
    if not isinstance(schema, dict):
        return "Any"
    try:
        return _gc_orig_json_to_python(schema, defs)
    except (TypeError, KeyError):
        return "Any"

_gc_utils._json_schema_to_python_type = _gc_json_to_python_safe

import gradio as gr
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
SCORER = ROOT / "scoring" / "eval_results.py"
LEADERBOARD_PATH = ROOT / "leaderboard.csv"

PERTURBATION_DISPLAY_NAMES = (
    "clean",
    "Typo", "QueryPara", "ToolPara", "ParamPara",
    "Dup-NoDesc", "Dup-Desc", "Dup-WrongP", "Dup-DescWP", "Dup-SwapDP",
    "RedunTool",
    "MisDesc", "TimeDesc", "MisDesc-N", "TimeDesc-N",
    "MisDesc-Abbr", "TimeDesc-Abbr",
    "Timeout", "RateLim", "AuthErr", "5xxErr", "Malform", "SchemaD",
)

DISPLAY_TO_CODE = {
    "clean": "clean",
    "Typo": "realistic_typos", "QueryPara": "query_paraphrase",
    "ToolPara": "paraphrase_tool_description",
    "ParamPara": "paraphrase_parameter_description",
    "Dup-NoDesc": "same_name_A", "Dup-Desc": "same_name_B",
    "Dup-WrongP": "same_name_C", "Dup-DescWP": "same_name_D",
    "Dup-SwapDP": "same_name_E", "RedunTool": "redundant",
    "MisDesc": "CD", "TimeDesc": "TD",
    "MisDesc-N": "CD_NT", "TimeDesc-N": "TD_NT",
    "MisDesc-Abbr": "CD_AB", "TimeDesc-Abbr": "TD_AB",
    "Timeout": "transient_timeout", "RateLim": "transient_rate_limit",
    "AuthErr": "transient_auth_error", "5xxErr": "transient_server_error",
    "Malform": "transient_malformed_response",
    "SchemaD": "transient_schema_drift",
}


# ---------------------------------------------------------------------------
# Submission validation
# ---------------------------------------------------------------------------

def validate(payload: dict[str, Any]) -> tuple[bool, str]:
    for k in ("model_name", "model_type", "contact", "results"):
        if k not in payload:
            return False, f"missing required field: {k!r}"
    if payload["model_type"] not in ("open", "closed-api", "fine-tuned"):
        return False, "model_type must be one of {open, closed-api, fine-tuned}"
    if not isinstance(payload["results"], list) or not payload["results"]:
        return False, "results must be a non-empty list"
    seen = {r.get("perturbation") for r in payload["results"]}
    unknown = seen - set(PERTURBATION_DISPLAY_NAMES)
    if unknown:
        return False, f"unknown perturbation names: {sorted(unknown)}"
    for r in payload["results"][:5]:
        for fld in ("sample_id", "perturbation", "tool_calls"):
            if fld not in r:
                return False, f"each result needs sample_id/perturbation/tool_calls; missing {fld!r}"
    return True, "ok"


# ---------------------------------------------------------------------------
# Scoring (REAL — no longer a stub)
# ---------------------------------------------------------------------------

def score_submission(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert the submission into the format eval_results.py expects, run the
    scorer, return a summary dict ready for the leaderboard CSV.
    """
    by_perturbation: dict[str, list[dict[str, Any]]] = {}
    for r in payload["results"]:
        code = DISPLAY_TO_CODE.get(r["perturbation"], r["perturbation"])
        by_perturbation.setdefault(code, []).append({
            "id": r["sample_id"],
            "benchmark": r.get("benchmark", "unknown"),
            "category": r.get("category"),
            "perturbation": code,
            "prediction": {
                "raw_output": r.get("raw_output", ""),
                "tool_calls": r["tool_calls"],
            },
            # Ground-truth fields are filled by the scorer from the dataset
            "golden_answers": r.get("golden_answers"),
            "tools": r.get("tools"),
            "eval_config": r.get("eval_config"),
        })

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for code, recs in by_perturbation.items():
            f = tmp / f"{code}.predictions.jsonl"
            with f.open("w") as out:
                for rec in recs:
                    out.write(json.dumps(rec) + "\n")

        out_json = tmp / "scores.json"
        cmd = [
            sys.executable, str(SCORER),
            "--predictions", str(tmp),
            "--output", str(out_json),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=600)
        except subprocess.CalledProcessError as e:
            return {"error": f"scorer failed: {e.stderr.decode()[-400:]}"}
        scores = json.loads(out_json.read_text())

    ov = scores["overall"]["overall"]
    by_mdp = scores["overall"]["by_mdp_category"]
    by_pert = scores["overall"].get("by_perturbation", {})

    # Perturbed-only overall: exclude the clean cell so the headline number
    # reflects sim-to-real gap rather than clean-task ability. Sum (correct, n)
    # across every cell whose key isn't 'clean', then ratio.
    pert_correct = sum(
        v.get("correct", round(v["accuracy"] * v["n"]))
        for k, v in by_pert.items() if k != "clean"
    )
    pert_n = sum(v["n"] for k, v in by_pert.items() if k != "clean")
    overall_pert_acc = pert_correct / pert_n if pert_n > 0 else 0.0

    clean_acc = by_mdp.get("clean", {}).get("accuracy", 0.0) if isinstance(by_mdp.get("clean"), dict) else by_mdp.get("clean", 0.0)
    return {
        "overall_acc": overall_pert_acc,             # perturbed-only avg (used as headline)
        "overall_acc_with_clean": ov["accuracy"],    # full average kept for reference
        "n_samples": ov["n"],                        # total predictions across 23 cells
        "n_perturbed": pert_n,                       # predictions excluding clean
        "clean_acc": clean_acc,
        "by_mdp": {k: (v["accuracy"] if isinstance(v, dict) else v) for k, v in by_mdp.items()},
    }


# ---------------------------------------------------------------------------
# Leaderboard CSV (Space-local; we will sync to HF Dataset hub in M4)
# ---------------------------------------------------------------------------

# Column glossary (also rendered in a markdown above the table):
#   Pert. Acc.   = accuracy on perturbed cells ONLY (clean excluded), i.e. the
#                  headline robustness number.
#   Clean        = accuracy on the 199 clean baseline samples; Clean − Pert. Acc.
#                  is the sim-to-real gap.
#   Obs/Action/Reward/Transition Acc. = average accuracy across the perturbations
#                                       within that MDP component.
LEADERBOARD_COLUMNS = [
    "Model", "Type", "Pert. Acc.", "Clean",
    "Obs Acc.", "Action Acc.", "Reward Acc.", "Transition Acc.",
    "Submitted",
]


def append_entry(model_name: str, model_type: str,
                 summary: dict[str, Any]) -> None:
    by_mdp = summary.get("by_mdp", {})
    row = {
        "Model": model_name,
        "Type": model_type,
        "Pert. Acc.": round(summary.get("overall_acc", 0), 4),
        "Clean":           round(summary.get("clean_acc", by_mdp.get("clean", 0)), 4),
        "Obs Acc.":        round(by_mdp.get("observation",0), 4),
        "Action Acc.":     round(by_mdp.get("action",     0), 4),
        "Reward Acc.":     round(by_mdp.get("reward",     0), 4),
        "Transition Acc.": round(by_mdp.get("transition", 0), 4),
        "Submitted": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
    }
    if LEADERBOARD_PATH.exists():
        df = pd.read_csv(LEADERBOARD_PATH)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])[LEADERBOARD_COLUMNS]
    df.to_csv(LEADERBOARD_PATH, index=False)


def load_leaderboard() -> pd.DataFrame:
    if LEADERBOARD_PATH.exists():
        df = pd.read_csv(LEADERBOARD_PATH)
        # Backward-compat: drop deprecated columns and rename legacy headers.
        df = df.rename(columns={"n": "N (eval)", "Overall Acc.": "Pert. Acc."})
        df = df.drop(columns=[c for c in ("N (eval)",) if c in df.columns])
        sort_col = "Pert. Acc." if "Pert. Acc." in df.columns else "Overall Acc."
        if sort_col in df.columns and df[sort_col].notna().any():
            df = df.sort_values(sort_col, ascending=False)
        return df
    return pd.DataFrame(columns=LEADERBOARD_COLUMNS)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def submit(file) -> tuple[str, pd.DataFrame]:
    if file is None:
        return "❌ no file selected", load_leaderboard()
    try:
        payload = json.loads(Path(file.name).read_text())
    except json.JSONDecodeError as e:
        return f"❌ JSON parse error: {e}", load_leaderboard()
    ok, msg = validate(payload)
    if not ok:
        return f"❌ {msg}", load_leaderboard()
    summary = score_submission(payload)
    if "error" in summary:
        return f"❌ {summary['error']}", load_leaderboard()
    append_entry(payload["model_name"], payload["model_type"], summary)
    return (
        f"✅ accepted: {payload['model_name']} — "
        f"perturbed acc {summary['overall_acc']:.4f} ({summary['n_perturbed']} perturbed samples), "
        f"clean acc {summary['clean_acc']:.4f} (199 samples)",
        load_leaderboard(),
    )


with gr.Blocks(title="AgentRobustBench Leaderboard",
               analytics_enabled=False) as demo:
    gr.Markdown("# AgentRobustBench Leaderboard")
    gr.Markdown(
        "A sim-to-real robustness leaderboard for tool-use language agents. "
        "**Run inference on your own hardware**, upload the predictions JSON, "
        "and we score it server-side. See the schema in the *Submit* tab.\n\n"
        "[Dataset](https://huggingface.co/datasets/robustbench-tc/RobustBench-TC) · "
        "[Evaluation code](https://github.com/WillChow66/AgentRobustBench)"
    )
    with gr.Tab("Leaderboard"):
        gr.Markdown(
            "**Column guide.** "
            "**Pert. Acc.** = accuracy on perturbed cells ONLY (clean excluded) — this is the "
            "headline robustness number. "
            "**Clean** = accuracy on the 199 clean samples; the gap "
            "**Clean − Pert. Acc.** quantifies the sim-to-real degradation. "
            "Per-MDP-component columns (Obs / Action / Reward / Transition) average "
            "across the perturbations in that component only. "
            "Each row is scored over 3,721 predictions (199 clean + 22 perturbation cells)."
        )
        leaderboard = gr.Dataframe(
            value=load_leaderboard(), interactive=False, wrap=True,
        )
    with gr.Tab("Submit"):
        gr.Markdown(
            "Upload the predictions JSON produced by the released inference helper. "
            "The Space scores it server-side and appends the row to the leaderboard."
        )
        # Force the upload widget into English; gr.File previously inherited
        # the visitor's browser locale (e.g. zh-CN) for the placeholder text.
        file = gr.File(
            label="predictions.json",
            file_types=[".json"],
            file_count="single",
            type="filepath",
        )
        out_msg = gr.Markdown()
        out_table = gr.Dataframe(
            label="Updated leaderboard after this submission",
            interactive=False,
            value=load_leaderboard(),
        )
        gr.Button("Submit").click(submit, inputs=file,
                                  outputs=[out_msg, out_table])
    with gr.Tab("About"):
        gr.Markdown(
            "**Why server-side scoring only?** AgentRobustBench includes 32B-scale "
            "models which exceed HuggingFace Space free-tier hardware limits. "
            "Following the BFCL leaderboard pattern, contributors run their model "
            "locally and submit a predictions JSON. We re-run only the deterministic "
            "rule-based scorer (no GPT judge) on the Space."
        )


if __name__ == "__main__":
    # Pin the gradio locale to English. Without this gradio detects the
    # visitor's Accept-Language and renders zh-CN strings such as
    # "将文件拖放到此处或点击上传" on the upload widget.
    import os as _os
    _os.environ.setdefault("GRADIO_DEFAULT_LANG", "en")
    _os.environ.setdefault("LANG", "en_US.UTF-8")

    # show_api=False: the bundled gradio_client 0.x crashes generating the
    # API schema when our nested data has dict-valued additionalProperties
    # ("TypeError: argument of type 'bool' is not iterable" in
    # gradio_client/utils.py get_type). We don't expose a programmatic
    # client, so dropping the schema endpoint is harmless.
    # server_name=0.0.0.0 / port 7860: HF Space requires bind to all
    # interfaces; without it gradio falls back to localhost which the
    # Space proxy cannot reach.
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
