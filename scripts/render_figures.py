"""Render Figures 2, 3 and Table 5 (main results) for the v2 paper.

Reads <repo-root>/results/robustbench/results/<model>/scores.json
and per_sample_scores.jsonl, computes per-MDP-component drop with
bootstrap 95% CI, and emits:

  - paper/neurips_2026_v2/figures/heatmap.pdf       (Fig 2)
  - paper/neurips_2026_v2/figures/scaling.pdf       (Fig 3)
  - paper/neurips_2026_v2/tables/main_results.tex   (Table 5)

No GPU, no API. Pure post-processing of existing eval data.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Use TrueType (Type 42) so Overleaf pdflatex reads the embedded fonts.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

REPO = Path(__file__).resolve().parents[1]
RESULTS = Path("<repo-root>/results/robustbench/results")
FIGS = REPO / "paper" / "neurips_2026_v2" / "figures"
TABS = REPO / "paper" / "neurips_2026_v2" / "tables"
FIGS.mkdir(parents=True, exist_ok=True)
TABS.mkdir(parents=True, exist_ok=True)

# Display order + paper-display name + family + size + result-dir basename
MODELS = [
    # name,                 family,        size_b,  dirname
    ("ToolRL-Qwen2.5-1.5B",  "ToolRL",      1.5,    "chengq9__ToolRL-Qwen2.5-1.5B"),
    ("ToolRL-Qwen2.5-3B",    "ToolRL",      3.0,    "chengq9__ToolRL-Qwen2.5-3B"),
    ("ToolRL-Llama3.2-3B",   "ToolRL",      3.0,    "chengq9__ToolRL-Llama3.2-3B"),
    ("TL-CodeLLaMA-2",       "TL",          7.0,    "Junjie-Ye__TL-CodeLLaMA-2"),
    ("LoopTool-8B",          "LoopTool",    8.0,    "zhuiguang-ning__LoopTool-8B"),
    ("LoopTool-32B",         "LoopTool",    32.0,   "zhuiguang-ning__LoopTool-32B"),
    ("MUA-RL-8B",            "MUA-RL",      8.0,    "zzwkk__MUA-RL-8B"),
    ("MUA-RL-14B",           "MUA-RL",      14.0,   "zzwkk__MUA-RL-14B"),
    ("MUA-RL-32B",           "MUA-RL",      32.0,   "zzwkk__MUA-RL-32B"),
    ("Qwen2.5-1.5B-Instruct","Qwen2.5",     1.5,    "Qwen__Qwen2.5-1.5B-Instruct"),
    ("Qwen2.5-3B-Instruct",  "Qwen2.5",     3.0,    "Qwen__Qwen2.5-3B-Instruct"),
    ("Llama-3.2-3B-Instruct","Llama3.2",    3.0,    "meta-llama__Llama-3.2-3B-Instruct"),
    ("Qwen3-8B",             "Qwen3",       8.0,    "Qwen__Qwen3-8B"),
    ("Qwen3-14B",            "Qwen3",       14.0,   "Qwen__Qwen3-14B"),
    ("Qwen3-32B",            "Qwen3",       32.0,   "Qwen__Qwen3-32B"),
    ("Qwen3.5-9B",           "Qwen3.5",     9.0,    "Qwen__Qwen3.5-9B"),
    ("DeepSeek-R1-Distill-14B","DeepSeek-R1", 14.0, "deepseek-ai__DeepSeek-R1-Distill-Qwen-14B"),
    ("SFT-Clean-4k-3B",       "sft-baseline", 3.0,
        "__projects__bfdz__anon-user__robustbench__sft_models__SFT-Clean-4k-Qwen2.5-3B-Instruct"),
    ("ToolRL-DR-Full (ours)", "ours-DR",     3.0,
        "<anonymized>__qwen2.5-3b-toolrl-robustbench-perturbed"),
    ("ToolRL-DR-Mixed (ours)","ours-DR",     3.0,
        "<anonymized>__qwen2.5-3b-toolrl-robustbench-mixed"),
    # Closed-source frontier — included only when the full eval has been
    # run (smoke-test predictions are filtered out by the n-threshold check
    # below). API parameters: tools= (FC mode), max_completion_tokens=8192,
    # default temperature (o-series rejects custom temp).
    ("o4-mini (OpenAI)",      "closed",      0.0,    "openai__o4-mini"),
]

# Display names in left-to-right column order, code-level identifiers
PERTURBATIONS = [
    ("Typo",            "realistic_typos",                 "Obs"),
    ("QueryPara",       "query_paraphrase",                "Obs"),
    ("ToolPara",        "paraphrase_tool_description",     "Obs"),
    ("ParamPara",       "paraphrase_parameter_description","Obs"),
    ("Dup-NoDesc",      "same_name_A",                     "Act"),
    ("Dup-Desc",        "same_name_B",                     "Act"),
    ("Dup-WrongP",      "same_name_C",                     "Act"),
    ("Dup-DescWP",      "same_name_D",                     "Act"),
    ("Dup-SwapDP",      "same_name_E",                     "Act"),
    ("RedunTool",       "redundant",                       "Act"),
    ("MisDesc",         "CD",                              "Rew"),
    ("TimeDesc",        "TD",                              "Rew"),
    ("MisDesc-N",       "CD_NT",                           "Rew"),
    ("TimeDesc-N",      "TD_NT",                           "Rew"),
    ("MisDesc-Abbr",    "CD_AB",                           "Rew"),
    ("TimeDesc-Abbr",   "TD_AB",                           "Rew"),
    ("Timeout",         "transient_timeout",               "Trn"),
    ("RateLim",         "transient_rate_limit",            "Trn"),
    ("AuthErr",         "transient_auth_error",            "Trn"),
    ("5xxErr",          "transient_server_error",          "Trn"),
    ("Malform",         "transient_malformed_response",    "Trn"),
    ("SchemaD",         "transient_schema_drift",          "Trn"),
]

MDP_COLORS = {
    "Obs": "#7F77DD",   # violet
    "Act": "#D85A30",   # coral
    "Rew": "#1D9E75",   # teal
    "Trn": "#B5872E",   # amber
}

FAMILY_MARKERS = {
    "ToolRL": "o",   "TL": "s",       "LoopTool": "D",   "MUA-RL": "^",
    "Qwen2.5": "v",  "Llama3.2": "P", "Qwen3": "X",      "Qwen3.5": "*",
    "DeepSeek-R1": "h", "ours-SFT": "P",
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_per_sample(dirname: str) -> pd.DataFrame | None:
    p = RESULTS / dirname / "per_sample_scores.jsonl"
    if not p.exists():
        return None
    rows = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append({
                "id": r.get("id"),
                "perturbation": (r.get("perturbation") or {}).get("type"),
                "score": (r.get("score") or {}).get("score", 0),
            })
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, B: int = 10000, alpha: float = 0.05,
                 rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    """Return (mean, lo, hi) bootstrap CI for the mean of a 0/1 array."""
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = float(values.mean())
    if n < 5:
        return mean, mean, mean
    samples = rng.choice(values, size=(B, n), replace=True).mean(axis=1)
    return mean, float(np.quantile(samples, alpha / 2)), float(np.quantile(samples, 1 - alpha / 2))


# ---------------------------------------------------------------------------
# Compute the (model x perturbation) accuracy table + drop-from-clean table
# ---------------------------------------------------------------------------

def compute_grid() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (acc_df, drop_df, drop_ci_halfwidth_df) indexed by model name x display perturbation."""
    pert_codes = [code for _, code, _ in PERTURBATIONS]
    pert_display = [d for d, _, _ in PERTURBATIONS]
    code2disp = {code: disp for disp, code, _ in PERTURBATIONS}

    acc = pd.DataFrame(index=[m[0] for m in MODELS], columns=pert_display, dtype=float)
    drop = pd.DataFrame(index=[m[0] for m in MODELS], columns=pert_display, dtype=float)
    halfwidth = pd.DataFrame(index=[m[0] for m in MODELS], columns=pert_display, dtype=float)
    clean_acc = pd.Series(index=[m[0] for m in MODELS], dtype=float)

    for name, family, size, dirname in MODELS:
        df = load_per_sample(dirname)
        if df is None:
            continue
        # Clean accuracy
        clean_mask = df["perturbation"].isna() | (df["perturbation"] == "clean")
        clean_vals = df.loc[clean_mask, "score"].values.astype(float)
        if len(clean_vals) == 0:
            continue
        c_mean, c_lo, c_hi = bootstrap_ci(clean_vals)
        clean_acc[name] = c_mean
        # Per-perturbation accuracy and drop
        for code in pert_codes:
            sub = df.loc[df["perturbation"] == code, "score"].values.astype(float)
            if len(sub) == 0:
                continue
            p_mean, p_lo, p_hi = bootstrap_ci(sub)
            acc.at[name, code2disp[code]] = p_mean
            d = c_mean - p_mean
            drop.at[name, code2disp[code]] = d
            # Half-width on the drop = bootstrap on the difference: compute pair-resampled drops
            # Approximation: assume independence, sum variances → here we just use perturbed CI half-width
            halfwidth.at[name, code2disp[code]] = (p_hi - p_lo) / 2

    return acc, drop, halfwidth, clean_acc


# ---------------------------------------------------------------------------
# Figure 2 — heatmap
# ---------------------------------------------------------------------------

def render_heatmap(drop: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    data = drop.values.astype(float) * 100.0  # %
    # Mask NaN
    mask = np.isnan(data)
    data_masked = np.where(mask, 0, data)

    im = ax.imshow(data_masked, cmap="viridis", aspect="auto",
                   vmin=0, vmax=max(40, np.nanmax(data) if np.isfinite(np.nanmax(data)) else 40))
    ax.set_xticks(range(len(drop.columns)))
    ax.set_xticklabels(drop.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(drop.index)))
    ax.set_yticklabels(drop.index, fontsize=8)
    # MDP component vertical dividers
    boundaries = [4, 10, 16]  # after Obs(4), Act(6), Rew(6); Trn last
    for b in boundaries:
        ax.axvline(b - 0.5, color="white", lw=1.2)
    # Annotate cells. Pick text colour by the cell's background luminance
    # so dark-blue / dark-purple cells (low values in viridis) get white
    # text and light yellow-green cells (high values) get black text.
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if mask[i, j]:
                continue
            v = data[i, j]
            r, g, b, _ = im.cmap(im.norm(v))
            # Rec. 601 luminance; threshold 0.55 keeps text readable on
            # the mid-band (teal) while flipping to black on the yellow tail.
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    fontsize=6, color="black" if lum > 0.55 else "white")
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("Accuracy drop from clean (%)", fontsize=9)
    ax.set_title("Clean→perturbed accuracy drop, 18 models × 22 perturbations",
                 fontsize=10)
    plt.tight_layout()
    out = FIGS / "heatmap.pdf"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Figure 3 — scaling scatter
# ---------------------------------------------------------------------------

def render_scaling(drop: pd.DataFrame, clean_acc: pd.Series) -> None:
    # Average drop per MDP component for each model
    by_comp = {"Obs": [], "Act": [], "Rew": [], "Trn": []}
    for disp, code, comp in PERTURBATIONS:
        by_comp[comp].append(disp)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    # Top: per-MDP-component drop vs size
    for comp, color in MDP_COLORS.items():
        cols = by_comp[comp]
        for name, family, size, dirname in MODELS:
            if family in ("ours-SFT", "TL", "Qwen2.5", "Llama3.2", "DeepSeek-R1"):
                continue  # only the within-family scaling comparison
            sub = drop.loc[name, cols].astype(float).dropna()
            if len(sub) == 0:
                continue
            ax1.scatter(size, sub.mean() * 100, s=70,
                        marker=FAMILY_MARKERS.get(family, "o"),
                        c=color, alpha=0.85, edgecolor="white", lw=0.6)

    ax1.set_ylabel("Avg drop within MDP component (%)", fontsize=10)
    ax1.set_xscale("log")
    ax1.grid(True, which="both", alpha=0.25)
    # Legend: MDP color
    handles_color = [plt.Line2D([0], [0], marker="o", linestyle="", color=c,
                                label=k, markersize=8)
                     for k, c in MDP_COLORS.items()]
    leg1 = ax1.legend(handles=handles_color, loc="upper right", title="MDP",
                      fontsize=8, title_fontsize=9, frameon=False)
    ax1.add_artist(leg1)
    # Legend: family marker
    handles_marker = []
    for fam, m in FAMILY_MARKERS.items():
        if fam in ("ours-SFT", "TL", "Qwen2.5", "Llama3.2", "DeepSeek-R1"):
            continue
        handles_marker.append(plt.Line2D([0], [0], marker=m, linestyle="",
                                          color="black", label=fam, markersize=8))
    ax1.legend(handles=handles_marker, loc="upper left", title="Family",
               fontsize=8, title_fontsize=9, frameon=False)
    ax1.set_title("Drop is invariant to model size; clean accuracy scales as expected",
                  fontsize=10)

    # Bottom: clean accuracy vs size
    for name, family, size, dirname in MODELS:
        if family in ("ours-SFT", "TL", "Qwen2.5", "Llama3.2", "DeepSeek-R1"):
            continue
        v = clean_acc.get(name)
        if v is None or np.isnan(v):
            continue
        ax2.scatter(size, v * 100, s=70,
                    marker=FAMILY_MARKERS.get(family, "o"),
                    c="black", alpha=0.85, edgecolor="white", lw=0.6)
    ax2.set_xlabel("Model size (B parameters, log scale)", fontsize=10)
    ax2.set_ylabel("Clean acc (%)", fontsize=10)
    ax2.grid(True, which="both", alpha=0.25)

    plt.tight_layout()
    out = FIGS / "scaling.pdf"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Table 5 — main results LaTeX
# ---------------------------------------------------------------------------

OUR_MODELS = {"ToolRL-DR-Full (ours)", "ToolRL-DR-Mixed (ours)"}

MODEL_GROUPS = [
    # Group order: ours first to immediately establish the 3B-beats-8B story,
    # then 8B/9B/14B open-source competitors, then small-and-base, then 32B,
    # then the closed frontier. Within each group rows are sorted by Pert. Acc.
    # at render time (descending).
    ("Our 3B trained checkpoints", [
        "ToolRL-DR-Full (ours)", "ToolRL-DR-Mixed (ours)",
    ]),
    ("Open-source RL tool models, 8B--14B", [
        "MUA-RL-8B", "MUA-RL-14B", "LoopTool-8B",
    ]),
    ("Open-source RL tool models, 1.5B--7B", [
        "ToolRL-Qwen2.5-1.5B", "ToolRL-Qwen2.5-3B", "ToolRL-Llama3.2-3B",
        "TL-CodeLLaMA-2",
    ]),
    ("Open-source RL tool models, 32B", [
        "MUA-RL-32B", "LoopTool-32B",
    ]),
    ("Base / instruct counterparts", [
        "Qwen2.5-1.5B-Instruct", "Qwen2.5-3B-Instruct", "Llama-3.2-3B-Instruct",
        "Qwen3-8B", "Qwen3-14B", "Qwen3-32B",
    ]),
    ("Frontier (open weights, reasoning / general)", [
        "Qwen3.5-9B", "DeepSeek-R1-Distill-14B",
    ]),
    ("Closed-source frontier", [
        "o4-mini (OpenAI)",
    ]),
    ("ToolRL SFT version", [
        "SFT-Clean-4k-3B",
    ]),
]


def render_main_table(drop: pd.DataFrame, halfwidth: pd.DataFrame,
                      clean_acc: pd.Series, acc: pd.DataFrame) -> None:
    """Compact main results table (model x clean+pert.acc+4 per-component drops).

    Convention for the four right-most columns: $\\Delta_X$ = (clean accuracy)
    minus (mean accuracy under perturbations of MDP component X), in
    percentage points. Larger $\\Delta$ = more brittle on that component.
    The two leftmost numeric columns (Clean, Pert.\\ Acc.) are accuracies
    where larger = better; the four $\\Delta$ columns are gaps where smaller
    = better. We bold our two trained checkpoints.
    """
    by_comp = {"Obs": [], "Act": [], "Rew": [], "Trn": []}
    for disp, code, comp in PERTURBATIONS:
        by_comp[comp].append(disp)

    def model_row_data(name: str):
        if pd.isna(clean_acc.get(name)):
            return None
        non_clean_cols = [d for (d, _c, _comp) in PERTURBATIONS]
        accs = acc.loc[name, non_clean_cols].astype(float).dropna()
        pert_overall = accs.mean() * 100 if len(accs) else float("nan")
        comp_drops = {}
        for comp_name in ["Obs", "Act", "Rew", "Trn"]:
            cols = by_comp[comp_name]
            d = drop.loc[name, cols].astype(float).dropna()
            comp_drops[comp_name] = (d.mean() * 100) if len(d) > 0 else None
        return {
            "name": name,
            "clean": clean_acc[name] * 100,
            "pert": pert_overall,
            "drops": comp_drops,
        }

    def fmt_row(rec):
        bold = rec["name"] in OUR_MODELS
        wrap = (lambda s: f"\\textbf{{{s}}}") if bold else (lambda s: s)
        cells = [
            wrap(rec["name"]),
            wrap(f"{rec['clean']:.1f}"),
            wrap(f"{rec['pert']:.1f}"),
        ]
        for comp_name in ["Obs", "Act", "Rew", "Trn"]:
            d = rec["drops"][comp_name]
            cells.append(wrap("---" if d is None else f"{d:.1f}"))
        return " & ".join(cells) + " \\\\"

    lines = [
        "% Auto-generated by scripts/render_figures.py.",
        "\\begin{table}[!htb]",
        "\\centering",
        "\\caption{Main results across 20 models and 22 perturbations. \\emph{Clean} (199 unperturbed samples) and \\emph{Pert.\\ Acc.} (perturbed-only weighted average across the 22 perturbation cells, clean excluded) are accuracies (\\emph{higher = better}). The four $\\Delta$ columns are per-MDP-component robustness gaps in percentage points: $\\Delta_X = \\text{Clean} - \\text{mean}(\\text{Acc on }X\\text{-perturbations})$ (\\emph{smaller = better}; large $\\Delta$ means the model is brittle on that MDP component). All values are percentages. Rows are sorted within each group by Pert.\\ Acc.\\ descending; \\textbf{bold} marks our two trained checkpoints.}",
        "\\label{tab:main}",
        "\\small",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        " & Clean $\\uparrow$ & Pert.\\ Acc.\\ $\\uparrow$ & $\\Delta_{\\mathrm{Obs}}$ $\\downarrow$ & $\\Delta_{\\mathrm{Act}}$ $\\downarrow$ & $\\Delta_{\\mathrm{Rew}}$ $\\downarrow$ & $\\Delta_{\\mathrm{Trn}}$ $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for i, (group_name, members) in enumerate(MODEL_GROUPS):
        if i > 0:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{group_name}}}}} \\\\")
        # Sort group members by Pert. Acc. descending
        recs = [model_row_data(m) for m in members]
        recs = [r for r in recs if r is not None]
        recs.sort(key=lambda r: -r["pert"])
        for r in recs:
            lines.append(fmt_row(r))
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    out = TABS / "main_results.tex"
    out.write_text("\n".join(lines))
    print(f"  wrote {out}")


def render_per_perturbation_tables(drop: pd.DataFrame, acc: pd.DataFrame,
                                   clean_acc: pd.Series) -> None:
    """4 sub-tables (one per MDP component) showing per-perturbation acc with drop."""
    by_comp = {"Obs": [], "Act": [], "Rew": [], "Trn": []}
    for disp, code, comp in PERTURBATIONS:
        by_comp[comp].append(disp)

    component_titles = {
        "Obs": ("Observation", "obs"),
        "Act": ("Action", "act"),
        "Rew": ("Reward", "rew"),
        "Trn": ("Transition", "trn"),
    }

    out_lines = [
        "% Auto-generated by scripts/render_figures.py.",
        "% One sub-table per MDP component, showing per-perturbation accuracy and",
        "% drop-from-clean (in parentheses) for every model.",
        "",
    ]

    for comp_short, (comp_full, comp_label) in component_titles.items():
        cols = by_comp[comp_short]
        # +1 for the Clean reference column we now include up front so the
        # drops (in parens) are anchored to a visible baseline in the table itself.
        col_spec = "lr" + "r" * len(cols)
        # Header: Clean | per-perturbation display names
        header = " & Clean & " + " & ".join(cols) + " \\\\"

        out_lines.append(f"\\begin{{table}}[!htb]")
        out_lines.append("\\centering")
        out_lines.append(f"\\caption{{Per-perturbation accuracy under \\textbf{{{comp_full}}} perturbations. The first numeric column is Clean accuracy (the unperturbed baseline); each subsequent cell shows \\emph{{accuracy on that perturbation}} with the drop \\emph{{from this model's own Clean}} in parentheses. A more negative parenthesised number means a larger sim-to-real gap (worse robustness). All values are percentages. \\textbf{{Bold}} = our trained checkpoints.}}")
        out_lines.append(f"\\label{{tab:per-pert-{comp_label}}}")
        out_lines.append("\\scriptsize")
        out_lines.append("\\setlength{\\tabcolsep}{3pt}")
        out_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        out_lines.append("\\toprule")
        out_lines.append(header)
        out_lines.append("\\midrule")

        for i, (group_name, members) in enumerate(MODEL_GROUPS):
            if i > 0:
                out_lines.append("\\midrule")
            out_lines.append(f"\\multicolumn{{{len(cols)+2}}}{{l}}{{\\emph{{{group_name}}}}} \\\\")
            for m in members:
                if pd.isna(clean_acc.get(m)):
                    continue
                bold = m in OUR_MODELS
                wrap = (lambda s: f"\\textbf{{{s}}}") if bold else (lambda s: s)
                cells = [wrap(m), wrap(f"{clean_acc[m]*100:.1f}")]
                for c in cols:
                    a = acc.loc[m, c]
                    d = drop.loc[m, c]
                    if pd.isna(a):
                        cells.append(wrap("---"))
                    else:
                        cells.append(wrap(f"{a*100:.1f} ({-d*100:+.1f})"))
                out_lines.append(" & ".join(cells) + " \\\\")

        out_lines.append("\\bottomrule")
        out_lines.append("\\end{tabular}")
        out_lines.append("\\end{table}")
        out_lines.append("")

    out = TABS / "per_perturbation_tables.tex"
    out.write_text("\n".join(out_lines))
    print(f"  wrote {out}")


def main():
    print("[1/4] Loading per-sample scores...")
    acc, drop, halfwidth, clean_acc = compute_grid()
    print(f"      computed for {(~clean_acc.isna()).sum()} models")
    print("[2/4] Rendering heatmap...")
    render_heatmap(drop)
    # NOTE: scaling.pdf is now produced by scripts/fig2_scaling.py (curated
    # 2x2 small-multiples version). The auto-generated render_scaling output
    # was visually inferior; we keep the function for reference but skip the
    # call so a fig2_scaling.pdf copy isn't overwritten by the auto version.
    print("[3/5] Rendering scaling scatter (SKIPPED — using scripts/fig2_scaling.py output)")
    # render_scaling(drop, clean_acc)
    print("[4/5] Writing main results LaTeX...")
    render_main_table(drop, halfwidth, clean_acc, acc)
    print("[5/5] Writing per-perturbation appendix tables...")
    render_per_perturbation_tables(drop, acc, clean_acc)
    print("\nDone. Files:")
    print(f"  {FIGS / 'heatmap.pdf'}")
    print(f"  {FIGS / 'scaling.pdf'}")
    print(f"  {TABS / 'main_results.tex'}")
    print(f"  {TABS / 'per_perturbation_tables.tex'}")


if __name__ == "__main__":
    main()
