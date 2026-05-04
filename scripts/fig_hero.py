"""
Hero figure: ToolRL-DR robustness retention vs model size.  [REVISED LAYOUT]

Same data, same scientific content as the original fig_hero.py.  The only
changes are cosmetic and target the label-collision issues in the v1 figure:

  - figure widened from 5.8x4.0 to 7.0x4.3 in (extra horizontal room)
  - label placement uses `adjustText` (force-directed) for all non-pinned
    labels; the original hand-rolled left/right solver is gone
  - PINNED dict expanded so the high-value labels (the two red stars + o4-mini)
    sit on the side of their marker that doesn't collide with neighbours and
    doesn't cross the trend line

If `adjustText` is unavailable, we fall back to a (still improved) manual
solver — never silently overlap.

Data is read directly from each model's
<repo-root>/results/robustbench/results/<dirname>/scores.json so the figure
re-renders correctly when scores change.  No hand-entered numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    from adjustText import adjust_text
    HAVE_ADJUSTTEXT = True
except ImportError:
    HAVE_ADJUSTTEXT = False

# NeurIPS-friendly rcParams
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})

RESULTS = Path("<repo-root>/results/robustbench/results")

O4MINI_X = 50.0
MODELS = [
    ("Qwen2.5-1.5B-Inst",       1.5, "Qwen__Qwen2.5-1.5B-Instruct",                            "general"),
    ("Qwen2.5-3B-Inst",         3.0, "Qwen__Qwen2.5-3B-Instruct",                              "general"),
    ("Llama-3.2-3B-Inst",       3.0, "meta-llama__Llama-3.2-3B-Instruct",                      "general"),
    ("Qwen3-8B",                8.0, "Qwen__Qwen3-8B",                                         "general"),
    ("Qwen3-14B",              14.0, "Qwen__Qwen3-14B",                                        "general"),
    ("Qwen3-32B",              32.0, "Qwen__Qwen3-32B",                                        "general"),
    ("Qwen3.5-9B",              9.0, "Qwen__Qwen3.5-9B",                                       "general"),
    ("DeepSeek-R1-Distill-14B",14.0, "deepseek-ai__DeepSeek-R1-Distill-Qwen-14B",              "general"),
    ("ToolRL-1.5B",             1.5, "chengq9__ToolRL-Qwen2.5-1.5B",                           "rl_tool"),
    ("ToolRL-3B",               3.0, "chengq9__ToolRL-Qwen2.5-3B",                             "rl_tool"),
    ("ToolRL-Llama3.2-3B",      3.0, "chengq9__ToolRL-Llama3.2-3B",                            "rl_tool"),
    ("TL-CodeLLaMA-2",          7.0, "Junjie-Ye__TL-CodeLLaMA-2",                              "rl_tool"),
    ("LoopTool-8B",             8.0, "zhuiguang-ning__LoopTool-8B",                            "rl_tool"),
    ("LoopTool-32B",           32.0, "zhuiguang-ning__LoopTool-32B",                           "rl_tool"),
    ("MUA-RL-8B",               8.0, "zzwkk__MUA-RL-8B",                                       "rl_tool"),
    ("MUA-RL-14B",             14.0, "zzwkk__MUA-RL-14B",                                      "rl_tool"),
    ("MUA-RL-32B",             32.0, "zzwkk__MUA-RL-32B",                                      "rl_tool"),
    ("SFT-Clean-4k",            3.0, "__projects__bfdz__anon-user__robustbench__sft_models__SFT-Clean-4k-Qwen2.5-3B-Instruct",
                                                                                                "sft"),
    ("o4-mini",            O4MINI_X, "openai__o4-mini",                                        "closed"),
    ("ToolRL-DR-Full (ours)",   3.0, "<anonymized>__qwen2.5-3b-toolrl-robustbench-perturbed",    "ours"),
    ("ToolRL-DR-Mixed (ours)",  3.0, "<anonymized>__qwen2.5-3b-toolrl-robustbench-mixed",        "ours"),
]

# Map perturbation cell -> MDP component for the retention computation.
PERT_TO_MDP = {
    "realistic_typos": "obs", "query_paraphrase": "obs",
    "paraphrase_tool_description": "obs", "paraphrase_parameter_description": "obs",
    "same_name_A": "act", "same_name_B": "act", "same_name_C": "act",
    "same_name_D": "act", "same_name_E": "act", "redundant": "act",
    "CD": "rew", "TD": "rew", "CD_NT": "rew", "TD_NT": "rew",
    "CD_AB": "rew", "TD_AB": "rew",
}

SKIP_TEXT: set[str] = set()

# Hand-pinned offsets (in display points) for high-value labels.  These three
# are placed FIRST and excluded from the adjustText force solver below so
# their position is deterministic across re-renders.
PINNED = {
    # Both red stars sit at x=3B and on the LEFT side of every other 3B point,
    # so we route their labels to the RIGHT (into open space).
    "ToolRL-DR-Full (ours)":  dict(dx=16, dy=0,   ha="left",  va="center"),
    "ToolRL-DR-Mixed (ours)": dict(dx=16, dy=0,   ha="left",  va="center"),
    # o4-mini's marker sits ON the trend line; pin its label above so the
    # trend line doesn't bisect the text.
    "o4-mini":                dict(dx=0,  dy=14,  ha="center", va="bottom"),
}


def retention_for_model(scores_path: Path) -> float | None:
    if not scores_path.exists():
        return None
    data = json.loads(scores_path.read_text())
    by_pert = data["overall"].get("by_perturbation", {})
    if "clean" not in by_pert:
        return None
    clean_acc = by_pert["clean"]["accuracy"]
    if clean_acc <= 0:
        return None
    drops_by_comp: dict[str, list[float]] = {"obs": [], "act": [], "rew": []}
    for cell, info in by_pert.items():
        comp = PERT_TO_MDP.get(cell)
        if comp is None:
            continue
        drops_by_comp[comp].append(clean_acc - info["accuracy"])
    if any(len(v) == 0 for v in drops_by_comp.values()):
        return None
    mean_per_comp = [sum(v) / len(v) for v in drops_by_comp.values()]
    avg_drop = sum(mean_per_comp) / len(mean_per_comp)
    return (1.0 - avg_drop / clean_acc) * 100.0


def collect():
    grouped = {"general": [], "rl_tool": [], "sft": [], "closed": [], "ours": []}
    for label, size_b, dirname, group in MODELS:
        ret = retention_for_model(RESULTS / dirname / "scores.json")
        if ret is None:
            print(f"  SKIP {label}: no usable scores at {dirname}")
            continue
        grouped[group].append((size_b, ret, label))
        print(f"  {label:30s}  size={size_b:5.1f}B  retention={ret:5.1f}%")
    return grouped


def color_for(group: str) -> tuple[str, str]:
    if group == "ours":
        return "#791F1F", "bold"
    if group == "closed":
        return "#7A5810", "normal"
    return "#2C2C2A", "normal"


def font_for(group: str) -> tuple[float, str]:
    """(fontsize, weight). Stars get a hair larger; everyone else uniform."""
    if group == "ours":
        return 7.5, "bold"
    return 6.8, "normal"


def place_pinned(ax, points):
    """Draw the hand-pinned high-value labels and return the matplotlib Text
    objects so adjustText can be told to avoid them."""
    pinned_texts = []
    pinned_xy = []  # (x, y) for each pinned data point — adjustText avoids these
    for p in points:
        cfg = PINNED.get(p["txt"])
        if cfg is None:
            continue
        color, weight = color_for(p["group"])
        fs, _ = font_for(p["group"])
        # Connector line from marker to label anchor (only if displaced).
        if abs(cfg["dx"]) + abs(cfg["dy"]) > 6:
            disp_x, disp_y = ax.transData.transform((p["x"], p["y"]))
            anchor_data = ax.transData.inverted().transform(
                (disp_x + cfg["dx"], disp_y + cfg["dy"])
            )
            ax.plot([p["x"], anchor_data[0]], [p["y"], anchor_data[1]],
                    color="#C8C5BD", linewidth=0.4, zorder=2.5)
        t = ax.annotate(
            p["txt"], xy=(p["x"], p["y"]),
            xytext=(cfg["dx"], cfg["dy"]),
            textcoords="offset points",
            ha=cfg["ha"], va=cfg["va"],
            fontsize=fs, fontweight=weight, color=color, zorder=6,
        )
        pinned_texts.append(t)
        pinned_xy.append((p["x"], p["y"]))
    return pinned_texts, pinned_xy


def place_auto_with_adjusttext(ax, fig, points, all_xy):
    """Place every non-pinned label with adjustText.  `all_xy` is the list of
    EVERY data point coordinate (including pinned ones) so adjustText pushes
    labels away from all markers."""
    auto_records = [p for p in points if p["txt"] not in PINNED]
    texts = []
    for p in auto_records:
        color, weight = color_for(p["group"])
        fs, _ = font_for(p["group"])
        # Initial placement: a bit right and up.  adjustText will move it.
        texts.append(ax.text(
            p["x"], p["y"], p["txt"],
            fontsize=fs, fontweight=weight, color=color,
            ha="left", va="center", zorder=5,
        ))
    xs = [xy[0] for xy in all_xy]
    ys = [xy[1] for xy in all_xy]
    # Force-directed placement.  Tuned for log-x mixed-density scatter:
    #   - expand_*  : bounding-box padding around text/points
    #   - force_*   : strength of repulsion (text-text, text-point, text-axes)
    #   - arrowprops: thin gray leader lines for displaced labels
    adjust_text(
        texts,
        x=xs, y=ys,
        ax=ax,
        expand=(1.20, 1.35),
        force_text=(0.35, 0.55),
        force_static=(0.35, 0.45),
        force_explode=(0.05, 0.08),
        only_move={"text": "xy", "static": "xy", "explode": "xy"},
        arrowprops=dict(arrowstyle="-", color="#B6B3AA", lw=0.4, shrinkA=2, shrinkB=2),
        max_move=(60, 60),
    )


def place_auto_fallback(ax, fig, points):
    """Manual fallback if adjustText is unavailable.  Tries 8 directions per
    label instead of the original 2, with a global box-overlap test."""
    inv = ax.transData
    fig.canvas.draw()
    CHAR_W = 4.0
    LABEL_H = 9.0
    PAD = 3.0
    x0d, y0d = ax.transData.transform((ax.get_xlim()[0], ax.get_ylim()[0]))
    x1d, y1d = ax.transData.transform((ax.get_xlim()[1], ax.get_ylim()[1]))
    placed = []
    for p in points:
        mx, my = ax.transData.transform((p["x"], p["y"]))
        placed.append((mx - 7, mx + 7, my - 7, my + 7, "marker"))

    def overlaps(box):
        l, r, b, t, _ = box
        for L, R, B, T, _ in placed:
            if not (r < L or l > R or t < B or b > T):
                return True
        return False

    pinned_records = [p for p in points if p["txt"] in PINNED]
    auto_records = [p for p in points if p["txt"] not in PINNED]

    for p in pinned_records:
        cfg = PINNED[p["txt"]]
        mx, my = ax.transData.transform((p["x"], p["y"]))
        w = max(8, len(p["txt"]) * CHAR_W)
        ax_x = mx + cfg["dx"]; ay = my + cfg["dy"]
        if cfg["ha"] == "left":  l, r = ax_x, ax_x + w
        elif cfg["ha"] == "right": l, r = ax_x - w, ax_x
        else: l, r = ax_x - w/2, ax_x + w/2
        b, t = ay - LABEL_H/2, ay + LABEL_H/2
        placed.append((l, r, b, t, "label"))
        color, weight = color_for(p["group"])
        fs, _ = font_for(p["group"])
        if abs(cfg["dx"]) + abs(cfg["dy"]) > 6:
            anchor = inv.inverted().transform((mx + cfg["dx"], my + cfg["dy"]))
            ax.plot([p["x"], anchor[0]], [p["y"], anchor[1]],
                    color="#C8C5BD", linewidth=0.4, zorder=2.5)
        ax.annotate(p["txt"], xy=(p["x"], p["y"]),
                    xytext=(cfg["dx"], cfg["dy"]), textcoords="offset points",
                    ha=cfg["ha"], va=cfg["va"], fontsize=fs,
                    fontweight=weight, color=color, zorder=6)

    auto_records.sort(key=lambda p: -p["y"])
    OFF = 9.0
    # 8 anchor directions: E, NE, N, NW, W, SW, S, SE  (in display pts)
    DIRS = [
        (OFF,   0,  "left"),
        (OFF,  10,  "left"),
        (-OFF,  0,  "right"),
        (-OFF, 10,  "right"),
        (OFF, -10,  "left"),
        (-OFF,-10,  "right"),
        (OFF,  20,  "left"),
        (-OFF, 20,  "right"),
        (OFF, -20,  "left"),
        (-OFF,-20,  "right"),
        (OFF,  30,  "left"),
        (-OFF, 30,  "right"),
    ]
    for p in auto_records:
        mx, my = ax.transData.transform((p["x"], p["y"]))
        w = max(8, len(p["txt"]) * CHAR_W)
        chosen = None
        for dx, dy, ha in DIRS:
            ay = my + dy
            if ha == "left":   l, r = mx + dx, mx + dx + w
            else:              l, r = mx + dx - w, mx + dx
            b, t = ay - LABEL_H/2 - PAD/2, ay + LABEL_H/2 + PAD/2
            if l < x0d - 4 or r > x1d + 4: continue
            if b < y0d - 4 or t > y1d + 4: continue
            box = (l, r, b, t, "label")
            if not overlaps(box):
                chosen = (dx, dy, ha, box); break
        if chosen is None:
            chosen = (OFF, 0, "left",
                      (mx+OFF, mx+OFF+w, my-LABEL_H/2, my+LABEL_H/2, "label"))
        dx, dy, ha, box = chosen
        placed.append(box)
        if abs(dy) > 4 or abs(dx) > 4:
            anchor = inv.inverted().transform((mx + dx, my + dy))
            ax.plot([p["x"], anchor[0]], [p["y"], anchor[1]],
                    color="#C8C5BD", linewidth=0.4, zorder=2.5)
        color, weight = color_for(p["group"])
        fs, _ = font_for(p["group"])
        ax.annotate(p["txt"], xy=(p["x"], p["y"]),
                    xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="center", fontsize=fs,
                    fontweight=weight, color=color, zorder=5)


def main() -> None:
    print("Reading per-model scores...")
    grouped = collect()
    general = grouped["general"]; rl_tool = grouped["rl_tool"]
    sft = grouped["sft"]; closed = grouped["closed"]; ours = grouped["ours"]

    trend_pts = general + rl_tool + sft
    if len(trend_pts) < 2:
        raise RuntimeError("Need >=2 trend points to fit baseline trend.")
    xs_arr = np.array([p[0] for p in trend_pts])
    ys_arr = np.array([p[1] for p in trend_pts])
    slope, intercept = np.polyfit(np.log10(xs_arr), ys_arr, 1)
    xt = np.logspace(np.log10(1.3), np.log10(40), 80)
    yt = slope * np.log10(xt) + intercept

    all_y = ys_arr.tolist() + [p[1] for p in closed] + [p[1] for p in ours]
    y_lo = max(0, min(all_y) - 4)
    y_hi = min(100, max(all_y) + 4)

    # Wider canvas — the single biggest gain.  Aspect kept similar.
    fig, ax = plt.subplots(figsize=(7.0, 4.3))

    ax.plot(xt, yt, linestyle="--", color="#888780", linewidth=1.0,
            label="Open-source baseline trend", zorder=1)

    def scatter(data, marker, face, edge, size, label, zorder=3, lw=0.6):
        xs = [d[0] for d in data]; ys = [d[1] for d in data]
        ax.scatter(xs, ys, marker=marker, c=face, edgecolors=edge,
                   s=size, label=label, zorder=zorder, linewidths=lw)

    scatter(general, "o", "#185FA5", "#0C447C", 32, "Base / Instruct / Reasoning (open)")
    scatter(rl_tool, "^", "#534AB7", "#3C3489", 42, "RL-trained tool models")
    scatter(closed,  "D", "#D4A017", "#8B6914", 60, "o4-mini (closed-source)", zorder=3)
    scatter(sft,     "x", "#5F5E5A", "#5F5E5A", 50, "SFT-Clean baseline", zorder=2, lw=1.4)
    scatter(ours,    "*", "#A32D2D", "#501313", 220, "ToolRL-DR (ours)", zorder=4, lw=0.7)

    if closed:
        ax.axvline(43, ymin=0.04, ymax=0.96,
                   color="#B4B2A9", linewidth=0.5, linestyle=":", zorder=0)
        ax.text(O4MINI_X, y_lo + 0.7, "(undisclosed)",
                ha="center", va="center",
                fontsize=7, fontstyle="italic", color="#888780")

    ax.set_xscale("log")
    ax.set_xlim(1.2, 60 if closed else 44)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Model size (parameters, log scale)")
    ax.set_ylabel("Robustness retention on Obs+Act+Rew (%)")
    if closed:
        ax.set_xticks([1.5, 3, 8, 14, 32, O4MINI_X])
        ax.set_xticklabels(["1.5B", "3B", "8B", "14B", "32B", "o4-mini"])
    else:
        ax.set_xticks([1.5, 3, 8, 14, 32])
        ax.set_xticklabels(["1.5B", "3B", "8B", "14B", "32B"])
    ax.grid(True, alpha=0.25, linewidth=0.4)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.legend(loc="lower right", frameon=False,
              handletextpad=0.4, borderaxespad=0.3, labelspacing=0.3)

    # Build the annotation list (group tag carried for color/weight).
    annotated = []
    for size_b, ret, label in (general + rl_tool + sft + closed + ours):
        if label in SKIP_TEXT:
            continue
        if   any(p[2] == label for p in general): grp = "general"
        elif any(p[2] == label for p in rl_tool): grp = "rl_tool"
        elif any(p[2] == label for p in sft):     grp = "sft"
        elif any(p[2] == label for p in closed):  grp = "closed"
        else:                                     grp = "ours"
        annotated.append({"x": size_b, "y": ret, "txt": label, "group": grp})

    all_xy = [(p["x"], p["y"]) for p in annotated]

    # 1. Pinned labels first (fixed by hand)
    place_pinned(ax, annotated)
    # 2. Everything else: adjustText if available, fallback otherwise
    if HAVE_ADJUSTTEXT:
        place_auto_with_adjusttext(ax, fig, annotated, all_xy)
    else:
        print("  adjustText not installed -> using fallback solver")
        place_auto_fallback(ax, fig, annotated)

    plt.tight_layout()
    out_pdf = Path(__file__).resolve().parent / "fig_hero.pdf"
    out_png = Path(__file__).resolve().parent / "fig_hero.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"Wrote {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()
