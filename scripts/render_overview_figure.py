"""Render Figure 1 — the MDP-perturbation taxonomy overview diagram.

Pure matplotlib (no external API), saves as PDF, uses Okabe-Ito-style
palette consistent with the rest of the paper. 4 panels, one per MDP
component, each with: a code-style example showing the perturbation,
and a citation tag for the verified GitHub issue.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Use TrueType (Type 42) fonts so Overleaf / NeurIPS pdflatex always reads it —
# avoids the silent "reading image file failed" Type-3 issue.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

OUT = Path(__file__).resolve().parents[1] / "paper" / "neurips_2026_v2" / "figures" / "overview.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Palette (matches Table 3 group banding)
COLORS = {
    "Observation": "#7F77DD",
    "Action":      "#D85A30",
    "Reward":      "#1D9E75",
    "Transition":  "#B5872E",
}

PANEL_DATA = [
    ("Observation", "User query / tool description corrupted",
     "User: \"find a flght frm Boston to NYC\"\n          v\nLLM emits tool name with typo →\n  ValueError: tool not found",
     "LlamaIndex #7170"),
    ("Action", "Action space contaminated by distractors",
     "Tools registered:\n  search_flights(...)   ← canonical\n  search_flights(...)   ← MCP duplicate\n          v\nAgent SDK hangs",
     "openai-agents #1167"),
    ("Reward", "Tool metadata misleading vs. implementation",
     "Description: \"Filter by id__in (all fields)\"\nReality: relationship fields ignore __in\n          v\nReturns ALL records silently",
     "netbox-mcp #79"),
    ("Transition", "Tool execution returns transient error",
     "tool_call()  ->  Timeout / 429 /\n                 401 / 5xx /\n                 malformed / drift\n          v\nDoes the agent retry?",
     "langchain #35597"),
]


def draw_panel(ax, title, subtitle, body, citation, color):
    # Background rectangle
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.02",
        facecolor=color + "20", edgecolor=color, linewidth=1.6, transform=ax.transAxes))
    # Title strip
    ax.add_patch(mpatches.Rectangle(
        (0.02, 0.82), 0.96, 0.16, facecolor=color, edgecolor="none",
        transform=ax.transAxes))
    ax.text(0.5, 0.90, title, transform=ax.transAxes, ha="center", va="center",
            fontsize=12, fontweight="bold", color="white")
    ax.text(0.5, 0.85, subtitle, transform=ax.transAxes, ha="center", va="center",
            fontsize=8, color="white", style="italic")
    # Body (monospace example)
    ax.text(0.07, 0.45, body, transform=ax.transAxes,
            ha="left", va="center", fontsize=8.5, family="monospace",
            color="#222")
    # Citation tag at bottom
    ax.text(0.5, 0.10, f"  Production case:  {citation}  ", transform=ax.transAxes,
            ha="center", va="center", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=color, linewidth=1))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.5))
    fig.suptitle("Tool-use MDP $\\langle\\mathcal{S}, \\mathcal{A}, \\mathcal{P}, \\mathcal{R}\\rangle$ "
                 "$\\to$ four perturbation categories, each grounded in a verified production failure",
                 fontsize=11, y=1.01)
    for ax, (title, subtitle, body, cit) in zip(axes.flat, PANEL_DATA):
        draw_panel(ax, title, subtitle, body, cit, COLORS[title])
    plt.tight_layout()
    plt.savefig(OUT, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
