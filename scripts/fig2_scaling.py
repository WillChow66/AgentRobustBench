"""
Fig 2 redesign: per-MDP-component drop vs model size, 2x2 small multiples.
Output: fig2_scaling.pdf, fig2_scaling.png

Message: across 4 model families with multiple sizes, drops on Reward and
Transition do NOT decrease as model size grows from 1.5B to 32B. This
motivates needing a training-side fix beyond simply scaling parameters.

Data is taken from main-paper Table 5 (per-component robustness gaps).
ToolRL-DR is intentionally excluded — this figure is about the problem,
not the solution. The solution (ToolRL-DR-Full as a hero point) lives
in fig_hero.pdf.
"""

import matplotlib.pyplot as plt

# ---------- NeurIPS-friendly rcParams ----------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
})

# ---------- Data ----------
# Each family: sizes + per-component drop (%) at each size.
# Drops are taken from Table 5 columns Delta_Obs, Delta_Act, Delta_Rew, Delta_Trn.
families = [
    {
        "name": "ToolRL",
        "color": "#BA7517",
        "marker": "o",
        "sizes": [1.5, 3.0],
        "obs": [3.4, 2.1],
        "act": [16.7, 13.4],
        "rew": [45.8, 43.1],
        "trn": [38.0, 31.3],
    },
    {
        "name": "MUA-RL",
        "color": "#185FA5",
        "marker": "s",
        "sizes": [8.0, 14.0, 32.0],
        "obs": [2.2, 1.1, 2.7],
        "act": [15.0, 8.3, -1.4],
        "rew": [37.8, 31.6, 37.8],
        "trn": [23.0, 20.2, 22.3],
    },
    {
        "name": "LoopTool",
        "color": "#534AB7",
        "marker": "^",
        "sizes": [8.0, 32.0],
        "obs": [4.8, 3.5],
        "act": [16.9, 12.1],
        "rew": [40.5, 49.1],
        "trn": [29.3, 39.9],
    },
    {
        "name": "Qwen3 base",
        "color": "#1D9E75",
        "marker": "D",
        "sizes": [8.0, 14.0, 32.0],
        "obs": [1.9, 3.9, 2.3],
        "act": [17.2, 13.0, 10.3],
        "rew": [39.0, 45.1, 33.5],
        "trn": [29.4, 25.3, 37.7],
    },
]

# (key, panel title, ylim). All titles black; the cross-panel observation
# (Reward and Transition do not shrink with scale) is stated in the caption.
panels = [
    ("obs", r"(a) Observation $\Delta$ (%)",  (0, 10),  "#2C2C2A"),
    ("act", r"(b) Action $\Delta$ (%)",       (-5, 22), "#2C2C2A"),
    ("rew", r"(c) Reward $\Delta$ (%)",       (25, 55), "#2C2C2A"),
    ("trn", r"(d) Transition $\Delta$ (%)",   (15, 45), "#2C2C2A"),
]

# Single-row layout: four panels side by side. The figure is wider but
# much shorter than the 2x2 layout, which saves vertical space in §6.3.
fig, axes = plt.subplots(1, 4, figsize=(11.0, 2.4), sharex=True)

for ax, (key, title, ylim, title_color) in zip(axes, panels):
    for f in families:
        ax.plot(
            f["sizes"], f[key],
            marker=f["marker"], color=f["color"],
            markersize=4.5, linewidth=1.3, label=f["name"],
            markeredgewidth=0,
        )
    ax.set_xscale("log")
    ax.set_xlim(1.3, 38)
    ax.set_ylim(*ylim)
    ax.set_title(title, color=title_color, loc="left", pad=4)
    ax.grid(True, alpha=0.25, linewidth=0.4)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.set_xticks([1.5, 3, 8, 14, 32])
    ax.set_xticklabels(["1.5", "3", "8", "14", "32"])
    ax.set_xlabel("Model size (B, log scale)")

# Shared legend below the figure
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels, loc="lower center", ncol=4,
    frameon=False, bbox_to_anchor=(0.5, -0.04),
    handletextpad=0.4, columnspacing=1.6,
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.28)
plt.savefig("fig2_scaling.pdf", bbox_inches="tight")
plt.savefig("fig2_scaling.png", bbox_inches="tight", dpi=200)
print("Wrote fig2_scaling.pdf and fig2_scaling.png")
