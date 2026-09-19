import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = "anchor_efficiency_plots"
DPI = 600
FILE_FORMAT = "png"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# GLOBAL STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "savefig.dpi": DPI,
    "figure.dpi": 150,
})

# ============================================================
# DATA
# ============================================================

datasets = [
    "P12",
    "USHCN",
    "HumanActivity",
    "MIMIC",
]

methods = [
    "NaiveAnchor",
    "ExpoAnchor",
    "SparseAnchor",
    "ERMAnchor",
    "AutoAnchor",
]

# CPU milliseconds per prediction
data = {
    "NaiveAnchor": [
        0.4245,
        0.8301,
        0.2578,
        1.4117,
    ],

    "ExpoAnchor": [
        0.4354,
        0.7725,
        0.2853,
        1.4151,
    ],

    "SparseAnchor": [
        0.4242,
        0.8339,
        0.2435,
        1.4466,
    ],

    "ERMAnchor": [
        0.4387,
        0.8832,
        0.2851,
        1.4475,
    ],

    "AutoAnchor": [
        0.4463,
        0.9278,
        0.2829,
        1.4703,
    ],
}

# ============================================================
# COLORS
# ============================================================

colors = {
    "NaiveAnchor": "#B8C7B1",
    "ExpoAnchor": "#D28B7B",
    "SparseAnchor": "#1F3A56",
    "ERMAnchor": "#8A8A8A",
    "AutoAnchor": "#111111",
}

# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(10.5, 5.8))

x = np.arange(len(datasets))

bar_width = 0.15

offsets = (
    np.arange(len(methods))
    - (len(methods) - 1) / 2
) * bar_width

# ============================================================
# BARS
# ============================================================

for offset, method in zip(offsets, methods):

    values = data[method]

    bars = ax.bar(
        x + offset,
        values,
        width=bar_width,
        label=method,
        color=colors[method],
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

# ============================================================
# AXIS
# ============================================================

ax.set_ylabel(
    "CPU ms / Prediction",
    labelpad=10,
)

ax.set_xticks(x)
ax.set_xticklabels(datasets)

ax.set_xlim(
    -0.55,
    len(datasets) - 0.45,
)

# Start at zero for meaningful bar comparison
ax.set_ylim(
    0,
    1.60,
)

# ============================================================
# GRID
# ============================================================

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.25,
)

ax.set_axisbelow(True)

# ============================================================
# SPINES
# ============================================================

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ============================================================
# LEGEND
# ============================================================

ax.legend(
    loc="upper left",
    ncol=2,
    frameon=False,
    handlelength=1.3,
    handletextpad=0.5,
    columnspacing=1.2,
)

# ============================================================
# LAYOUT
# ============================================================

fig.tight_layout()

# ============================================================
# SAVE
# ============================================================

output_path = os.path.join(
    OUTPUT_DIR,
    f"anchor_efficiency_ms_per_prediction.{FILE_FORMAT}",
)

fig.savefig(
    output_path,
    dpi=DPI,
    bbox_inches="tight",
    facecolor="white",
)

plt.show()

print(f"Saved: {output_path}")