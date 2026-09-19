#!/usr/bin/env python3
"""Create clear paper plots for Anchor uncertainty and ablation results.

The values are embedded from the manuscript tables. The default output is
600-DPI PNG, ready to include in Overleaf.

Outputs:
  docs/paper/figures/generated/anchor_bootstrap_ci.png
  docs/paper/figures/generated/anchor_candidate_ablation.png

Optional:
  --formats png pdf svg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


BOOTSTRAP_ROWS = [
    ("P12", "Naive", 0.3951, 0.3831, 0.4073, 0.3992, 0.3734, 0.4264, 1181),
    ("P12", "ERM", 0.3585, 0.3491, 0.3686, 0.3020, 0.2826, 0.3249, 1181),
    ("P12", "Auto", 0.3585, 0.3487, 0.3690, 0.3020, 0.2817, 0.3249, 1181),
    ("MIMIC", "Naive", 0.3708, 0.3563, 0.3849, 0.4561, 0.4136, 0.5036, 1534),
    ("MIMIC", "ERM", 0.3658, 0.3535, 0.3785, 0.3782, 0.3394, 0.4218, 1534),
    ("MIMIC", "Auto", 0.3657, 0.3537, 0.3770, 0.3785, 0.3367, 0.4187, 1534),
    ("USHCN", "Naive", 0.3291, 0.2796, 0.3784, 0.3640, 0.2664, 0.4755, 112),
    ("USHCN", "Sparse", 0.2812, 0.2384, 0.3273, 0.2661, 0.2005, 0.3385, 112),
    ("USHCN", "Auto", 0.2187, 0.1858, 0.2524, 0.1565, 0.1193, 0.1937, 112),
    ("HumanActivity", "Naive", 0.1225, 0.1122, 0.1335, 0.0605, 0.0489, 0.0738, 218),
    ("HumanActivity", "ERM", 0.1131, 0.1039, 0.1226, 0.0428, 0.0349, 0.0512, 218),
    ("HumanActivity", "Auto", 0.1148, 0.1056, 0.1249, 0.0420, 0.0345, 0.0501, 218),
]

ABLATION_ROWS = [
    ("P12", "ERM only", 0.0000, 0.0000),
    ("P12", "rules only", 0.0972, 0.0366),
    ("P12", "minus EMA", 0.0137, 0.0098),
    ("P12", "minus trend", -0.0001, -0.0001),
    ("USHCN", "ERM only", 0.0097, 0.0447),
    ("USHCN", "rules only", 0.0000, 0.0000),
    ("USHCN", "minus sparse", 0.0283, 0.0532),
    ("USHCN", "minus phase", 0.0616, 0.0418),
    ("HumanActivity", "ERM only", 0.0009, -0.0016),
    ("HumanActivity", "rules only", 0.0000, 0.0000),
    ("HumanActivity", "minus EMA", 0.0064, 0.0046),
    ("HumanActivity", "minus sparse/trend/phase", 0.0000, 0.0000),
    ("MIMIC", "ERM only", -0.0002, 0.0001),
    ("MIMIC", "rules only", 0.0775, 0.0053),
    ("MIMIC", "minus EMA", 0.0094, 0.0027),
    ("MIMIC", "minus sparse", -0.0002, 0.0001),
    ("MIMIC", "minus trend", 0.0011, 0.0016),
    ("MIMIC", "minus phase", -0.0003, 0.0003),
]

DATASETS = ["P12", "MIMIC", "USHCN", "HumanActivity"]
METHOD_COLORS = {
    "Naive": "#A1A7B3",
    "ERM": "#4C78A8",
    "Sparse": "#59A14F",
    "Auto": "#D04A5B",
}
MSE_COLOR = "#4C78A8"
MAE_COLOR = "#D04A5B"
GRID = "#E8EAF0"
TEXT = "#1F2937"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 600,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_all(fig: plt.Figure, output_dir: Path, stem: str, formats: list[str], dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output_dir / f"{stem}.{fmt}", bbox_inches="tight", dpi=dpi)


def rows_for_dataset(dataset: str) -> list[tuple]:
    return [row for row in BOOTSTRAP_ROWS if row[0] == dataset]


def set_clean_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.spines["left"].set_color("#D4D7DE")
    ax.spines["bottom"].set_color("#D4D7DE")


def add_bar_labels(ax: plt.Axes, bars, values: list[float], y_pad: float) -> None:
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_pad,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.2,
            color=TEXT,
        )


def plot_bootstrap_ci(output_dir: Path, formats: list[str], dpi: int) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(12.0, 5.8), constrained_layout=True)
    metric_specs = [
        ("MAE", 2, 3, 4, "Scaled MAE"),
        ("MSE", 5, 6, 7, "Scaled MSE"),
    ]

    for col, dataset in enumerate(DATASETS):
        rows = rows_for_dataset(dataset)
        methods = [row[1] for row in rows]
        entities = rows[0][8]
        x = np.arange(len(methods))

        for row_idx, (_metric, value_i, low_i, high_i, ylabel) in enumerate(metric_specs):
            ax = axes[row_idx, col]
            values = [float(row[value_i]) for row in rows]
            lows = np.array([float(row[low_i]) for row in rows])
            highs = np.array([float(row[high_i]) for row in rows])
            yerr = np.vstack([np.array(values) - lows, highs - np.array(values)])
            colors = [METHOD_COLORS[m] for m in methods]

            bars = ax.bar(
                x,
                values,
                yerr=yerr,
                width=0.62,
                color=colors,
                edgecolor="white",
                linewidth=0.8,
                capsize=3,
                error_kw={"elinewidth": 1.1, "ecolor": "#2F3747", "capthick": 1.1},
            )
            set_clean_axis(ax)
            y_max = max(highs) * 1.20
            ax.set_ylim(0, y_max)
            ax.set_xticks(x, methods)
            ax.set_ylabel(ylabel if col == 0 else "")
            if row_idx == 0:
                ax.set_title(f"{dataset}  (n={entities:,})", fontweight="bold", color=TEXT)
            add_bar_labels(ax, bars, values, y_pad=y_max * 0.015)

    legend_methods = ["Naive", "ERM", "Sparse", "Auto"]
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=METHOD_COLORS[m], label=m)
        for m in legend_methods
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle(
        "Entity-bootstrap confidence intervals for Anchor-family errors",
        y=1.105,
        fontsize=12.5,
        fontweight="bold",
        color=TEXT,
    )
    fig.text(0.5, -0.015, "Bars show global scaled error; whiskers show 95% entity-bootstrap confidence intervals. Lower is better.", ha="center", fontsize=8.5, color="#4B5563")
    save_all(fig, output_dir, "anchor_bootstrap_ci", formats, dpi)
    plt.close(fig)


def ablation_rows_for_dataset(dataset: str) -> list[tuple[str, float, float]]:
    return [(condition, mse_delta, mae_delta) for ds, condition, mse_delta, mae_delta in ABLATION_ROWS if ds == dataset]


def plot_candidate_ablation(output_dir: Path, formats: list[str], dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2), constrained_layout=True)
    axes_flat = axes.ravel()

    for ax, dataset in zip(axes_flat, DATASETS):
        rows = ablation_rows_for_dataset(dataset)
        conditions = [row[0] for row in rows]
        mse = np.array([row[1] for row in rows], dtype=float)
        mae = np.array([row[2] for row in rows], dtype=float)
        y = np.arange(len(conditions))
        height = 0.34

        ax.axvline(0, color="#111827", lw=0.9, alpha=0.65)
        ax.barh(y + height / 2, mse, height=height, color=MSE_COLOR, label=r"$\Delta$MSE")
        ax.barh(y - height / 2, mae, height=height, color=MAE_COLOR, label=r"$\Delta$MAE")

        max_abs = max(0.01, float(np.max(np.abs(np.concatenate([mse, mae])))) * 1.35)
        ax.set_xlim(-max_abs, max_abs)
        ax.set_yticks(y, conditions)
        ax.invert_yaxis()
        ax.set_title(dataset, fontweight="bold", color=TEXT)
        ax.set_xlabel("Error change relative to full AutoAnchor")
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        ax.spines["left"].set_color("#D4D7DE")
        ax.spines["bottom"].set_color("#D4D7DE")

        for yi, value in zip(y + height / 2, mse):
            if abs(value) >= max_abs * 0.055:
                ha = "left" if value >= 0 else "right"
                offset = max_abs * (0.025 if value >= 0 else -0.025)
                ax.text(value + offset, yi, f"{value:+.3f}", va="center", ha=ha, fontsize=7.5, color=TEXT)
        for yi, value in zip(y - height / 2, mae):
            if abs(value) >= max_abs * 0.055:
                ha = "left" if value >= 0 else "right"
                offset = max_abs * (0.025 if value >= 0 else -0.025)
                ax.text(value + offset, yi, f"{value:+.3f}", va="center", ha=ha, fontsize=7.5, color=TEXT)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=MSE_COLOR, label=r"$\Delta$MSE"),
        plt.Rectangle((0, 0), 1, 1, color=MAE_COLOR, label=r"$\Delta$MAE"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.035))
    fig.suptitle(
        "Candidate-removal ablation: how much error increases when a component is removed",
        y=1.085,
        fontsize=12.5,
        fontweight="bold",
        color=TEXT,
    )
    fig.text(0.5, -0.012, "Positive bars mean the ablated variant is worse than full AutoAnchor; values near zero mean little sensitivity.", ha="center", fontsize=8.5, color="#4B5563")
    save_all(fig, output_dir, "anchor_candidate_ablation", formats, dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_output = project_root / "docs" / "paper" / "figures" / "generated"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--formats", nargs="+", default=["png"], choices=["pdf", "png", "svg"])
    parser.add_argument("--dpi", type=int, default=600, help="Raster output resolution for PNG files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    plot_bootstrap_ci(args.output_dir, args.formats, args.dpi)
    plot_candidate_ablation(args.output_dir, args.formats, args.dpi)
    print(f"Wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
