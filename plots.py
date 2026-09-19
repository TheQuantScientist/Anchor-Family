import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = "lookback_horizon_plots"
DPI = 600
FILE_FORMAT = "png"

os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "savefig.dpi": DPI,
    "figure.dpi": 150,
})

STYLES = {
    "AutoAnchor": {
        "color": "#111111",
        "marker": "o",
        "linestyle": "-",
        "linewidth": 2.8,
        "markersize": 8.0,
    },
    "APN": {
        "color": "#D28B7B",
        "marker": "s",
        "linestyle": "--",
        "linewidth": 2.2,
        "markersize": 7.0,
    },
    "GraFITi": {
        "color": "#1F3A56",
        "marker": "^",
        "linestyle": "-.",
        "linewidth": 2.2,
        "markersize": 7.5,
    },
    "tPatchGNN": {
        "color": "#7A8F72",
        "marker": "D",
        "linestyle": ":",
        "linewidth": 2.2,
        "markersize": 6.8,
    },
}

# ============================================================
# DATA
# ============================================================

data = {

    "P12": {
        "labels": ["12 / 1", "24 / 3", "36 / 3", "36 / 6", "48 / 6"],
        "standard_idx": 2,

        "AutoAnchor": {
            "mse": [0.2767, 0.3063, 0.3020, 0.3286, 0.4941],
            "mae": [0.3426, 0.3619, 0.3585, 0.3825, 0.4430],
            "mse_std": None,
            "mae_std": None,
        },

        "APN": {
            "mse": [0.3249, 0.3213, 0.3093, 0.3345, 0.6242],
            "mae": [0.3744, 0.3711, 0.3650, 0.3881, 0.5184],
            "mse_std": [0.0107, 0.0007, 0.0011, 0.0011, 0.0502],
            "mae_std": [0.0147, 0.0012, 0.0026, 0.0016, 0.0378],
        },

        "GraFITi": {
            "mse": [0.3089, 0.3170, 0.3075, 0.3239, 0.7612],
            "mae": [0.3568, 0.3706, 0.3637, 0.3813, 0.6071],
            "mse_std": [0.0018, 0.0001, 0.0015, 0.0020, 0.1443],
            "mae_std": [0.0015, 0.0037, 0.0036, 0.0007, 0.0917],
        },

        "tPatchGNN": {
            "mse": [0.3143, 0.3229, 0.3133, 0.3280, 0.6102],
            "mae": [0.3645, 0.3747, 0.3697, 0.3860, 0.5048],
            "mse_std": [0.0013, 0.0025, 0.0053, 0.0049, 0.0292],
            "mae_std": [0.0014, 0.0058, 0.0049, 0.0051, 0.0169],
        },
    },

    "MIMIC": {
        "labels": [
            "12 / 1",
            "24 / 3",
            "36 / 3",
            "48 / 6",
            "72 / 3",
            "72 / 6",
        ],
        "standard_idx": 4,

        "AutoAnchor": {
            "mse": [0.4586, 0.5071, 0.3660, 0.3855, 0.3785, 0.3931],
            "mae": [0.3895, 0.4111, 0.3717, 0.3665, 0.3657, 0.3764],
            "mse_std": None,
            "mae_std": None,
        },

        "APN": {
            "mse": [0.9828, 0.7252, 0.5712, 0.4596, 0.4292, 0.4837],
            "mae": [0.5982, 0.5558, 0.4833, 0.4153, 0.4016, 0.4278],
            "mse_std": [0.0026, 0.0028, 0.0092, 0.0041, 0.0027, 0.0107],
            "mae_std": [0.0010, 0.0019, 0.0104, 0.0028, 0.0016, 0.0042],
        },

        "GraFITi": {
            "mse": [0.9915, 0.7319, 0.5963, 0.4493, 0.4359, 0.5357],
            "mae": [0.6086, 0.5592, 0.4888, 0.4141, 0.4142, 0.4711],
            "mse_std": [0.0668, 0.0073, 0.0024, 0.0484, 0.0455, 0.0651],
            "mae_std": [0.0452, 0.0104, 0.0052, 0.0328, 0.0297, 0.0704],
        },

        "tPatchGNN": {
            "mse": [1.1484, 0.7210, 0.6605, 0.4293, 0.4431, 0.5145],
            "mae": [0.5791, 0.5517, 0.5128, 0.3912, 0.4077, 0.4311],
            "mse_std": [0.0093, 0.0123, 0.0154, 0.0051, 0.0115, 0.0357],
            "mae_std": [0.0078, 0.0044, 0.0047, 0.0024, 0.0088, 0.0196],
        },
    },

    "USHCN": {
        "labels": ["50 / 1", "100 / 3", "150 / 3", "150 / 7"],
        "standard_idx": 2,

        "AutoAnchor": {
            "mse": [0.7173, 1.1591, 0.1565, 0.4254],
            "mae": [0.3889, 0.5757, 0.2187, 0.2745],
            "mse_std": None,
            "mae_std": None,
        },

        "APN": {
            "mse": [0.5989, 0.6457, 0.1590, 0.4103],
            "mae": [0.4020, 0.4262, 0.2611, 0.3266],
            "mse_std": [0.0118, 0.0027, 0.0137, 0.0020],
            "mae_std": [0.0065, 0.0059, 0.0167, 0.0160],
        },

        "GraFITi": {
            "mse": [0.5805, 0.6576, 0.1691, 0.4378],
            "mae": [0.3976, 0.4394, 0.2777, 0.3539],
            "mse_std": [0.0256, 0.0352, 0.0093, 0.0055],
            "mae_std": [0.0489, 0.0203, 0.0248, 0.0262],
        },

        "tPatchGNN": {
            "mse": [0.5504, 0.6109, 0.1885, 0.4067],
            "mae": [0.4045, 0.4338, 0.3084, 0.3437],
            "mse_std": [0.0061, 0.0017, 0.0403, 0.0084],
            "mae_std": [0.0071, 0.0014, 0.0479, 0.0093],
        },
    },

    "HumanActivity": {
        "labels": [
            "1000 / 100",
            "2000 / 200",
            "3000 / 300",
            "3000 / 600",
        ],
        "standard_idx": 2,

        "AutoAnchor": {
            "mse": [0.0491, 0.0491, 0.0420, 0.0486],
            "mae": [0.1098, 0.1173, 0.1148, 0.1240],
            "mse_std": None,
            "mae_std": None,
        },

        "APN": {
            "mse": [0.0545, 0.0483, 0.0421, 0.0483],
            "mae": [0.1139, 0.1159, 0.1159, 0.1265],
            "mse_std": [0.0001, 0.0005, 0.0001, 0.0002],
            "mae_std": [0.0011, 0.0015, 0.0006, 0.0010],
        },

        "GraFITi": {
            "mse": [0.0603, 0.0514, 0.0437, 0.0506],
            "mae": [0.1304, 0.1290, 0.1221, 0.1369],
            "mse_std": [0.0001, 0.0029, 0.0005, 0.0007],
            "mae_std": [0.0007, 0.0074, 0.0017, 0.0018],
        },

        "tPatchGNN": {
            "mse": [0.0620, 0.0531, 0.0443, 0.0524],
            "mae": [0.1322, 0.1311, 0.1247, 0.1398],
            "mse_std": [0.0031, 0.0018, 0.0009, 0.0014],
            "mae_std": [0.0023, 0.0037, 0.0031, 0.0034],
        },
    },
}

# ============================================================
# PLOTTING
# ============================================================

def plot_dataset(dataset_name, dataset):
    labels = dataset["labels"]
    standard_idx = dataset["standard_idx"]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 5.0),
        constrained_layout=False,
    )

    metrics = [
        ("mse", "MSE"),
        ("mae", "MAE"),
    ]

    for ax, (metric, ylabel) in zip(axes, metrics):

        # Highlight standard benchmark setting
        ax.axvspan(
            standard_idx - 0.34,
            standard_idx + 0.34,
            color="0.5",
            alpha=0.10,
            linewidth=0,
            zorder=0,
        )

        for method in ["AutoAnchor", "APN", "GraFITi", "tPatchGNN"]:

            values = np.array(dataset[method][metric])
            std = dataset[method][f"{metric}_std"]
            style = STYLES[method]

            if std is None:
                ax.plot(
                    x,
                    values,
                    label=method,
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    markersize=style["markersize"],
                    markeredgewidth=0.9,
                    markeredgecolor="white",
                    zorder=5,
                )
            else:
                ax.errorbar(
                    x,
                    values,
                    yerr=np.array(std),
                    label=method,
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    markersize=style["markersize"],
                    capsize=3.2,
                    capthick=1.1,
                    elinewidth=1.1,
                    markeredgewidth=0.8,
                    markeredgecolor="white",
                    zorder=4,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(labels)

        ax.set_xlabel("Lookback / Horizon", labelpad=10)
        ax.set_ylabel(ylabel, labelpad=10)

        ax.grid(
            axis="y",
            linestyle="--",
            linewidth=0.7,
            alpha=0.25,
        )

        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.set_xlim(-0.35, len(labels) - 0.65)

        # Bold standard benchmark tick
        ax.get_xticklabels()[standard_idx].set_fontweight("bold")

        # Standard-window label
        ymin, ymax = ax.get_ylim()
        yrange = ymax - ymin

        ax.text(
            standard_idx,
            ymax - 0.025 * yrange,
            "Standard",
            ha="center",
            va="top",
            fontsize=16,
            fontstyle="italic",
            color="0.35",
        )

    # Shared legend
    handles, legend_labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
        columnspacing=2.0,
        handlelength=2.6,
        handletextpad=0.6,
    )

    # No dataset title
    fig.subplots_adjust(
        top=0.80,
        bottom=0.19,
        left=0.08,
        right=0.985,
        wspace=0.27,
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{dataset_name}_lookback_horizon.{FILE_FORMAT}",
    )

    fig.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


# ============================================================
# GENERATE 4 FIGURES
# ============================================================

for dataset_name, dataset_values in data.items():
    plot_dataset(dataset_name, dataset_values)

print("\nDone. Generated 4 figures at 600 DPI.")