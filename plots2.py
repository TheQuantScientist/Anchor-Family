import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = "history_thinning_plots"
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
    "legend.fontsize": 16,
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "savefig.dpi": DPI,
    "figure.dpi": 150,
})

# ============================================================
# MODEL STYLES
# ============================================================

STYLES = {
    "AutoAnchor": {
        "color": "#111111",
        "marker": "o",
        "linestyle": "-",
        "linewidth": 2.8,
        "markersize": 8.5,
    },

    "APN": {
        "color": "#D28B7B",
        "marker": "s",
        "linestyle": "--",
        "linewidth": 2.2,
        "markersize": 7.5,
    },

    "GraFITi": {
        "color": "#1F3A56",
        "marker": "^",
        "linestyle": "-.",
        "linewidth": 2.2,
        "markersize": 8.0,
    },

    "tPatchGNN": {
        "color": "#7A8F72",
        "marker": "D",
        "linestyle": ":",
        "linewidth": 2.2,
        "markersize": 7.2,
    },
}

# ============================================================
# DATA
# ============================================================

# Original = 100% retained history
x_values = np.array([10, 25, 50, 75, 100])

data = {

    # --------------------------------------------------------
    # P12
    # --------------------------------------------------------
    "P12": {

        "AutoAnchor": {
            "mse": [
                0.5224,
                0.3957,
                0.3459,
                0.3177,
                0.3020,
            ],
            "std": None,
        },

        "APN": {
            "mse": [
                0.5433,
                0.4428,
                0.3642,
                0.3350,
                0.3093,
            ],
            "std": [
                0.0043,
                0.0033,
                0.0018,
                0.0017,
                0.0011,
            ],
        },

        "GraFITi": {
            "mse": [
                0.5279,
                0.4244,
                0.3575,
                0.3266,
                0.3075,
            ],
            "std": [
                0.0090,
                0.0057,
                0.0047,
                0.0037,
                0.0015,
            ],
        },

        "tPatchGNN": {
            "mse": [
                0.5399,
                0.4430,
                0.3673,
                0.3325,
                0.3133,
            ],
            "std": [
                0.0058,
                0.0050,
                0.0056,
                0.0041,
                0.0053,
            ],
        },
    },

    # --------------------------------------------------------
    # MIMIC
    # --------------------------------------------------------
    "MIMIC": {

        "AutoAnchor": {
            "mse": [
                0.7253,
                0.5724,
                0.4490,
                0.4433,
                0.3785,
            ],
            "std": None,
        },

        "APN": {
            "mse": [
                0.8529,
                0.6934,
                0.5782,
                0.5151,
                0.4292,
            ],
            "std": [
                0.0048,
                0.0086,
                0.0043,
                0.0047,
                0.0027,
            ],
        },

        "GraFITi": {
            "mse": [
                0.8375,
                0.7198,
                0.6093,
                0.5450,
                0.4359,
            ],
            "std": [
                0.0125,
                0.0225,
                0.0337,
                0.0308,
                0.0455,
            ],
        },

        "tPatchGNN": {
            "mse": [
                0.8526,
                0.6941,
                0.5777,
                0.5174,
                0.4431,
            ],
            "std": [
                0.0127,
                0.0065,
                0.0051,
                0.0074,
                0.0115,
            ],
        },
    },

    # --------------------------------------------------------
    # USHCN
    # --------------------------------------------------------
    "USHCN": {

        "AutoAnchor": {
            "mse": [
                0.4182,
                0.2640,
                0.1917,
                0.1777,
                0.1565,
            ],
            "std": None,
        },

        "APN": {
            "mse": [
                0.2673,
                0.1711,
                0.1602,
                0.1568,
                0.1590,
            ],
            "std": [
                0.0441,
                0.0224,
                0.0206,
                0.0204,
                0.0137,
            ],
        },

        "GraFITi": {
            "mse": [
                0.2351,
                0.2022,
                0.1622,
                0.1744,
                0.1691,
            ],
            "std": [
                0.0125,
                0.0144,
                0.0013,
                0.0095,
                0.0093,
            ],
        },

        "tPatchGNN": {
            "mse": [
                0.2527,
                0.2177,
                0.1688,
                0.1437,
                0.1885,
            ],
            "std": [
                0.0388,
                0.0809,
                0.0252,
                0.0099,
                0.0403,
            ],
        },
    },

    # --------------------------------------------------------
    # HumanActivity
    # --------------------------------------------------------
    "HumanActivity": {

        "AutoAnchor": {
            "mse": [
                0.0978,
                0.0698,
                0.0519,
                0.0447,
                0.0420,
            ],
            "std": None,
        },

        "APN": {
            "mse": [
                0.2740,
                0.0682,
                0.0490,
                0.0445,
                0.0421,
            ],
            "std": [
                0.0176,
                0.0006,
                0.0002,
                0.0001,
                0.0001,
            ],
        },

        "GraFITi": {
            "mse": [
                1.1108,
                0.3313,
                0.0836,
                0.0501,
                0.0437,
            ],
            "std": [
                0.1831,
                0.0730,
                0.0110,
                0.0006,
                0.0005,
            ],
        },

        "tPatchGNN": {
            "mse": [
                1.1527,
                0.3848,
                0.1083,
                0.0598,
                0.0443,
            ],
            "std": [
                0.0758,
                0.0864,
                0.0191,
                0.0044,
                0.0009,
            ],
        },
    },
}

# ============================================================
# PLOTTING FUNCTION
# ============================================================

def plot_dataset(dataset_name, dataset):

    fig, ax = plt.subplots(
        figsize=(7.5, 5.5)
    )

    # --------------------------------------------------------
    # Plot each model
    # --------------------------------------------------------

    for method in [
        "AutoAnchor",
        "APN",
        "GraFITi",
        "tPatchGNN",
    ]:

        values = np.array(dataset[method]["mse"])
        std = dataset[method]["std"]
        style = STYLES[method]

        # AutoAnchor is deterministic
        if std is None:

            ax.plot(
                x_values,
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

        # Neural models: mean +/- std
        else:

            ax.errorbar(
                x_values,
                values,
                yerr=np.array(std),
                label=method,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                markersize=style["markersize"],
                capsize=3.5,
                capthick=1.1,
                elinewidth=1.1,
                markeredgewidth=0.8,
                markeredgecolor="white",
                zorder=4,
            )

    # ========================================================
    # AXES
    # ========================================================

    ax.set_xlabel(
        "Retained History (%)",
        labelpad=10,
    )

    ax.set_ylabel(
        "MSE",
        labelpad=10,
    )

    ax.set_xticks(
        [10, 25, 50, 75, 100]
    )

    ax.set_xticklabels(
        ["10", "25", "50", "75", "100"]
    )

    ax.set_xlim(
        5,
        105,
    )

    # ========================================================
    # GRID
    # ========================================================

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.7,
        alpha=0.25,
    )

    ax.set_axisbelow(True)

    # ========================================================
    # CLEAN SPINES
    # ========================================================

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Keep reasonable number of y ticks
    ax.yaxis.set_major_locator(
        MaxNLocator(nbins=6)
    )

    # ========================================================
    # LEGEND
    # ========================================================

    ax.legend(
        loc="upper right",
        frameon=False,
        handlelength=2.6,
        handletextpad=0.6,
        borderaxespad=0.3,
    )

    # No dataset name/title inside figure

    fig.tight_layout()

    # ========================================================
    # SAVE
    # ========================================================

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{dataset_name}_history_thinning.{FILE_FORMAT}",
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
# GENERATE ALL 4 FIGURES
# ============================================================

for dataset_name, dataset_values in data.items():
    plot_dataset(
        dataset_name,
        dataset_values,
    )

print("\nDone. Generated 4 history-thinning figures at 600 DPI.")