#!/usr/bin/env python3
"""Draw the Anchor architecture as vector PDF/SVG and a high-resolution PNG.

Run from any directory with a Python environment containing numpy/matplotlib:
    python docs/figures/draw_anchor_family.py

No LaTeX installation or benchmark data is needed. Coordinates and type sizes
are in points; the output is designed to remain readable at 5.5 inches wide.
The miniature forecasts use the actual, pure component functions in the repo.
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/chronolm-figure-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODEL = ROOT / "src/chronolm/experiments/anchor_baseline.py"

INK = "#202B3B"
MUTED = "#536273"
LINE = "#CCD4DC"
BLUE = "#286A9A"
BLUE_BG = "#F0F6FA"
TEAL = "#247C72"
TEAL_BG = "#EEF7F4"
PURPLE = "#73528F"
PURPLE_BG = "#F5F1F8"
ORANGE = "#AD5B29"
ORANGE_BG = "#FBF3EC"
GRAY_BG = "#F7F8FA"
W, H = 504, 480

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "svg.hashsalt": "chronolm-anchor-family",
    "axes.unicode_minus": False,
})


def load_pure_model_functions():
    """Read only the candidate builder and pure forecasts, without APN imports."""
    names = {
        "PHASE_COMPONENTS", "AnchorCandidate", "weight_tag",
        "build_auto_anchor_candidates", "most_common_rounded",
        "phase_anchor_scaled", "recent_trend_anchor_scaled",
        "component_forecast_scaled",
    }
    tree = ast.parse(MODEL.read_text())
    nodes = [
        node for node in tree.body
        if getattr(node, "name", None) in names
        or isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)
    ]
    namespace = {"np": np, "Counter": Counter, "dataclass": dataclass}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(MODEL), "exec"), namespace)
    return namespace


class Drawing:
    def __init__(self, width=W, height=H):
        self.width, self.height = width, height
        self.fig = plt.figure(figsize=(width / 72, height / 72), facecolor="white")
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set(xlim=(0, width), ylim=(height, 0))
        self.ax.set_axis_off()
        self.labels = []

    def text(self, x, y, value, size=10, color=INK, weight="normal",
             align="left", bounds=None, **kwargs):
        artist = self.ax.text(x, y, value, fontsize=size, color=color,
                              fontweight=weight, ha=align, va="center",
                              linespacing=1.35, zorder=5, **kwargs)
        self.labels.append((artist, bounds))
        return artist

    def box(self, x, y, width, height, fill="white", edge=LINE, lw=.8,
            radius=4):
        self.ax.add_patch(FancyBboxPatch(
            (x, y), width, height,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=fill, edgecolor=edge, linewidth=lw, zorder=1))
        return (x, y, x + width, y + height)

    def line(self, points, color=LINE, lw=.8, dashed=False, zorder=2):
        xs, ys = zip(*points)
        self.ax.add_line(Line2D(xs, ys, color=color, linewidth=lw,
                              linestyle=(0, (3, 2)) if dashed else "-",
                              solid_capstyle="round", zorder=zorder))

    def arrow(self, start, end, color=INK, dashed=False, lw=1.0):
        self.ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=8,
            linewidth=lw, color=color, shrinkA=0, shrinkB=0,
            linestyle=(0, (3, 2)) if dashed else "-", zorder=3))

    def panel_title(self, letter, title, y):
        self.box(8, y - 7, 14, 14, fill=INK, edge=INK, radius=2)
        self.text(15, y, letter, size=9.3, color="white", weight="bold", align="center")
        self.text(29, y, title, size=11.3, weight="bold")

    def save(self, stem):
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        errors = []
        # Check every label against the canvas and, where supplied, its box.
        for artist, bounds in self.labels:
            bbox = artist.get_window_extent(renderer).transformed(self.ax.transData.inverted())
            left, right = sorted((bbox.x0, bbox.x1))
            top, bottom = sorted((bbox.y0, bbox.y1))
            allowed = bounds or (0, 0, self.width, self.height)
            if left < allowed[0] - .3 or top < allowed[1] - .3 or right > allowed[2] + .3 or bottom > allowed[3] + .3:
                errors.append(f"{artist.get_text()!r}: {(left, top, right, bottom)} outside {allowed}")
        if errors:
            raise RuntimeError("Text overflow:\n" + "\n".join(errors))
        metadata = {"Title": "The Anchor family: calibration and forecasting",
                    "Author": "", "Creator": "ChronoLM vector figure generator",
                    "CreationDate": None, "ModDate": None}
        self.fig.savefig(HERE / f"{stem}.pdf", metadata=metadata)
        self.fig.savefig(HERE / f"{stem}.svg", metadata={"Date": None})
        self.fig.savefig(HERE / f"{stem}.png", dpi=300)
        # A 5.5-inch-wide proof at 150 dpi gives a useful final-size check.
        self.fig.savefig(HERE / f"{stem}_print_proof.png", dpi=150 * 5.5 / (self.width / 72))
        print(f"Rendered {stem}: {self.width / 72:.2f} x {self.height / 72:.2f} in; {len(self.labels)} labels fit.")
        plt.close(self.fig)


def miniature(d, x, y, width, height, component, model, color=TEAL):
    """An illustrative history (gray) and actual component forecast (color)."""
    times = np.array([0, .8, 1.6, 2.1, 3.4, 4.8, 5.2, 6.0, 7.3, 8.1])
    values = np.array([.30, .34, .72, .64, .35, .38, .36, .64, .68, .57])
    if component == "mode":
        values = np.array([.32, .32, .32, .7, .32, .32, .32, .58, .32, .32])
    query = np.array([8.7, 9.6, 10.4])
    pred = model["component_forecast_scaled"](
        component, times.tolist(), values.tolist(), query.tolist(), lookback_span=9.0)
    tx = lambda t: x + np.asarray(t) / 11 * width
    ty = lambda v: y + height - np.clip(np.asarray(v), 0, 1) * height
    d.line([(x, y + height), (x + width, y + height)], color=LINE, lw=.45)
    d.line([(float(tx(8.4)), y), (float(tx(8.4)), y + height)], lw=.55, dashed=True)
    d.ax.plot(tx(times), ty(values), color="#A1AEB9", linewidth=.65,
              marker="o", markersize=1.4, markeredgewidth=0, zorder=3)
    d.ax.plot(tx(query), ty(pred), color=color, linewidth=1.2,
              marker="o", markersize=2.1, markeredgewidth=0, zorder=4)


def draw_main(model):
    d = Drawing()
    d.panel_title("a", "AutoAnchor: calibrate once per variable", 12)

    # Three explicit inputs make the source of each decision unambiguous.
    b = d.box(8, 29, 143, 25, fill=GRAY_BG)
    d.text(79.5, 41.5, "Histories + query times", 9.6, align="center", bounds=b)
    b = d.box(181, 29, 135, 25, fill=PURPLE_BG, edge="#CFBEDD")
    d.text(248.5, 41.5, "Train + validation targets", 9.1, align="center", bounds=b)
    b = d.box(344, 29, 152, 25, fill=GRAY_BG)
    d.text(420, 41.5, "Histories + times + windows", 9.0, align="center", bounds=b)
    d.arrow((79.5, 54), (79.5, 67), TEAL)
    d.arrow((248.5, 54), (248.5, 78), PURPLE)
    d.arrow((420, 54), (420, 67), ORANGE)

    bank = d.box(8, 67, 143, 186, fill=TEAL_BG, edge="#83B4A9")
    d.text(79.5, 82, "Shared anchor library", 10.2, weight="bold", align="center", bounds=bank)
    d.text(79.5, 97, r"$f_a(\mathcal{H}_v, t^*)$", 10, color=TEAL, align="center", bounds=bank)
    tiles = [
        ("Last value", "last"), ("Means / trim", "mean5"),
        ("EMA", "ema03"), ("Mode", "mode"),
        ("Local trend", "trend8"), ("Phase-nearest", "phase033k5"),
    ]
    for index, (label, component) in enumerate(tiles):
        col, row = index % 2, index // 2
        x, y = 16 + col * 68, 109 + row * 39
        d.text(x + 29, y, label, 8.8, align="center", bounds=bank)
        miniature(d, x, y + 7, 58, 21, component, model)
    d.line([(17, 227), (142, 227)], color="#B8D5CD", lw=.7)
    d.text(79.5, 237, "+ fixed convex mixtures", 9.2, color=TEAL, align="center", bounds=bank)
    d.text(79.5, 247, f"{len(model['build_auto_anchor_candidates']())} candidates", 8.7, color=MUTED, align="center", bounds=bank)

    erm = d.box(181, 78, 135, 82, fill=PURPLE_BG, edge=PURPLE)
    d.text(248.5, 93, "ERMAnchor", 11.2, color=PURPLE, weight="bold", align="center", bounds=erm)
    d.text(248.5, 115, r"$(a_v^{\mathrm{E}},\beta_v^{\mathrm{E}})=\arg\min_{a,\beta}\ R_v(a,\beta)$", 9.0, align="center", bounds=erm)
    d.text(248.5, 138, "Pooled masked scaled MSE", 8.9, align="center", bounds=erm)
    d.text(248.5, 151, r"$\beta$: identity or fitted scalar", 8.9, color=MUTED, align="center", bounds=erm)
    d.arrow((151, 119), (181, 119), TEAL)

    prior = d.box(344, 67, 152, 186, fill=ORANGE_BG, edge="#CB9A77")
    d.text(420, 83, "History-structure prior", 10.0, color=ORANGE, weight="bold", align="center", bounds=prior)
    d.text(420, 99, "Ordered rules; history values only", 8.7, color=MUTED, align="center", bounds=prior)
    rows = [
        (119, "1", "Repeated-value structure", "mode or last / phase mixture"),
        (163, "2", "Long-window structure", "short-horizon EMA from K / L"),
        (207, "3", "Phase / trend agreement", "phase / trend / last mixtures"),
    ]
    for y, number, title, detail in rows:
        d.box(352, y - 6, 12, 12, fill=ORANGE, edge=ORANGE, radius=2)
        d.text(358, y, number, 8.1, color="white", weight="bold", align="center")
        d.text(369, y, title, 8.6, weight="bold", bounds=prior)
        d.text(352, y + 15, detail, 8.9, bounds=prior)
        if number != "3":
            d.line([(352, y + 28), (488, y + 28)], color="#E6CDBB", lw=.6)
    d.text(420, 242, r"First matching rule $\longrightarrow\ (a_v^{\mathrm{P}},1)$", 9.1, color=ORANGE, align="center", bounds=prior)

    auto = d.box(181, 191, 135, 62, fill=BLUE_BG, edge=BLUE, lw=1.35)
    d.text(248.5, 205, "AutoAnchor", 11.2, color=BLUE, weight="bold", align="center", bounds=auto)
    d.text(248.5, 224, r"Use $(a_v^{\mathrm{P}},1)$ if a rule fires;", 9.2, align="center", bounds=auto)
    d.text(248.5, 241, r"otherwise keep $(a_v^{\mathrm{E}},\beta_v^{\mathrm{E}})$.", 9.2, align="center", bounds=auto)
    d.arrow((248.5, 160), (248.5, 191), PURPLE)
    d.text(255, 176, "ERM choice", 8.8, color=PURPLE)
    d.arrow((344, 224), (316, 224), ORANGE)

    # Inference receives a frozen choice, not a newly calibrated selector.
    d.arrow((248.5, 253), (248.5, 314), BLUE, dashed=True)
    d.text(257, 273, r"Freeze $(a_v,\beta_v)$ before test evaluation", 9.3, color=BLUE)
    d.panel_title("b", "Forecast each new history", 290)
    b = d.box(8, 314, 107, 58, fill=GRAY_BG)
    d.text(61.5, 328, "Irregular history", 9.7, weight="bold", align="center", bounds=b)
    d.text(61.5, 346, r"$\mathcal{H}_{b,v}=\{(t_j,x_j)\}$", 10, align="center", bounds=b)
    d.text(61.5, 362, r"Query times $t^*_{b,k,v}$", 9.2, color=MUTED, align="center", bounds=b)

    b = d.box(141, 314, 126, 58, fill=TEAL_BG, edge="#83B4A9")
    d.text(204, 328, "Evaluate anchor", 10.0, weight="bold", align="center", bounds=b)
    d.text(204, 348, r"$z=f_{a_v}(\mathcal{H}_{b,v},t^*_{b,k,v})$", 10.0, align="center", bounds=b)
    d.text(204, 363, "Values in benchmark scale", 8.8, color=MUTED, align="center", bounds=b)
    d.arrow((115, 343), (141, 343), INK)

    b = d.box(293, 314, 124, 58, fill=BLUE_BG, edge="#A0BED3")
    d.text(355, 328, "Scale and bound", 9.8, weight="bold", align="center", bounds=b)
    d.text(355, 348, r"$\beta_v z\ \rightarrow$ inverse scaling", 9.1, align="center", bounds=b)
    d.text(355, 363, "Recent-history clipping", 9.2, color=MUTED, align="center", bounds=b)
    d.arrow((267, 343), (293, 343), INK)
    d.arrow((417, 343), (438, 343), BLUE)
    d.text(469, 322, "Forecast", 9.7, weight="bold", align="center")
    d.ax.plot([447, 466, 489], [347, 340, 344], color=BLUE, linewidth=1.2,
              marker="o", markersize=3, zorder=3)
    d.text(468, 363, r"$\hat{y}_{b,k,v}$", 11, color=BLUE, align="center")

    d.line([(8, 389), (496, 389)], lw=.8)
    d.panel_title("c", "Five variants share the forecast pipeline", 405)
    family = [
        ("NaiveAnchor", "Last value", r"$\beta_v=1$", MUTED, GRAY_BG),
        ("ExpoAnchor", "Fixed EMA", r"$\beta_v=1$", TEAL, TEAL_BG),
        ("SparseAnchor", "Sparse rules", r"$\beta_v=1$", ORANGE, ORANGE_BG),
        ("ERMAnchor", "Minimum-risk choice", r"Select $a_v,\beta_v$", PURPLE, PURPLE_BG),
        ("AutoAnchor", "ERM + ordered prior", r"Select $a_v,\beta_v$", BLUE, BLUE_BG),
    ]
    for i, (name, description, tail, color, fill) in enumerate(family):
        x = 8 + i * 99
        b = d.box(x, 422, 92, 51, fill=fill, edge=color, lw=.8)
        d.text(x + 46, 435, name, 10.0, color=color, weight="bold", align="center", bounds=b)
        d.text(x + 46, 450, description, 8.7, align="center", bounds=b)
        d.text(x + 46, 464, tail, 9.0, color=MUTED, align="center", bounds=b)
    d.save("anchor_family_architecture")


def write_audit(model):
    candidates = model["build_auto_anchor_candidates"]()
    tree = ast.parse(MODEL.read_text())
    names = [
        "build_auto_anchor_candidates", "component_forecast_scaled",
        "variable_history_statistics", "horizon_ema_method",
        "choose_structural_prior_method", "choose_sparse_anchor_method",
        "fit_beta", "candidate_beta_options", "candidate_forecast_scaled",
        "score_candidate_on_dataset", "anchor_predictions_scaled", "select_erm_spec",
        "build_family_spec", "calibrate_anchor_family", "build_forecast_anchors",
        "recent_history_bounds", "run",
    ]
    functions = {
        node.name: {"start": node.lineno, "end": node.end_lineno}
        for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    }
    audit = {
        "source": str(MODEL.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        "candidate_count": len(candidates),
        "candidate_arity_counts": dict(sorted(Counter(len(c.components) for c in candidates).items())),
        "source_functions": functions,
        "nominal_width_inches": W / 72,
        "intended_paper_width_inches": 5.5,
        "minimum_main_text_points_at_paper_width": round(8.6 * 5.5 / (W / 72), 2),
        "note": "Miniature curves are illustrative; predictions use repository component functions.",
    }
    (HERE / "anchor_family_source_map.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    model_functions = load_pure_model_functions()
    draw_main(model_functions)
    write_audit(model_functions)
