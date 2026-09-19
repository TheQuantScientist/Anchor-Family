"""Create paper-ready cross-baseline defense-study tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_ORDER = ["AutoAnchor", "APN", "GraFITi", "tPatchGNN"]
DATASET_ORDER = ["P12", "MIMIC", "USHCN", "HumanActivity"]
PERTURBATION_ORDER = [
    "original",
    "shuffle_timestamps",
    "swap_halves",
    "drop_early_history",
    "keep_last_obs",
    "keep_time_gaps",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cross-baseline study tables from collected metrics.")
    parser.add_argument("--input", type=Path, default=Path("cross_baseline_results/cross_baseline_metrics.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("cross_baseline_results/ablation"))
    return parser.parse_args()


def order_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["Dataset"] = pd.Categorical(out["Dataset"], DATASET_ORDER, ordered=True)
    out["Model"] = pd.Categorical(out["Model"], MODEL_ORDER, ordered=True)
    return out.sort_values(["Dataset", "Seq_len", "Pred_len", "Model", "Condition"])


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    grouped = (
        frame.groupby(
            ["Dataset", "Model", "Seq_len", "Pred_len", "History_perturbation", "History_keep_fraction", "Condition"],
            dropna=False,
            observed=True,
        )
        .agg(
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            MSE_mean=("MSE", "mean"),
            MSE_std=("MSE", "std"),
            Runs=("MSE", "count"),
        )
        .reset_index()
    )
    grouped["MAE_std"] = grouped["MAE_std"].fillna(0.0)
    grouped["MSE_std"] = grouped["MSE_std"].fillna(0.0)
    return order_frame(grouped)


def fmt_metric(mean: float, std: float, runs: int, model: str) -> str:
    if not np.isfinite(mean):
        return "--"
    if model == "AutoAnchor" or runs <= 1:
        return f"{mean:.4f}"
    return f"{mean:.4f} +/- {std:.4f}"


def fmt_tex_metric(mean: float, std: float, runs: int, model: str) -> str:
    if not np.isfinite(mean):
        return "--"
    if model == "AutoAnchor" or runs <= 1:
        return f"{mean:.4f}"
    return f"{mean:.4f}$\\pm${std:.4f}"


def write_md(path: Path, title: str, frame: pd.DataFrame, columns: list[str]) -> None:
    lines = [f"# {title}", ""]
    if frame.empty:
        lines.append("No rows collected yet.")
        path.write_text("\n".join(lines) + "\n")
        return
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join(["---"] * len(columns)) + "|")
    for _, row in frame.iterrows():
        values = [str(row[col]) for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n")


def condition_label(row: pd.Series) -> str:
    if row["History_perturbation"] != "original":
        return str(row["History_perturbation"]).replace("_", " ")
    keep = float(row["History_keep_fraction"])
    if keep < 1.0:
        return f"keep {int(round(keep * 100))}%"
    return "original"


def build_long_table(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        for column in ["Window", "ConditionLabel", "MSE", "MAE"]:
            out[column] = []
        return out
    out["Window"] = out.apply(lambda r: f"{int(r['Seq_len'])} / {int(r['Pred_len'])}", axis=1)
    out["ConditionLabel"] = out.apply(condition_label, axis=1)
    out["MSE"] = out.apply(lambda r: fmt_metric(r["MSE_mean"], r["MSE_std"], int(r["Runs"]), str(r["Model"])), axis=1)
    out["MAE"] = out.apply(lambda r: fmt_metric(r["MAE_mean"], r["MAE_std"], int(r["Runs"]), str(r["Model"])), axis=1)
    return out


def write_long_tex(path: Path, caption: str, label: str, frame: pd.DataFrame) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\caption{" + caption + r"}",
        r"\label{" + label + r"}",
        r"\centering",
        r"\scriptsize",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{lllrrr}",
        r"\toprule",
        r"Dataset & Window/condition & Model & MSE & MAE & Runs \\",
        r"\midrule",
    ]
    if frame.empty:
        lines.append(r"No rows & -- & -- & -- & -- & -- \\")
    else:
        for _, row in frame.iterrows():
            dataset = str(row["Dataset"])
            window = str(row.get("Window", row.get("ConditionLabel", ""))).replace("_", " ")
            if "ConditionLabel" in row and row["ConditionLabel"] != "original":
                window = f"{row['Window']}; {row['ConditionLabel']}"
            model = str(row["Model"])
            mse = fmt_tex_metric(row["MSE_mean"], row["MSE_std"], int(row["Runs"]), model)
            mae = fmt_tex_metric(row["MAE_mean"], row["MAE_std"], int(row["Runs"]), model)
            lines.append(f"{dataset} & {window} & {model} & {mse} & {mae} & {int(row['Runs'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    path.write_text("\n".join(lines) + "\n")


def add_original_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        out = frame.copy()
        for column in ["Base_MSE", "Base_MAE", "Delta_MSE", "Delta_MAE"]:
            out[column] = []
        return out
    out = frame.copy()
    base = out[out["Condition"] == "original"][["Dataset", "Model", "Seq_len", "Pred_len", "MSE_mean", "MAE_mean"]]
    base = base.rename(columns={"MSE_mean": "Base_MSE", "MAE_mean": "Base_MAE"})
    out = out.merge(base, on=["Dataset", "Model", "Seq_len", "Pred_len"], how="left")
    out["Delta_MSE"] = out["MSE_mean"] - out["Base_MSE"]
    out["Delta_MAE"] = out["MAE_mean"] - out["Base_MAE"]
    return out


def write_outputs(frame: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    agg = aggregate(frame)
    agg.to_csv(output_dir / "cross_baseline_aggregate.csv", index=False)

    lookback = agg[(agg["History_perturbation"] == "original") & (agg["History_keep_fraction"].astype(float) >= 1.0)]
    lookback_long = build_long_table(lookback)
    lookback_long[["Dataset", "Window", "Model", "MSE", "MAE", "Runs"]].to_csv(output_dir / "cross_baseline_lookback_horizon.csv", index=False)
    write_md(output_dir / "cross_baseline_lookback_horizon.md", "Cross-Baseline Lookback/Horizon Sweep", lookback_long[["Dataset", "Window", "Model", "MSE", "MAE", "Runs"]], ["Dataset", "Window", "Model", "MSE", "MAE", "Runs"])
    write_long_tex(output_dir / "cross_baseline_lookback_horizon.tex", "Cross-baseline lookback/horizon sweep. Neural baselines are mean $\\pm$ std over training iterations; AutoAnchor is deterministic.", "tab:cross-baseline-lookback", lookback_long)

    temporal = agg[agg["History_perturbation"].isin(PERTURBATION_ORDER)]
    temporal = temporal[temporal["History_keep_fraction"].astype(float) >= 1.0]
    if not temporal.empty:
        temporal_keys = ["Dataset", "Model", "Seq_len", "Pred_len"]
        has_perturb = temporal[temporal["History_perturbation"] != "original"][temporal_keys].drop_duplicates()
        temporal = temporal.merge(has_perturb, on=temporal_keys, how="inner")
    temporal = add_original_deltas(temporal)
    temporal_long = build_long_table(temporal)
    temporal_long["Delta MSE"] = temporal_long["Delta_MSE"].map(lambda x: "--" if pd.isna(x) else f"{x:+.4f}")
    temporal_long[["Dataset", "Window", "ConditionLabel", "Model", "MSE", "Delta MSE", "MAE", "Runs"]].to_csv(output_dir / "cross_baseline_temporal.csv", index=False)
    write_md(output_dir / "cross_baseline_temporal.md", "Cross-Baseline Temporal Perturbations", temporal_long[["Dataset", "Window", "ConditionLabel", "Model", "MSE", "Delta MSE", "MAE", "Runs"]], ["Dataset", "Window", "ConditionLabel", "Model", "MSE", "Delta MSE", "MAE", "Runs"])
    write_long_tex(output_dir / "cross_baseline_temporal.tex", "Cross-baseline temporal perturbations. Deltas are relative to each model's original-history evaluation at the same window.", "tab:cross-baseline-temporal", temporal_long)

    sparsity = agg[agg["History_perturbation"] == "original"]
    sparsity = sparsity[sparsity["History_keep_fraction"].astype(float) <= 1.0]
    if not sparsity.empty:
        sparsity_keys = ["Dataset", "Model", "Seq_len", "Pred_len"]
        has_thinning = sparsity[sparsity["History_keep_fraction"].astype(float) < 1.0][sparsity_keys].drop_duplicates()
        sparsity = sparsity.merge(has_thinning, on=sparsity_keys, how="inner")
    sparsity = add_original_deltas(sparsity)
    sparsity_long = build_long_table(sparsity)
    sparsity_long["Delta MSE"] = sparsity_long["Delta_MSE"].map(lambda x: "--" if pd.isna(x) else f"{x:+.4f}")
    sparsity_long[["Dataset", "Window", "ConditionLabel", "Model", "MSE", "Delta MSE", "MAE", "Runs"]].to_csv(output_dir / "cross_baseline_sparsity.csv", index=False)
    write_md(output_dir / "cross_baseline_sparsity.md", "Cross-Baseline History Thinning", sparsity_long[["Dataset", "Window", "ConditionLabel", "Model", "MSE", "Delta MSE", "MAE", "Runs"]], ["Dataset", "Window", "ConditionLabel", "Model", "MSE", "Delta MSE", "MAE", "Runs"])
    write_long_tex(output_dir / "cross_baseline_sparsity.tex", "Cross-baseline history thinning. Deltas are relative to each model's full-history evaluation at the same window.", "tab:cross-baseline-sparsity", sparsity_long)

    seeds = frame[(frame["Source"] == "neural") & (frame["Condition"] == "original")].copy()
    if not seeds.empty:
        seeds = order_frame(seeds)
        seeds["Window"] = seeds.apply(lambda r: f"{int(r['Seq_len'])} / {int(r['Pred_len'])}", axis=1)
        seeds["MSE"] = seeds["MSE"].map(lambda x: f"{x:.4f}")
        seeds["MAE"] = seeds["MAE"].map(lambda x: f"{x:.4f}")
        seeds[["Dataset", "Window", "Model", "Seed", "MSE", "MAE", "Metric_path"]].to_csv(output_dir / "cross_baseline_seed_rows.csv", index=False)
        write_md(output_dir / "cross_baseline_seed_rows.md", "Neural Seed Rows", seeds[["Dataset", "Window", "Model", "Seed", "MSE", "MAE"]], ["Dataset", "Window", "Model", "Seed", "MSE", "MAE"])


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Missing input metrics file: {args.input}")
    frame = pd.read_csv(args.input)
    write_outputs(frame, args.output_dir)
    print(f"Wrote cross-baseline tables to {args.output_dir}")


if __name__ == "__main__":
    main()
