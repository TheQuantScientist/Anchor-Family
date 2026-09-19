"""Build compact tables for AnchorFamily CPU experiment summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "anchor_results_family" / "ablation"
METRIC_COLUMNS = {
    "global": ("MAE_scaled", "MSE_scaled"),
    "equal-variable": ("Equal_variable_MAE_scaled", "Equal_variable_MSE_scaled"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate candidate-removal, temporal-perturbation, or sweep tables."
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--kind", choices=["candidate", "temporal", "sweep"], required=True)
    parser.add_argument("--metric-mode", choices=sorted(METRIC_COLUMNS), default="global")
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns in summary CSV: {sorted(missing)}")


def load_summary(summary_path: Path, metric_mode: str) -> pd.DataFrame:
    mae_col, mse_col = METRIC_COLUMNS[metric_mode]
    frame = pd.read_csv(summary_path)
    require_columns(frame, {"Dataset", "Method", mae_col, mse_col, "Predictions"})
    frame = frame.copy()
    frame["MAE"] = pd.to_numeric(frame[mae_col], errors="coerce")
    frame["MSE"] = pd.to_numeric(frame[mse_col], errors="coerce")
    frame["Predictions"] = pd.to_numeric(frame["Predictions"], errors="coerce")
    frame = frame.dropna(subset=["MAE", "MSE", "Predictions"])
    frame = frame.loc[frame["Predictions"] > 0].copy()
    for column, default in [
        ("Base_method", frame["Method"]),
        ("Variant", "default"),
        ("Seq_len", np.nan),
        ("Pred_len", np.nan),
        ("Auto_strategy", "full"),
        ("Candidate_exclusions", ""),
        ("History_perturbation", "original"),
    ]:
        if column not in frame.columns:
            frame[column] = default
    frame["Variant"] = frame["Variant"].fillna("default").replace("", "default")
    frame["Candidate_exclusions"] = frame["Candidate_exclusions"].fillna("")
    return frame


def baseline_lookup(frame: pd.DataFrame, kind: str) -> dict[str, pd.Series]:
    baselines: dict[str, pd.Series] = {}
    for dataset, group in frame.groupby("Dataset", sort=False):
        if kind == "candidate":
            mask = (group["Base_method"] == "AutoAnchor") & (group["Variant"] == "default")
        elif kind == "temporal":
            mask = group["History_perturbation"].fillna("original").eq("original")
        else:
            defaults = group.loc[group["Variant"].eq("default")]
            mask = group.index.isin(defaults.index[:1])
            if not mask.any():
                mask = group.index == group.index[0]
        if mask.any():
            baselines[str(dataset)] = group.loc[mask].iloc[0]
    return baselines


def label_row(row: pd.Series, kind: str) -> str:
    if kind == "candidate":
        if row["Base_method"] == "ERMAnchor":
            return "ERM only"
        if row["Auto_strategy"] == "rules_only" or row["Variant"] == "rules_only":
            return "Auto rules only"
        exclusions = str(row["Candidate_exclusions"] or "")
        if exclusions:
            return "Auto minus " + exclusions.replace(",", "+")
        return "AutoAnchor"
    if kind == "temporal":
        return str(row["History_perturbation"])
    return f"seq={int(row['Seq_len'])}, pred={int(row['Pred_len'])}"


def experiment_table(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    baselines = baseline_lookup(frame, kind)
    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        dataset = str(row["Dataset"])
        baseline = baselines.get(dataset)
        if baseline is None:
            continue
        rows.append(
            {
                "Dataset": dataset,
                "Condition": label_row(row, kind),
                "Method": row["Method"],
                "Variant": row["Variant"],
                "Seq_len": row["Seq_len"],
                "Pred_len": row["Pred_len"],
                "MAE": float(row["MAE"]),
                "MSE": float(row["MSE"]),
                "Delta_MAE_vs_baseline": float(row["MAE"] - baseline["MAE"]),
                "Delta_MSE_vs_baseline": float(row["MSE"] - baseline["MSE"]),
                "Predictions": int(round(float(row["Predictions"]))),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, kind: str, metric_mode: str) -> str:
    title = {
        "candidate": "Anchor Candidate-Removal Ablation",
        "temporal": "Anchor Temporal-Information Test",
        "sweep": "Anchor Lookback/Horizon Sweep",
    }[kind]
    lines = [
        f"# {title}",
        "",
        f"Metric mode: `{metric_mode}`. Lower is better. Deltas are row metric minus the dataset baseline.",
        "",
        "| Dataset | Condition | MAE | Delta MAE | MSE | Delta MSE | Predictions |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in frame.iterrows():
        lines.append(
            "| "
            f"{row['Dataset']} | {row['Condition']} | "
            f"{row['MAE']:.4f} | {row['Delta_MAE_vs_baseline']:+.4f} | "
            f"{row['MSE']:.4f} | {row['Delta_MSE_vs_baseline']:+.4f} | "
            f"{int(row['Predictions'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    frame = experiment_table(load_summary(args.summary, args.metric_mode), args.kind)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"anchor_family_{args.kind}_{args.metric_mode.replace('-', '_')}"
    csv_path = args.output_dir / f"{stem}.csv"
    md_path = args.output_dir / f"{stem}.md"
    frame.to_csv(csv_path, index=False)
    md_path.write_text(markdown_table(frame, args.kind, args.metric_mode), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
