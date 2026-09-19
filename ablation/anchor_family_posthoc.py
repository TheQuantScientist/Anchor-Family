"""Bootstrap confidence intervals and history-stratified errors from detail logs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "anchor_results_family"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "anchor_results_family" / "ablation"
STRATIFY_COLUMNS = [
    "History_density",
    "History_last_gap",
    "History_median_gap",
    "History_volatility",
    "History_mode_fraction",
    "History_n",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute entity-bootstrap CIs and stratified errors from *_DetailLog.csv files."
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args()


def read_detail_logs(result_root: Path) -> pd.DataFrame:
    paths = sorted(result_root.glob("**/*_DetailLog.csv"))
    if not paths:
        raise FileNotFoundError(f"No *_DetailLog.csv files found under {result_root}")
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["Detail_log"] = str(path)
        frames.append(frame)
    detail = pd.concat(frames, ignore_index=True)
    required = {"Dataset", "Method", "Entity", "Actual_scaled", "Predicted_scaled"}
    missing = required - set(detail.columns)
    if missing:
        raise ValueError(f"Detail logs are missing required columns: {sorted(missing)}")
    if "Variant" not in detail.columns:
        detail["Variant"] = "default"
    if "Base_method" not in detail.columns:
        detail["Base_method"] = detail["Method"]
    detail["Variant"] = detail["Variant"].fillna("default").replace("", "default")
    valid = detail[["Actual_scaled", "Predicted_scaled"]].notna().all(axis=1)
    detail = detail.loc[valid].copy()
    residual = detail["Actual_scaled"].astype(float) - detail["Predicted_scaled"].astype(float)
    detail["AE_scaled"] = residual.abs()
    detail["SE_scaled"] = residual**2
    return detail


def bootstrap_group(group: pd.DataFrame, n_bootstrap: int, rng: np.random.Generator) -> dict[str, float]:
    entity_stats = (
        group.groupby("Entity", sort=False)
        .agg(AE_sum=("AE_scaled", "sum"), SE_sum=("SE_scaled", "sum"), Count=("AE_scaled", "size"))
        .reset_index()
    )
    ae_sum = entity_stats["AE_sum"].to_numpy(dtype=float)
    se_sum = entity_stats["SE_sum"].to_numpy(dtype=float)
    counts = entity_stats["Count"].to_numpy(dtype=float)
    n_entities = len(entity_stats)
    point_count = float(counts.sum())
    mae_samples = np.empty(n_bootstrap, dtype=float)
    mse_samples = np.empty(n_bootstrap, dtype=float)

    for index in range(n_bootstrap):
        draw = rng.integers(0, n_entities, size=n_entities)
        denom = float(counts[draw].sum())
        mae_samples[index] = float(ae_sum[draw].sum() / denom)
        mse_samples[index] = float(se_sum[draw].sum() / denom)

    return {
        "Entities": float(n_entities),
        "Predictions": point_count,
        "MAE_scaled": float(group["AE_scaled"].mean()),
        "MSE_scaled": float(group["SE_scaled"].mean()),
        "MAE_ci_low": float(np.quantile(mae_samples, 0.025)),
        "MAE_ci_high": float(np.quantile(mae_samples, 0.975)),
        "MSE_ci_low": float(np.quantile(mse_samples, 0.025)),
        "MSE_ci_high": float(np.quantile(mse_samples, 0.975)),
    }


def bootstrap_table(detail: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    group_cols = ["Dataset", "Method", "Base_method", "Variant"]
    for keys, group in detail.groupby(group_cols, sort=False, dropna=False):
        stats = bootstrap_group(group, n_bootstrap=n_bootstrap, rng=rng)
        rows.append(dict(zip(group_cols, keys), **stats))
    return pd.DataFrame(rows)


def quantile_bucket(values: pd.Series, bins: int = 3) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.notna()
    labels = pd.Series("missing", index=values.index, dtype=object)
    unique_count = numeric.loc[valid].nunique()
    if unique_count < 2:
        labels.loc[valid] = "all"
        return labels
    q = min(bins, unique_count)
    bucket_ids = pd.qcut(numeric.loc[valid].rank(method="first"), q=q, labels=False, duplicates="drop")
    names = ["low", "mid", "high"] if q == 3 else [f"q{idx + 1}" for idx in range(q)]
    labels.loc[valid] = [names[int(bucket)] for bucket in bucket_ids]
    return labels


def stratified_table(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    available = [column for column in STRATIFY_COLUMNS if column in detail.columns]
    if not available:
        return pd.DataFrame()

    group_cols = ["Dataset", "Method", "Base_method", "Variant"]
    for metric in available:
        working = detail.copy()
        working["Stratum"] = quantile_bucket(working[metric])
        for keys, group in working.groupby([*group_cols, "Stratum"], sort=False, dropna=False):
            rows.append(
                {
                    **dict(zip([*group_cols, "Stratum"], keys)),
                    "Stratify_by": metric,
                    "Predictions": int(len(group)),
                    "Entities": int(group["Entity"].nunique()),
                    "MAE_scaled": float(group["AE_scaled"].mean()),
                    "MSE_scaled": float(group["SE_scaled"].mean()),
                    "Mean_stratifier": float(pd.to_numeric(group[metric], errors="coerce").mean()),
                }
            )
    return pd.DataFrame(rows)


def markdown_ci(frame: pd.DataFrame) -> str:
    lines = [
        "# Anchor Bootstrap Confidence Intervals",
        "",
        "Intervals are 95% entity-bootstrap intervals over patients/stations/activity windows.",
        "",
        "| Dataset | Method | Variant | MAE (95% CI) | MSE (95% CI) | Predictions | Entities |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for _, row in frame.iterrows():
        lines.append(
            "| "
            f"{row['Dataset']} | {row['Base_method']} | {row['Variant']} | "
            f"{row['MAE_scaled']:.4f} [{row['MAE_ci_low']:.4f}, {row['MAE_ci_high']:.4f}] | "
            f"{row['MSE_scaled']:.4f} [{row['MSE_ci_low']:.4f}, {row['MSE_ci_high']:.4f}] | "
            f"{int(row['Predictions'])} | {int(row['Entities'])} |"
        )
    return "\n".join(lines) + "\n"


def markdown_strata(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "# Anchor Stratified Error\n\nNo history diagnostic columns were found in the detail logs. Rerun with the updated runner.\n"
    lines = [
        "# Anchor Stratified Error",
        "",
        "History diagnostics are repeated per prediction row; strata are low/mid/high quantiles within each file group.",
        "",
        "| Dataset | Method | Variant | Stratify by | Stratum | MAE | MSE | Predictions |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for _, row in frame.iterrows():
        lines.append(
            "| "
            f"{row['Dataset']} | {row['Base_method']} | {row['Variant']} | "
            f"{row['Stratify_by']} | {row['Stratum']} | "
            f"{row['MAE_scaled']:.4f} | {row['MSE_scaled']:.4f} | {int(row['Predictions'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    detail = read_detail_logs(args.result_root)
    ci = bootstrap_table(detail, n_bootstrap=args.n_bootstrap, seed=args.seed)
    strata = stratified_table(detail)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ci_csv = args.output_dir / "anchor_bootstrap_ci.csv"
    ci_md = args.output_dir / "anchor_bootstrap_ci.md"
    strata_csv = args.output_dir / "anchor_stratified_error.csv"
    strata_md = args.output_dir / "anchor_stratified_error.md"
    ci.to_csv(ci_csv, index=False)
    ci_md.write_text(markdown_ci(ci), encoding="utf-8")
    strata.to_csv(strata_csv, index=False)
    strata_md.write_text(markdown_strata(strata), encoding="utf-8")
    print(f"Wrote {ci_csv}")
    print(f"Wrote {ci_md}")
    print(f"Wrote {strata_csv}")
    print(f"Wrote {strata_md}")


if __name__ == "__main__":
    main()
