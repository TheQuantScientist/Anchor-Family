"""Build paper tables from completed Anchor-family prediction logs.

This script performs post-processing only. It does not train or evaluate a
forecasting model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHODS = {"NaiveAnchor", "ExpoAnchor", "SparseAnchor", "ERMAnchor", "AutoAnchor"}
PAIR_KEYS = ["Variable", "Entity", "Step", "Target_time"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build uncertainty, paired-test, and stratified tables from saved predictions."
    )
    parser.add_argument(
        "--p12-root",
        type=Path,
        default=PROJECT_ROOT / "reruns/p12_aligned/anchor_results_family",
    )
    parser.add_argument(
        "--mimic-root",
        type=Path,
        default=PROJECT_ROOT / "reruns/mimic_debrouwer/anchor_results_family",
    )
    parser.add_argument(
        "--shared-root",
        type=Path,
        default=PROJECT_ROOT / "anchor_results_family",
        help="Family-result root containing USHCN and HumanActivity.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "docs/figures",
    )
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--n-permutations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args()


def read_dataset(root: Path, dataset: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(root.glob("**/*_DetailLog.csv")):
        frame = pd.read_csv(path)
        if "Dataset" not in frame or dataset not in set(frame["Dataset"].astype(str)):
            continue
        frames.append(frame.loc[frame["Dataset"].astype(str) == dataset].copy())
    if not frames:
        raise FileNotFoundError(f"No {dataset} detail logs found under {root}")

    detail = pd.concat(frames, ignore_index=True)
    if "Base_method" not in detail:
        detail["Base_method"] = detail["Method"]
    if "Variant" not in detail:
        detail["Variant"] = "default"
    if "History_perturbation" not in detail:
        detail["History_perturbation"] = "original"
    detail["Variant"] = detail["Variant"].fillna("default").replace("", "default")
    detail["History_perturbation"] = (
        detail["History_perturbation"].fillna("original").replace("", "original")
    )
    detail = detail.loc[
        detail["Base_method"].isin(METHODS)
        & detail["Variant"].eq("default")
        & detail["History_perturbation"].eq("original")
    ].copy()
    detail["Actual_scaled"] = pd.to_numeric(detail["Actual_scaled"], errors="coerce")
    detail["Predicted_scaled"] = pd.to_numeric(detail["Predicted_scaled"], errors="coerce")
    detail = detail.dropna(subset=["Actual_scaled", "Predicted_scaled"])
    residual = detail["Actual_scaled"] - detail["Predicted_scaled"]
    detail["AE_scaled"] = residual.abs()
    detail["SE_scaled"] = residual**2
    return detail


def read_all(args: argparse.Namespace) -> pd.DataFrame:
    roots = {
        "P12": args.p12_root,
        "MIMIC": args.mimic_root,
        "USHCN": args.shared_root,
        "HumanActivity": args.shared_root,
    }
    return pd.concat(
        [read_dataset(root, dataset) for dataset, root in roots.items()],
        ignore_index=True,
    )


def cluster_bootstrap_ci(
    group: pd.DataFrame, n_bootstrap: int, rng: np.random.Generator
) -> dict[str, float]:
    entity = group.groupby("Entity", sort=False).agg(
        ae_sum=("AE_scaled", "sum"),
        se_sum=("SE_scaled", "sum"),
        count=("AE_scaled", "size"),
    )
    ae = entity["ae_sum"].to_numpy(float)
    se = entity["se_sum"].to_numpy(float)
    count = entity["count"].to_numpy(float)
    mae_draws = np.empty(n_bootstrap)
    mse_draws = np.empty(n_bootstrap)
    for index in range(n_bootstrap):
        draw = rng.integers(0, len(entity), len(entity))
        denominator = count[draw].sum()
        mae_draws[index] = ae[draw].sum() / denominator
        mse_draws[index] = se[draw].sum() / denominator
    return {
        "Predictions": float(count.sum()),
        "Entities": float(len(entity)),
        "MAE": float(group["AE_scaled"].mean()),
        "MAE_low": float(np.quantile(mae_draws, 0.025)),
        "MAE_high": float(np.quantile(mae_draws, 0.975)),
        "MSE": float(group["SE_scaled"].mean()),
        "MSE_low": float(np.quantile(mse_draws, 0.025)),
        "MSE_high": float(np.quantile(mse_draws, 0.975)),
    }


def bootstrap_table(detail: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    selected = {
        "P12": ["NaiveAnchor", "ERMAnchor", "AutoAnchor"],
        "MIMIC": ["NaiveAnchor", "ERMAnchor", "AutoAnchor"],
        "USHCN": ["NaiveAnchor", "SparseAnchor", "AutoAnchor"],
        "HumanActivity": ["NaiveAnchor", "ERMAnchor", "AutoAnchor"],
    }
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    for dataset, methods in selected.items():
        for method in methods:
            group = detail.loc[
                detail["Dataset"].eq(dataset) & detail["Base_method"].eq(method)
            ]
            if group.empty:
                raise ValueError(f"Missing {dataset}/{method} predictions")
            rows.append(
                {
                    "Dataset": dataset,
                    "Method": method,
                    **cluster_bootstrap_ci(group, args.n_bootstrap, rng),
                }
            )
    return pd.DataFrame(rows)


def paired_frames(detail: pd.DataFrame, dataset: str) -> pd.DataFrame:
    columns = [*PAIR_KEYS, "Actual_scaled", "Predicted_scaled"]
    auto = detail.loc[
        detail["Dataset"].eq(dataset) & detail["Base_method"].eq("AutoAnchor"), columns
    ].copy()
    naive = detail.loc[
        detail["Dataset"].eq(dataset) & detail["Base_method"].eq("NaiveAnchor"), columns
    ].copy()
    auto = auto.rename(
        columns={"Actual_scaled": "actual_auto", "Predicted_scaled": "pred_auto"}
    )
    naive = naive.rename(
        columns={"Actual_scaled": "actual_naive", "Predicted_scaled": "pred_naive"}
    )
    paired = auto.merge(naive, on=PAIR_KEYS, how="inner", validate="one_to_one")
    if len(paired) != len(auto) or len(paired) != len(naive):
        raise ValueError(
            f"Unmatched {dataset} predictions: auto={len(auto)}, naive={len(naive)}, paired={len(paired)}"
        )
    if not np.allclose(paired["actual_auto"], paired["actual_naive"], equal_nan=True):
        raise ValueError(f"Target mismatch while pairing {dataset}")
    paired["ae_auto"] = (paired["actual_auto"] - paired["pred_auto"]).abs()
    paired["ae_naive"] = (paired["actual_naive"] - paired["pred_naive"]).abs()
    paired["se_auto"] = (paired["actual_auto"] - paired["pred_auto"]) ** 2
    paired["se_naive"] = (paired["actual_naive"] - paired["pred_naive"]) ** 2
    return paired


def paired_metric_test(
    entity: pd.DataFrame,
    metric: str,
    n_bootstrap: int,
    n_permutations: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    auto = entity[f"{metric}_auto_sum"].to_numpy(float)
    naive = entity[f"{metric}_naive_sum"].to_numpy(float)
    counts = entity["count"].to_numpy(float)
    difference = auto - naive
    observed = float(difference.sum() / counts.sum())
    draws = np.empty(n_bootstrap)
    for index in range(n_bootstrap):
        sample = rng.integers(0, len(entity), len(entity))
        draws[index] = difference[sample].sum() / counts[sample].sum()
    extreme = 0
    for _ in range(n_permutations):
        signs = rng.choice((-1.0, 1.0), size=len(entity))
        permuted = float((difference * signs).sum() / counts.sum())
        extreme += abs(permuted) >= abs(observed)
    return {
        "Delta": observed,
        "CI_low": float(np.quantile(draws, 0.025)),
        "CI_high": float(np.quantile(draws, 0.975)),
        "p_value": float((extreme + 1) / (n_permutations + 1)),
    }


def paired_table(detail: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed + 1)
    rows: list[dict[str, object]] = []
    for dataset in ["P12", "MIMIC", "USHCN", "HumanActivity"]:
        paired = paired_frames(detail, dataset)
        entity = paired.groupby("Entity", sort=False).agg(
            ae_auto_sum=("ae_auto", "sum"),
            ae_naive_sum=("ae_naive", "sum"),
            se_auto_sum=("se_auto", "sum"),
            se_naive_sum=("se_naive", "sum"),
            count=("ae_auto", "size"),
        )
        for metric in ["mae", "mse"]:
            prefix = "ae" if metric == "mae" else "se"
            renamed = entity.rename(
                columns={
                    f"{prefix}_auto_sum": f"{metric}_auto_sum",
                    f"{prefix}_naive_sum": f"{metric}_naive_sum",
                }
            )
            stats = paired_metric_test(
                renamed,
                metric,
                args.n_bootstrap,
                args.n_permutations,
                rng,
            )
            rows.append(
                {
                    "Dataset": dataset,
                    "Metric": metric.upper(),
                    "Predictions": len(paired),
                    "Entities": len(entity),
                    **stats,
                }
            )
    return pd.DataFrame(rows)


def stratified_table(detail: pd.DataFrame, column: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset in ["P12", "MIMIC", "USHCN", "HumanActivity"]:
        auto = detail.loc[
            detail["Dataset"].eq(dataset) & detail["Base_method"].eq("AutoAnchor")
        ].copy()
        naive = detail.loc[
            detail["Dataset"].eq(dataset) & detail["Base_method"].eq("NaiveAnchor")
        ].copy()
        auto[column] = pd.to_numeric(auto[column], errors="coerce")
        auto = auto.dropna(subset=[column])
        auto["Stratum"] = pd.qcut(
            auto[column].rank(method="first"),
            q=3,
            labels=["low", "mid", "high"],
        )
        paired = auto[[*PAIR_KEYS, "Stratum", "AE_scaled", "SE_scaled"]].merge(
            naive[[*PAIR_KEYS, "AE_scaled", "SE_scaled"]],
            on=PAIR_KEYS,
            suffixes=("_auto", "_naive"),
            how="inner",
            validate="one_to_one",
        )
        for stratum, group in paired.groupby("Stratum", observed=True, sort=False):
            auto_mse = float(group["SE_scaled_auto"].mean())
            naive_mse = float(group["SE_scaled_naive"].mean())
            rows.append(
                {
                    "Dataset": dataset,
                    "Stratum": str(stratum),
                    "Predictions": len(group),
                    "Auto_MAE": float(group["AE_scaled_auto"].mean()),
                    "Naive_MAE": float(group["AE_scaled_naive"].mean()),
                    "Auto_MSE": auto_mse,
                    "Naive_MSE": naive_mse,
                    "MSE_reduction_pct": 100.0 * (naive_mse - auto_mse) / naive_mse,
                }
            )
    return pd.DataFrame(rows)


def format_p(value: float) -> str:
    return "$<10^{-4}$" if value < 1e-4 else f"{value:.4f}"


def p12_bootstrap_rows(frame: pd.DataFrame) -> str:
    names = {"AutoAnchor": "\\method{}"}
    lines: list[str] = []
    for _, row in frame.loc[frame["Dataset"].eq("P12")].iterrows():
        method = names.get(row["Method"], row["Method"])
        lines.append(
            f"P12 & {method} & "
            f"{row['MAE']:.4f} [{row['MAE_low']:.4f}, {row['MAE_high']:.4f}] & "
            f"{row['MSE']:.4f} [{row['MSE_low']:.4f}, {row['MSE_high']:.4f}] & "
            f"{int(row['Entities']):,} \\\\"
        )
    return "\n".join(lines) + "\n"


def paired_rows(frame: pd.DataFrame) -> str:
    lines: list[str] = []
    for _, row in frame.iterrows():
        lines.append(
            f"{row['Dataset']} & {row['Metric']} & {row['Delta']:+.4f} "
            f"[{row['CI_low']:+.4f}, {row['CI_high']:+.4f}] & "
            f"{format_p(row['p_value'])} & {int(row['Predictions']):,} & "
            f"{int(row['Entities']):,} \\\\"
        )
    return "\n".join(lines) + "\n"


def stratified_rows(frame: pd.DataFrame) -> str:
    lines: list[str] = []
    for _, row in frame.iterrows():
        lines.append(
            f"{row['Dataset']} & {row['Stratum']} & {row['Auto_MAE']:.4f} & "
            f"{row['Naive_MAE']:.4f} & {row['Auto_MSE']:.4f} & "
            f"{row['Naive_MSE']:.4f} & {row['MSE_reduction_pct']:+.1f} & "
            f"{int(row['Predictions']):,} \\\\"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    detail = read_all(args)
    bootstrap = bootstrap_table(detail, args)
    paired = paired_table(detail, args)
    volatility = stratified_table(detail, "History_volatility")
    density = stratified_table(detail, "History_density")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bootstrap.to_csv(args.output_dir / "anchor_bootstrap_ci.csv", index=False)
    paired.to_csv(args.output_dir / "anchor_paired_vs_naive.csv", index=False)
    volatility.to_csv(args.output_dir / "anchor_stratified_volatility.csv", index=False)
    density.to_csv(args.output_dir / "anchor_stratified_density.csv", index=False)
    (args.output_dir / "p12_bootstrap_rows.tex").write_text(
        p12_bootstrap_rows(bootstrap), encoding="utf-8"
    )
    (args.output_dir / "anchor_paired_vs_naive_rows.tex").write_text(
        paired_rows(paired), encoding="utf-8"
    )
    (args.output_dir / "anchor_stratified_volatility_rows.tex").write_text(
        stratified_rows(volatility),
        encoding="utf-8",
    )
    (args.output_dir / "anchor_stratified_density_rows.tex").write_text(
        stratified_rows(density),
        encoding="utf-8",
    )
    print(f"Wrote derived tables to {args.output_dir}")


if __name__ == "__main__":
    main()
