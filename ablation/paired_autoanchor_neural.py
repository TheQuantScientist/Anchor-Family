"""Paired entity-level tests between AutoAnchor and saved neural predictions."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = PROJECT_ROOT / "vendor" / "upstream_benchmark"
SRC_ROOT = PROJECT_ROOT / "src"
for path in [BENCHMARK_ROOT, SRC_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

DATASETS = {
    "P12": {"apn_name": "P12", "window": (36, 3)},
    "MIMIC": {"apn_name": "MIMIC_III", "window": (72, 3)},
    "USHCN": {"apn_name": "USHCN", "window": (150, 3)},
    "HumanActivity": {"apn_name": "HumanActivity", "window": (3000, 300)},
}
UPSTREAM_PATCH_MODEL = "A" + "PN"
MODELS = [UPSTREAM_PATCH_MODEL, "GraFITi", "tPatchGNN"]
ARRAY_NAMES = ["input_y.npy", "input_y_mask.npy", "input_sample_ID.npy", "output_pred.npy"]
KEYS = ["Entity", "Variable_index", "Step"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute paired AutoAnchor-versus-neural tests from compact prediction arrays."
    )
    parser.add_argument(
        "--benchmark-results-root",
        dest="benchmark_results_root",
        type=Path,
        default=BENCHMARK_ROOT / "storage/results",
    )
    parser.add_argument("--ablation-name", default="cross_baseline")
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
    )
    parser.add_argument(
        "--humanactivity-root",
        type=Path,
        default=BENCHMARK_ROOT / "storage/datasets/HumanActivity",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "cross_baseline_results/ablation",
    )
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--n-permutations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--min-coordinate-coverage", type=float, default=0.75)
    return parser.parse_args()


def normalize_entity(value: object) -> str:
    text = str(value)
    try:
        number = float(text)
    except ValueError:
        return text
    if np.isfinite(number) and number.is_integer():
        return str(int(number))
    return text


def anchor_root(args: argparse.Namespace, dataset: str) -> Path:
    if dataset == "P12":
        return args.p12_root
    if dataset == "MIMIC":
        return args.mimic_root
    return args.shared_root


def find_anchor_file(root: Path, dataset: str, suffix: str) -> Path:
    candidates = sorted(root.glob(f"**/*auto_anchor*{suffix}.csv"))
    for path in candidates:
        frame = pd.read_csv(path, nrows=5)
        if "Dataset" in frame and dataset in set(frame["Dataset"].astype(str)):
            return path
    raise FileNotFoundError(f"No AutoAnchor {suffix} file for {dataset} under {root}")


def read_anchor(dataset: str, args: argparse.Namespace) -> pd.DataFrame:
    root = anchor_root(args, dataset)
    detail_path = find_anchor_file(root, dataset, "DetailLog")
    calibration_path = find_anchor_file(root, dataset, "Calibration")
    detail = pd.read_csv(detail_path, dtype={"Entity": str})
    calibration = pd.read_csv(calibration_path)
    seq_len, pred_len = DATASETS[dataset]["window"]

    if "Base_method" in detail:
        detail = detail.loc[detail["Base_method"].eq("AutoAnchor")]
    if "Variant" in detail:
        detail = detail.loc[detail["Variant"].fillna("default").eq("default")]
    if "History_perturbation" in detail:
        detail = detail.loc[
            detail["History_perturbation"].fillna("original").eq("original")
        ]
    detail = detail.loc[
        detail["Dataset"].astype(str).eq(dataset)
        & pd.to_numeric(detail["Seq_len"], errors="coerce").eq(seq_len)
        & pd.to_numeric(detail["Pred_len"], errors="coerce").eq(pred_len)
    ].copy()
    if detail.empty:
        raise ValueError(f"No standard-window AutoAnchor rows in {detail_path}")

    variable_order = list(dict.fromkeys(calibration["Variable"].astype(str)))
    variable_index = {name: index for index, name in enumerate(variable_order)}
    detail["Variable_index"] = detail["Variable"].astype(str).map(variable_index)
    if detail["Variable_index"].isna().any():
        missing = sorted(detail.loc[detail["Variable_index"].isna(), "Variable"].unique())
        raise ValueError(f"Missing variable indices for {dataset}: {missing}")

    detail["Entity"] = detail["Entity"].map(normalize_entity)
    detail["Step"] = pd.to_numeric(detail["Step"], errors="raise").astype(int)
    detail["Variable_index"] = detail["Variable_index"].astype(int)
    detail["Actual_scaled"] = pd.to_numeric(detail["Actual_scaled"], errors="raise")
    detail["Predicted_scaled"] = pd.to_numeric(detail["Predicted_scaled"], errors="raise")
    detail["Auto_AE"] = (detail["Actual_scaled"] - detail["Predicted_scaled"]).abs()
    detail["Auto_SE"] = (detail["Actual_scaled"] - detail["Predicted_scaled"]) ** 2
    columns = [*KEYS, "Actual_scaled", "Auto_AE", "Auto_SE"]
    out = detail[columns].copy()
    if out.duplicated(KEYS).any():
        raise ValueError(f"Duplicate AutoAnchor target coordinates for {dataset}")
    return out


def humanactivity_entity_map(args: argparse.Namespace) -> dict[str, str]:
    from sklearn import model_selection

    from data.dependencies.HumanActivity.HumanActivity import HumanActivity
    from anchorfamily.experiments.anchor_baseline import build_human_activity_samples

    raw = HumanActivity(root=str(args.humanactivity_root), download=False)
    _, test_records = model_selection.train_test_split(
        raw, train_size=0.9, random_state=42, shuffle=False
    )
    seq_len, pred_len = DATASETS["HumanActivity"]["window"]
    samples = build_human_activity_samples(test_records, seq_len, pred_len)
    return {str(index): normalize_entity(sample.key) for index, sample in enumerate(samples)}


def candidate_eval_dirs(
    args: argparse.Namespace, dataset: str, model: str
) -> dict[int, Path]:
    info = DATASETS[dataset]
    seq_len, pred_len = info["window"]
    model_id = f"cross_{model}_{dataset}_sl{seq_len}_pl{pred_len}"
    root = (
        args.benchmark_results_root
        / args.ablation_name
        / info["apn_name"]
        / model
        / model_id
        / f"{seq_len}_{pred_len}"
    )
    selected: dict[int, Path] = {}
    for metric_path in root.glob("*/iter*/eval_*/metric.json"):
        folder = metric_path.parent
        if not all((folder / name).exists() for name in ARRAY_NAMES):
            continue
        config_path = folder / "eval_configs.yaml"
        config = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
        config = config or {}
        if str(config.get("history_perturbation", "original")) != "original":
            continue
        if abs(float(config.get("history_keep_fraction", 1.0)) - 1.0) > 1e-9:
            continue
        iter_name = folder.parent.name
        if not iter_name.startswith("iter"):
            continue
        seed_index = int(iter_name.removeprefix("iter"))
        previous = selected.get(seed_index)
        if previous is None or folder.stat().st_mtime > previous.stat().st_mtime:
            selected[seed_index] = folder
    if not selected:
        raise FileNotFoundError(
            f"No compact prediction arrays for {model}/{dataset} under {root}"
        )
    return selected


def neural_coordinates(
    folder: Path,
    dataset: str,
    human_map: dict[str, str] | None,
) -> pd.DataFrame:
    y = np.load(folder / "input_y.npy")
    mask = np.load(folder / "input_y_mask.npy")
    sample_ids = np.load(folder / "input_sample_ID.npy").reshape(-1)
    pred = np.load(folder / "output_pred.npy")
    if y.shape != mask.shape or y.shape != pred.shape:
        raise ValueError(
            f"Array shape mismatch in {folder}: y={y.shape}, mask={mask.shape}, pred={pred.shape}"
        )
    if y.shape[0] != len(sample_ids) or y.ndim < 3:
        raise ValueError(f"Unexpected saved array shape in {folder}: {y.shape}")

    n_samples = y.shape[0]
    n_variables = y.shape[-1]
    y = y.reshape(n_samples, -1, n_variables)
    mask = mask.reshape(n_samples, -1, n_variables) > 0
    pred = pred.reshape(n_samples, -1, n_variables)
    rows: list[dict[str, object]] = []
    for sample_index, raw_id in enumerate(sample_ids):
        numeric_id = normalize_entity(raw_id)
        entity = human_map.get(numeric_id) if human_map is not None else numeric_id
        if entity is None:
            raise ValueError(f"No HumanActivity entity mapping for sample ID {numeric_id}")
        for variable_index in range(n_variables):
            positions = np.flatnonzero(mask[sample_index, :, variable_index])
            for step, position in enumerate(positions):
                actual = float(y[sample_index, position, variable_index])
                prediction = float(pred[sample_index, position, variable_index])
                rows.append(
                    {
                        "Entity": entity,
                        "Variable_index": variable_index,
                        "Step": step,
                        "Neural_actual": actual,
                        "Neural_AE": abs(actual - prediction),
                        "Neural_SE": (actual - prediction) ** 2,
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.duplicated(KEYS).any():
        raise ValueError(f"Duplicate neural target coordinates in {folder}")
    return frame


def paired_seed(
    anchor: pd.DataFrame,
    neural: pd.DataFrame,
    dataset: str,
    model: str,
    seed_index: int,
    min_coordinate_coverage: float,
) -> pd.DataFrame:
    paired = anchor.merge(neural, on=KEYS, how="inner", validate="one_to_one")
    coverage = len(paired) / len(anchor)
    if coverage < min_coordinate_coverage:
        raise ValueError(
            f"{model}/{dataset}/iter{seed_index} pairs only {len(paired)}/{len(anchor)} "
            f"AutoAnchor coordinates ({coverage:.2%}); minimum is "
            f"{min_coordinate_coverage:.2%}"
        )
    if coverage < 1.0:
        warnings.warn(
            f"{model}/{dataset}/iter{seed_index} uses the shared-coordinate "
            f"subset: {len(paired)}/{len(anchor)} ({coverage:.2%})",
            stacklevel=2,
        )
    if not np.allclose(
        paired["Actual_scaled"], paired["Neural_actual"], rtol=1e-4, atol=2e-5
    ):
        max_delta = float(
            np.max(np.abs(paired["Actual_scaled"] - paired["Neural_actual"]))
        )
        raise ValueError(
            f"Target mismatch for {model}/{dataset}/iter{seed_index}; max delta={max_delta:g}"
        )
    paired["Seed_index"] = seed_index
    paired["Anchor_predictions"] = len(anchor)
    paired["Coordinate_coverage"] = coverage
    return paired


def paired_test(
    entity: pd.DataFrame,
    metric: str,
    n_bootstrap: int,
    n_permutations: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    auto = entity[f"Auto_{metric}_sum"].to_numpy(float)
    neural = entity[f"Neural_{metric}_sum"].to_numpy(float)
    counts = entity["Count"].to_numpy(float)
    differences = auto - neural
    observed = float(differences.sum() / counts.sum())

    draws = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sample = rng.integers(0, len(entity), len(entity))
        draws[index] = differences[sample].sum() / counts[sample].sum()

    extreme = 0
    for _ in range(n_permutations):
        signs = rng.choice((-1.0, 1.0), size=len(entity))
        value = float((differences * signs).sum() / counts.sum())
        extreme += abs(value) >= abs(observed)

    return {
        "Delta": observed,
        "CI_low": float(np.quantile(draws, 0.025)),
        "CI_high": float(np.quantile(draws, 0.975)),
        "p_value": float((extreme + 1) / (n_permutations + 1)),
    }


def compute(args: argparse.Namespace) -> pd.DataFrame:
    human_map: dict[str, str] | None = None
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []

    for dataset in DATASETS:
        if dataset == "HumanActivity":
            human_map = humanactivity_entity_map(args)
        anchor = read_anchor(dataset, args)
        for model in MODELS:
            seed_frames: list[pd.DataFrame] = []
            eval_dirs = candidate_eval_dirs(args, dataset, model)
            for seed_index, folder in sorted(eval_dirs.items()):
                neural = neural_coordinates(
                    folder,
                    dataset,
                    human_map,
                )
                seed_frames.append(
                    paired_seed(
                        anchor,
                        neural,
                        dataset,
                        model,
                        seed_index,
                        args.min_coordinate_coverage,
                    )
                )

            combined = pd.concat(seed_frames, ignore_index=True)
            anchor_predictions = int(combined["Anchor_predictions"].iloc[0])
            coordinate_coverage = float(combined["Coordinate_coverage"].min())
            coordinate = (
                combined.groupby(KEYS, sort=False)
                .agg(
                    Actual_scaled=("Actual_scaled", "first"),
                    Auto_AE=("Auto_AE", "first"),
                    Auto_SE=("Auto_SE", "first"),
                    Neural_AE=("Neural_AE", "mean"),
                    Neural_SE=("Neural_SE", "mean"),
                    Runs=("Seed_index", "nunique"),
                )
                .reset_index()
            )
            expected_runs = len(seed_frames)
            if not coordinate["Runs"].eq(expected_runs).all():
                raise ValueError(f"Incomplete seed pairing for {model}/{dataset}")

            entity = coordinate.groupby("Entity", sort=False).agg(
                Auto_MAE_sum=("Auto_AE", "sum"),
                Auto_MSE_sum=("Auto_SE", "sum"),
                Neural_MAE_sum=("Neural_AE", "sum"),
                Neural_MSE_sum=("Neural_SE", "sum"),
                Count=("Auto_AE", "size"),
            )
            for metric in ["MAE", "MSE"]:
                stats = paired_test(
                    entity,
                    metric,
                    args.n_bootstrap,
                    args.n_permutations,
                    rng,
                )
                rows.append(
                    {
                        "Dataset": dataset,
                        "Model": model,
                        "Metric": metric,
                        "Neural_runs": expected_runs,
                        "Predictions": int(entity["Count"].sum()),
                        "Anchor_predictions": anchor_predictions,
                        "Coordinate_coverage": coordinate_coverage,
                        "Entities": len(entity),
                        "AutoAnchor": float(
                            entity[f"Auto_{metric}_sum"].sum()
                            / entity["Count"].sum()
                        ),
                        "Neural": float(
                            entity[f"Neural_{metric}_sum"].sum()
                            / entity["Count"].sum()
                        ),
                        **stats,
                    }
                )
    return pd.DataFrame(rows)


def format_p(value: float) -> str:
    return "<1e-4" if value < 1e-4 else f"{value:.4f}"


def write_outputs(frame: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "paired_autoanchor_vs_neural.csv", index=False)

    markdown = [
        "# Paired AutoAnchor vs Neural Tests",
        "",
        "Delta is AutoAnchor minus the seed-averaged neural error; negative favors AutoAnchor.",
        "",
        "| Dataset | Model | Metric | AutoAnchor | Neural | Delta [95% CI] | p | Paired predictions | Coverage | Entities | Seeds |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    tex_rows: list[str] = []
    for _, row in frame.iterrows():
        interval = (
            f"{row['Delta']:+.4f} [{row['CI_low']:+.4f}, {row['CI_high']:+.4f}]"
        )
        p_value = format_p(float(row["p_value"]))
        markdown.append(
            f"| {row['Dataset']} | {row['Model']} | {row['Metric']} | "
            f"{row['AutoAnchor']:.4f} | {row['Neural']:.4f} | {interval} | "
            f"{p_value} | {int(row['Predictions']):,} | {row['Coordinate_coverage']:.2%} | "
            f"{int(row['Entities']):,} | "
            f"{int(row['Neural_runs'])} |"
        )
        tex_p = "$<10^{-4}$" if float(row["p_value"]) < 1e-4 else p_value
        tex_rows.append(
            f"{row['Dataset']} & {row['Model']} & {row['Metric']} & "
            f"{row['AutoAnchor']:.4f} & {row['Neural']:.4f} & {interval} & "
            f"{tex_p} & {int(row['Entities']):,} \\\\"
        )

    (output_dir / "paired_autoanchor_vs_neural.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    (output_dir / "paired_autoanchor_vs_neural_rows.tex").write_text(
        "\n".join(tex_rows) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    frame = compute(args)
    write_outputs(frame, args.output_dir)
    print(f"Wrote paired tests to {args.output_dir}")


if __name__ == "__main__":
    main()
