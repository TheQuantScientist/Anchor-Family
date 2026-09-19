"""Orchestrate CPU-only AnchorFamily experiments for the paper artifact trail."""

from __future__ import annotations

import argparse
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chronolm.cli_utils import slugify, split_csv_values, unique_preserve_order  # noqa: E402
from chronolm.experiments.anchor_baseline import (  # noqa: E402
    ANCHOR_FAMILY_METHODS,
    ANCHOR_METHOD_SLUGS,
    DATASET_DEFAULTS,
    HISTORY_PERTURBATIONS,
    RUN_NAME_PREFIX,
    AnchorConfig,
    canonical_dataset_name,
    run,
)

DATASET_ORDER = ["P12", "MIMIC", "USHCN", "HumanActivity"]
SUITE_ROOTS = {
    "family": "anchor_results_family",
    "candidate": "anchor_results_candidate_ablation",
    "temporal": "anchor_results_temporal",
    "sweeps": "anchor_results_sweeps",
}
SWEEP_GRIDS = {
    "P12": [(12, 1), (24, 3), (36, 3), (36, 6), (48, 6)],
    "MIMIC": [(12, 1), (24, 3), (36, 3), (48, 6), (72, 3), (72, 6)],
    "USHCN": [(50, 1), (100, 3), (150, 3), (150, 7), (220, 7)],
    "HumanActivity": [(1000, 100), (2000, 200), (3000, 300), (3000, 600), (6000, 600)],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CPU-only AnchorFamily paper experiments through the shared runner."
    )
    parser.add_argument(
        "--suite",
        action="append",
        choices=["family", "candidate", "temporal", "sweeps", "all"],
        default=[],
        help="Suite to run. Repeatable. Default: all.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset(s): P12, MIMIC, USHCN, HumanActivity, or all. Comma-separated allowed.",
    )
    parser.add_argument(
        "--output-parent",
        type=Path,
        default=PROJECT_ROOT,
        help="Parent directory for suite result roots.",
    )
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--trace-every", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=1729)
    parser.add_argument(
        "--parallel-runs",
        type=int,
        default=1,
        help="Independent Anchor configurations to evaluate concurrently on CPU.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite each run artifact and rebuild suite summaries from scratch.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running later configs if one dataset/method fails; failures are written to run_failures.csv.",
    )
    return parser.parse_args()


def selected_suites(args: argparse.Namespace) -> list[str]:
    raw = split_csv_values(args.suite, default=["all"])
    if "all" in raw:
        return ["family", "candidate", "temporal", "sweeps"]
    return unique_preserve_order(raw)


def selected_datasets(args: argparse.Namespace) -> list[str]:
    raw = split_csv_values(args.dataset, default=["all"])
    if any(value.lower() == "all" for value in raw):
        return DATASET_ORDER
    return unique_preserve_order(canonical_dataset_name(value) for value in raw)


def config_for(
    dataset: str,
    method: str,
    output_root: Path,
    args: argparse.Namespace,
    tag: str = "",
    seq_len: int | None = None,
    pred_len: int | None = None,
    auto_strategy: str = "full",
    candidate_exclusions: tuple[str, ...] = (),
    history_perturbation: str = "original",
) -> AnchorConfig:
    run_name = f"{RUN_NAME_PREFIX}_{ANCHOR_METHOD_SLUGS[method]}_{dataset.lower()}"
    if tag:
        run_name = f"{run_name}_{slugify(tag)}"
    return AnchorConfig(
        dataset_name=dataset,
        anchor_method=method,
        seq_len=seq_len,
        pred_len=pred_len,
        max_test_samples=args.max_test_samples,
        auto_strategy=auto_strategy,
        candidate_exclusions=candidate_exclusions,
        history_perturbation=history_perturbation,
        random_seed=args.random_seed,
        overwrite=args.overwrite,
        run_name=run_name,
        output_dir=output_root / dataset.lower(),
        trace_every=args.trace_every,
        progress_every=args.progress_every,
    )


def family_configs(datasets: list[str], output_root: Path, args: argparse.Namespace) -> list[AnchorConfig]:
    return [
        config_for(dataset, method, output_root, args)
        for dataset in datasets
        for method in ANCHOR_FAMILY_METHODS
    ]


def candidate_configs(datasets: list[str], output_root: Path, args: argparse.Namespace) -> list[AnchorConfig]:
    variants: list[tuple[str, str, str, tuple[str, ...]]] = [
        ("AutoAnchor", "default", "full", ()),
        ("ERMAnchor", "erm_only", "full", ()),
        ("AutoAnchor", "rules_only", "rules_only", ()),
        ("AutoAnchor", "minus_ema", "full", ("ema",)),
        ("AutoAnchor", "minus_sparse", "full", ("sparse",)),
        ("AutoAnchor", "minus_trend", "full", ("trend",)),
        ("AutoAnchor", "minus_phase", "full", ("phase",)),
    ]
    return [
        config_for(
            dataset=dataset,
            method=method,
            output_root=output_root,
            args=args,
            tag=tag,
            auto_strategy=auto_strategy,
            candidate_exclusions=exclusions,
        )
        for dataset in datasets
        for method, tag, auto_strategy, exclusions in variants
    ]


def temporal_configs(datasets: list[str], output_root: Path, args: argparse.Namespace) -> list[AnchorConfig]:
    return [
        config_for(
            dataset=dataset,
            method="AutoAnchor",
            output_root=output_root,
            args=args,
            tag=perturbation,
            history_perturbation=perturbation,
        )
        for dataset in datasets
        for perturbation in HISTORY_PERTURBATIONS
    ]


def sweep_configs(datasets: list[str], output_root: Path, args: argparse.Namespace) -> list[AnchorConfig]:
    configs: list[AnchorConfig] = []
    for dataset in datasets:
        defaults = DATASET_DEFAULTS[dataset]
        grid = list(dict.fromkeys([*SWEEP_GRIDS[dataset], (defaults["seq_len"], defaults["pred_len"])]))
        for seq_len, pred_len in grid:
            configs.append(
                config_for(
                    dataset=dataset,
                    method="AutoAnchor",
                    output_root=output_root,
                    args=args,
                    tag=f"sl{seq_len}_pl{pred_len}",
                    seq_len=seq_len,
                    pred_len=pred_len,
                )
            )
    return configs


def suite_configs(suite: str, datasets: list[str], output_root: Path, args: argparse.Namespace) -> list[AnchorConfig]:
    if suite == "family":
        return family_configs(datasets, output_root, args)
    if suite == "candidate":
        return candidate_configs(datasets, output_root, args)
    if suite == "temporal":
        return temporal_configs(datasets, output_root, args)
    if suite == "sweeps":
        return sweep_configs(datasets, output_root, args)
    raise ValueError(f"Unsupported suite: {suite}")


def run_config(config: AnchorConfig) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    try:
        return run(config), None
    except Exception as exc:
        return None, {
            "Dataset": config.dataset_name,
            "Method": config.anchor_method,
            "Run_name": config.run_name,
            "Error_type": type(exc).__name__,
            "Error": str(exc),
            "Traceback": traceback.format_exc(),
        }


def run_suite(suite: str, datasets: list[str], args: argparse.Namespace) -> None:
    output_root = args.output_parent / SUITE_ROOTS[suite]
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "anchor_summary.csv"
    failure_path = output_root / "run_failures.csv"
    if args.overwrite:
        summary_path.unlink(missing_ok=True)
        failure_path.unlink(missing_ok=True)

    summaries: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    configs = suite_configs(suite, datasets, output_root, args)
    print(f"Running suite={suite} datasets={','.join(datasets)} runs={len(configs)}")
    if args.parallel_runs == 1 or len(configs) <= 1:
        completed = ((index, config, *run_config(config)) for index, config in enumerate(configs, start=1))
        for index, config, summary, failure in completed:
            print(f"[{suite} {index}/{len(configs)}] {config.dataset_name} {config.run_name}")
            if summary is not None:
                summaries.append(summary)
                pd.DataFrame(summaries).to_csv(summary_path, index=False)
            if failure is not None:
                failure["Suite"] = suite
                failures.append(failure)
                pd.DataFrame(failures).to_csv(failure_path, index=False)
                print(f"FAILED [{suite} {index}/{len(configs)}] {config.run_name}: {failure['Error_type']}: {failure['Error']}")
                if not args.continue_on_error:
                    raise RuntimeError(str(failure["Error"]))
    else:
        with ProcessPoolExecutor(max_workers=min(args.parallel_runs, len(configs))) as pool:
            futures = {pool.submit(run_config, config): (index, config) for index, config in enumerate(configs, start=1)}
            for future in as_completed(futures):
                index, config = futures[future]
                print(f"[{suite} {index}/{len(configs)}] completed {config.dataset_name} {config.run_name}")
                summary, failure = future.result()
                if summary is not None:
                    summaries.append(summary)
                    pd.DataFrame(summaries).to_csv(summary_path, index=False)
                if failure is not None:
                    failure["Suite"] = suite
                    failures.append(failure)
                    pd.DataFrame(failures).to_csv(failure_path, index=False)
                    print(f"FAILED [{suite} {index}/{len(configs)}] {config.run_name}: {failure['Error_type']}: {failure['Error']}")
                    if not args.continue_on_error:
                        for remaining in futures:
                            remaining.cancel()
                        raise RuntimeError(str(failure["Error"]))
    print(f"Suite summary written to {summary_path}")
    if failures:
        print(f"Suite failures written to {failure_path}")


def main() -> None:
    args = parse_args()
    if args.parallel_runs < 1:
        raise ValueError("--parallel-runs must be at least 1")
    datasets = selected_datasets(args)
    for suite in selected_suites(args):
        run_suite(suite, datasets, args)


if __name__ == "__main__":
    main()
