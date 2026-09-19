"""Run the remaining paper studies from existing checkpoints and predictions.

This driver never trains a neural model. It evaluates saved neural checkpoints,
runs the requested deterministic Anchor analyses, and builds derived tables.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run remaining evaluation-only studies for the Anchor paper."
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--parallel-neural",
        type=int,
        default=3,
        help="Checkpoint groups evaluated concurrently on one GPU.",
    )
    parser.add_argument(
        "--parallel-anchor",
        type=int,
        default=4,
        help="Independent AutoAnchor conditions evaluated concurrently on CPU.",
    )
    parser.add_argument(
        "--eval-batch-multiplier",
        type=int,
        default=4,
        help="Multiplier applied to each model's training-time batch size.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--neural-itr", type=int, default=3)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--n-permutations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "cross_baseline_results",
    )
    parser.add_argument(
        "--mimic-output-parent",
        type=Path,
        default=PROJECT_ROOT / "reruns/mimic_debrouwer",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def printable(command: list[str]) -> str:
    return shlex.join(command)


def run_step(name: str, command: list[str], env: dict[str, str], dry_run: bool) -> None:
    print(f"\n[{name}]\n{printable(command)}", flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def main() -> None:
    args = parse_args()
    if args.parallel_anchor < 1 or args.parallel_neural < 1 or args.eval_batch_multiplier < 1:
        raise SystemExit("Parallelism and batch multiplier must be positive.")

    python = sys.executable
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    cross_command = [
        python,
        str(PROJECT_ROOT / "scripts/experiments/run_cross_baseline_suite.py"),
        "--suite",
        "temporal",
        "--suite",
        "sparsity",
        "--suite",
        "paired",
        "--dataset",
        "all",
        "--model",
        "all",
        "--evaluation-only",
        "--skip-existing-neural-eval",
        "--gpu-id",
        str(args.gpu_id),
        "--parallel-neural",
        str(args.parallel_neural),
        "--parallel-anchor",
        str(args.parallel_anchor),
        "--eval-batch-multiplier",
        str(args.eval_batch_multiplier),
        "--num-workers",
        str(args.num_workers),
        "--neural-itr",
        str(args.neural_itr),
        "--history-perturb-seed",
        str(args.seed),
        "--output-root",
        str(args.output_root),
    ]
    run_step("cross-model perturbation, sparsity, and prediction export", cross_command, env, args.dry_run)

    mimic_candidate_command = [
        python,
        str(PROJECT_ROOT / "scripts/experiments/run_anchor_cpu_suite.py"),
        "--suite",
        "candidate",
        "--dataset",
        "MIMIC",
        "--output-parent",
        str(args.mimic_output_parent),
        "--random-seed",
        str(args.seed),
        "--parallel-runs",
        str(args.parallel_anchor),
    ]
    run_step("MIMIC De Brouwer candidate-removal ablations", mimic_candidate_command, env, args.dry_run)

    existing_command = [
        python,
        str(PROJECT_ROOT / "ablation/build_existing_evidence_tables.py"),
        "--n-bootstrap",
        str(args.n_bootstrap),
        "--n-permutations",
        str(args.n_permutations),
        "--seed",
        str(args.seed),
    ]
    run_step("existing-result bootstrap and stratified analyses", existing_command, env, args.dry_run)

    paired_command = [
        python,
        str(PROJECT_ROOT / "ablation/paired_autoanchor_neural.py"),
        "--output-dir",
        str(args.output_root / "ablation"),
        "--n-bootstrap",
        str(args.n_bootstrap),
        "--n-permutations",
        str(args.n_permutations),
        "--seed",
        str(args.seed),
    ]
    run_step(
        "paired AutoAnchor-vs-neural significance",
        paired_command,
        env,
        args.dry_run,
    )


if __name__ == "__main__":
    main()
