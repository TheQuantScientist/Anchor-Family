<<<<<<< HEAD
# ChronoLM
=======
# AnchorFamily

AnchorFamily is now a compact benchmark workspace for asking a sharper question:
how far can simple history anchors go on irregular multivariate time-series
forecasting when evaluated with the the same upstream benchmark splits and scaled metrics?

AnchorFamily's code lives under `src/anchorfamily` and `scripts`. The `vendor/upstream_benchmark/` directory
is vendored upstream benchmark code used for reproducible upstream benchmark data loaders,
models, and baseline settings; it is marked as vendored in GitHub metadata so the
repository presents as AnchorFamily rather than as an upstream benchmark fork.

## Layout

| Path | Purpose |
|---|---|
| `src/anchorfamily/` | AnchorFamily package code. |
| `src/anchorfamily/experiments/anchor_baseline.py` | Anchor-family runner using upstream benchmark data loaders and metrics. |
| `scripts/experiments/` | Experiment orchestration entry points. |
| `scripts/check_repo_ready.py` | Pre-push check for large tracked or nonignored files. |
| `scripts/compute_global_metrics.py` | Utility for benchmark-style global MAE/MSE from detail logs. |
| `vendor/upstream_benchmark/` | Vendored upstream benchmark code required for reproducibility. |

Generated logs, CSVs, checkpoints, local datasets, local environments,
reference PDFs, paper drafts, docs, and checkpoint arrays are ignored by git. Before pushing, run:

```bash
python scripts/check_repo_ready.py
```

This fails if any tracked or nonignored file larger than 25 MB is visible to git.

## Setup

The upstream benchmark environment recommends Python 3.11.13 and PyTorch 2.6.0+cu124.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[apn]
# or, if you prefer the pinned compatibility file:
# pip install -r requirements.txt && pip install -e .
```

Public datasets are prepared by the benchmark loaders on first use:

| Dataset | Cache |
|---|---|
| P12 / PhysioNet 2012 | `~/.tsdm/` |
| USHCN | `~/.tsdm/` |
| HumanActivity | `vendor/upstream_benchmark/storage/datasets/HumanActivity` |

MIMIC requires credentialed access. Follow the vendored benchmark README and place
`complete_tensor.csv` under `~/.tsdm/rawdata/MIMIC_III_DeBrouwer2019/`.

## Run Anchor Baselines

Run the headline AutoAnchor method on all currently supported anchor datasets:

```bash
python scripts/experiments/run_anchor_baseline.py --all
```

Run one dataset:

```bash
python scripts/experiments/run_anchor_baseline.py --dataset P12
python scripts/experiments/run_anchor_baseline.py --dataset USHCN
python scripts/experiments/run_anchor_baseline.py --dataset HumanActivity
```

Run the full anchor family for ablation:

```bash
python scripts/experiments/run_anchor_baseline.py --all --family
```

Run selected family members:

```bash
python scripts/experiments/run_anchor_baseline.py --dataset USHCN --method NaiveAnchor --method AutoAnchor
python scripts/experiments/run_anchor_baseline.py --dataset P12 --method ERMAnchor
```

Cheap smoke test:

```bash
python scripts/experiments/run_anchor_baseline.py --dataset USHCN --max-test-samples 20 --trace-every 5
```

Outputs go under `anchor_results/<dataset>/` and include:

| File | Purpose |
|---|---|
| `*_Calibration.csv` | Per-variable anchor method, fitted shrinkage, source, and rationale. |
| `*_Results.csv` | Per-variable scaled MAE/MSE. |
| `*_Checkpoint.csv` | Per-variable resume checkpoint. |
| `*_DetailLog.csv` | Per-target actual, prediction, and anchor values. |
| `*_debug.log` | Progress and compact trace logging. |
| `anchor_results/anchor_summary.csv` | benchmark-style global summary across completed datasets. |

The anchor family contains `NaiveAnchor`, `ExpoAnchor`, `SparseAnchor`,
`ERMAnchor`, and `AutoAnchor`. AutoAnchor uses one shared candidate library for
every supported dataset and variable. Dataset-specific code is limited to upstream benchmark
data loading, split choice, and benchmark window sizes. The selector first
evaluates candidates on benchmark train+validation targets; when the observed
histories have a strong generic signature, such as sparse mode dominance,
long-window short-horizon dynamics, or seasonal phase structure, a
deterministic history-only prior selects the corresponding candidate. Test
labels are used only once for final reporting. The calibration CSV records the
family member, selected candidate, source, and rationale for every variable.

Compute benchmark-style metrics from any detail log:

```bash
python scripts/compute_global_metrics.py anchor_results/p12/anchor_family_auto_anchor_p12_P12_DetailLog.csv
```

## Run Upstream Paper Models

List the upstream paper-script matrix:

```bash
python scripts/experiments/run_upstream_paper_models.py --list
```

Dry-run the exact upstream benchmark commands from the root:

```bash
python scripts/experiments/run_upstream_paper_models.py --model all --dataset P12
```

Execute a script:

```bash
python scripts/experiments/run_upstream_paper_models.py --model all --dataset P12 --execute
```

Run every paper-table model on the public datasets:

```bash
python scripts/experiments/run_upstream_paper_models.py \
  --dataset HumanActivity,USHCN,P12 \
  --execute \
  --continue-on-error
```

The wrapper does not rewrite upstream hyperparameters or GPU ids. It runs each
vendored benchmark script with `cwd=vendor/upstream_benchmark`, creates local
benchmark logs there, and writes wrapper logs under `upstream_model_runs/`.
>>>>>>> AnchorFamily
