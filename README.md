<div align="center">

<h1>⚓ AnchorFamily</h1>

<p>
  <strong>How far can simple history-derived anchors go in irregular multivariate time-series forecasting?</strong>
</p>

<p>
  A compact and reproducible benchmark workspace for evaluating transparent,
  gradient-free forecasting controls under matched neural benchmark protocols.
</p>

<p>
  <img src="https://img.shields.io/badge/Research-IMTS%20Forecasting-4B5563?style=flat-square" />
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.6-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Training-Gradient--Free-6B7280?style=flat-square" />
  <img src="https://img.shields.io/badge/Benchmark-Reproducible-111827?style=flat-square" />
</p>

<p>
  <a href="#overview">Overview</a> •
  <a href="#repository-structure">Structure</a> •
  <a href="#setup">Setup</a> •
  <a href="#running-anchorfamily">Experiments</a> •
  <a href="#reproducing-neural-baselines">Neural Baselines</a> •
  <a href="#reproducibility">Reproducibility</a>
</p>

</div>

---

## Overview

**AnchorFamily** asks a deliberately simple question:

> **How much of current irregular multivariate time-series forecasting performance can be explained by structure already present in the observed history?**

The repository implements a family of transparent, history-derived forecasting controls and evaluates them using the **same upstream data loaders, dataset splits, forecasting windows, scaling conventions, and metrics** used by the corresponding neural benchmark.

AnchorFamily is intended as a **diagnostic benchmark companion** rather than a replacement benchmark implementation.

The core implementation lives in:

```text
src/anchorfamily/
scripts/
```

The original benchmark implementation required for reproducibility is vendored under:

```text
vendor/upstream_benchmark/
```

Vendored files are explicitly marked in repository metadata so that GitHub presents the project as **AnchorFamily**, rather than as an upstream benchmark fork.

---

## Repository Structure

```text
AnchorFamily/
│
├── src/
│   └── anchorfamily/
│       └── experiments/
│           └── anchor_baseline.py
│
├── scripts/
│   ├── experiments/
│   ├── check_repo_ready.py
│   └── compute_global_metrics.py
│
├── vendor/
│   └── upstream_benchmark/
│
├── anchor_results/          # generated
└── upstream_model_runs/     # generated
```

| Path | Description |
| :--- | :--- |
| `src/anchorfamily/` | Core AnchorFamily implementation. |
| `src/anchorfamily/experiments/anchor_baseline.py` | Anchor-family evaluation pipeline using benchmark-compatible loaders and metrics. |
| `scripts/experiments/` | Experiment orchestration entry points. |
| `scripts/check_repo_ready.py` | Repository hygiene check before pushing. |
| `scripts/compute_global_metrics.py` | Recomputes benchmark-style global MAE/MSE from detailed logs. |
| `vendor/upstream_benchmark/` | Vendored upstream benchmark required for reproducibility. |

Generated results, logs, checkpoints, datasets, environments, manuscript files, reference material, and checkpoint arrays are excluded from version control.

Before pushing:

```bash
python scripts/check_repo_ready.py
```

The check fails if Git can see a tracked or non-ignored file larger than **25 MB**.

---

## Setup

### Environment

The benchmark-compatible environment uses:

```text
Python   3.11.13
PyTorch  2.6.0+cu124
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -e .[apn]
```

Alternatively, install using the pinned compatibility file:

```bash
pip install -r requirements.txt
pip install -e .
```

---

## Datasets

Public datasets are prepared automatically by the benchmark loaders on first use.

| Dataset | Default Location |
| :--- | :--- |
| P12 / PhysioNet 2012 | `~/.tsdm/` |
| USHCN | `~/.tsdm/` |
| HumanActivity | `vendor/upstream_benchmark/storage/datasets/HumanActivity` |

### MIMIC-III

MIMIC-III requires credentialed access and is therefore not downloaded automatically.

Follow the preparation instructions in the vendored benchmark README and place:

```text
complete_tensor.csv
```

under:

```text
~/.tsdm/rawdata/MIMIC_III_DeBrouwer2019/
```

---

## Running AnchorFamily

### AutoAnchor

Run the headline **AutoAnchor** method on all supported datasets:

```bash
python scripts/experiments/run_anchor_baseline.py --all
```

Run a single dataset:

```bash
python scripts/experiments/run_anchor_baseline.py --dataset P12
python scripts/experiments/run_anchor_baseline.py --dataset USHCN
python scripts/experiments/run_anchor_baseline.py --dataset HumanActivity
```

---

### Full Anchor Family

Run all Anchor variants:

```bash
python scripts/experiments/run_anchor_baseline.py --all --family
```

Run selected family members:

```bash
python scripts/experiments/run_anchor_baseline.py \
  --dataset USHCN \
  --method NaiveAnchor \
  --method AutoAnchor
```

```bash
python scripts/experiments/run_anchor_baseline.py \
  --dataset P12 \
  --method ERMAnchor
```

---

### Smoke Test

For a lightweight end-to-end sanity check:

```bash
python scripts/experiments/run_anchor_baseline.py \
  --dataset USHCN \
  --max-test-samples 20 \
  --trace-every 5
```

---

## Anchor Family

The current family consists of:

```text
NaiveAnchor
ExpoAnchor
SparseAnchor
ERMAnchor
AutoAnchor
```

`AutoAnchor` operates over a **shared candidate library** across supported datasets and variables.

Dataset-specific logic is restricted to benchmark-facing configuration:

- data loading,
- split selection, and
- benchmark forecasting windows.

Candidate selection is performed using benchmark **training and validation targets**.

When an observed history exhibits a strong generic structural signature—such as sparse mode dominance, long-context short-horizon dynamics, or seasonal phase structure—a deterministic history-only prior may select the corresponding candidate.

> **Test labels are reserved exclusively for final evaluation.**

For every variable, the calibration output records the selected family member, candidate, fitted parameters, selection source, and rationale.

This design keeps AnchorFamily auditable while minimizing dataset-specific forecasting logic.

---

## Outputs

Results are written to:

```text
anchor_results/<dataset>/
```

| Artifact | Description |
| :--- | :--- |
| `*_Calibration.csv` | Selected Anchor rule, fitted shrinkage, source, and rationale per variable. |
| `*_Results.csv` | Per-variable scaled MAE and MSE. |
| `*_Checkpoint.csv` | Resume state for interrupted experiments. |
| `*_DetailLog.csv` | Per-target ground truth, prediction, and Anchor values. |
| `*_debug.log` | Compact execution and progress trace. |
| `anchor_results/anchor_summary.csv` | Global benchmark summary across completed datasets. |

---

## Global Metric Computation

Benchmark-style global metrics can be reconstructed directly from a detail log:

```bash
python scripts/compute_global_metrics.py \
  anchor_results/p12/anchor_family_auto_anchor_p12_P12_DetailLog.csv
```

This computes global scaled MAE/MSE using the same aggregation convention as the benchmark evaluation pipeline.

---

## Reproducing Neural Baselines

AnchorFamily includes an orchestration wrapper for executing the original upstream paper models **without rewriting their experiment definitions**.

### Inspect Available Runs

```bash
python scripts/experiments/run_upstream_paper_models.py --list
```

### Dry Run

Inspect the exact upstream commands before execution:

```bash
python scripts/experiments/run_upstream_paper_models.py \
  --model all \
  --dataset P12
```

### Execute

```bash
python scripts/experiments/run_upstream_paper_models.py \
  --model all \
  --dataset P12 \
  --execute
```

Run all paper-table models across the public datasets:

```bash
python scripts/experiments/run_upstream_paper_models.py \
  --dataset HumanActivity,USHCN,P12 \
  --execute \
  --continue-on-error
```

The wrapper intentionally does **not** modify upstream:

```text
hyperparameters
model configurations
GPU identifiers
experiment scripts
```

Each upstream experiment is executed with:

```text
cwd=vendor/upstream_benchmark
```

This preserves the original relative paths and benchmark assumptions.

Upstream model logs remain in the vendored benchmark workspace, while orchestration logs are written to:

```text
upstream_model_runs/
```

---

## Reproducibility

AnchorFamily is designed around a simple principle:

> **Change the forecasting rule, not the benchmark around it.**

Wherever possible, the repository preserves the upstream:

| Component | Preserved |
| :--- | :---: |
| Dataset preprocessing | ✓ |
| Train / validation / test splits | ✓ |
| Observation windows | ✓ |
| Forecasting horizons | ✓ |
| Scaling conventions | ✓ |
| Evaluation metrics | ✓ |
| Neural baseline execution paths | ✓ |

The objective is to isolate the contribution of the forecasting mechanism while holding the surrounding evaluation protocol fixed.

This makes AnchorFamily useful as a diagnostic control for determining whether benchmark performance genuinely requires learned irregular temporal representations—or can already be explained by transparent structure in the observed history.

---

<div align="center">

### AnchorFamily

<sub>
Transparent controls · Matched protocols · Reproducible evaluation
</sub>

</div>
