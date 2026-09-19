"""CPU synthetic stress tests showing where univariate anchors should fail."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chronolm.experiments.anchor_baseline import (  # noqa: E402
    AUTO_ANCHOR_CANDIDATES,
    AUTO_ANCHOR_BY_NAME,
    AnchorCandidate,
    candidate_forecast_scaled,
    fit_beta,
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "anchor_results_synthetic"
METHODS = ["NaiveAnchor", "ExpoAnchor", "ERMAnchor", "AutoAnchor", "Oracle"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic synthetic Anchor stress tests.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--seq-len", type=int, default=36)
    parser.add_argument("--pred-len", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args()


def make_sample(history: np.ndarray, target: np.ndarray) -> dict[str, object]:
    return {
        "history_times": [float(index) for index in range(len(history))],
        "history_scaled": [float(value) for value in history],
        "target_times": [float(len(history) + index) for index in range(len(target))],
        "actual_scaled": [float(value) for value in target],
    }


def stable_local(rng: np.random.Generator, seq_len: int, pred_len: int) -> dict[str, object]:
    values = np.cumsum(rng.normal(0.0, 0.08, size=seq_len + pred_len))
    return {"sample": make_sample(values[:seq_len], values[seq_len:]), "oracle": values[seq_len:]}


def regime_shift(rng: np.random.Generator, seq_len: int, pred_len: int) -> dict[str, object]:
    level = rng.normal(0.0, 0.2)
    history = level + rng.normal(0.0, 0.08, size=seq_len)
    shift = rng.choice([-1.0, 1.0]) * rng.uniform(2.0, 3.0)
    target = level + shift + rng.normal(0.0, 0.08, size=pred_len)
    return {"sample": make_sample(history, target), "oracle": target}


def delayed_effect(rng: np.random.Generator, seq_len: int, pred_len: int) -> dict[str, object]:
    x_history = rng.normal(0.0, 1.0, size=seq_len)
    history = rng.normal(0.0, 0.05, size=seq_len)
    delayed_driver = float(np.mean(x_history[-6:]))
    target = 1.8 * delayed_driver + rng.normal(0.0, 0.05, size=pred_len)
    return {"sample": make_sample(history, target), "oracle": target}


def phase_break(rng: np.random.Generator, seq_len: int, pred_len: int) -> dict[str, object]:
    times = np.arange(seq_len + pred_len, dtype=float)
    phase = rng.uniform(0.0, 2 * np.pi)
    history = np.sin(2 * np.pi * times[:seq_len] / 12.0 + phase) + rng.normal(0.0, 0.05, seq_len)
    target = -np.sin(2 * np.pi * times[seq_len:] / 12.0 + phase) + rng.normal(0.0, 0.05, pred_len)
    return {"sample": make_sample(history, target), "oracle": target}


SCENARIOS = {
    "stable_local": stable_local,
    "regime_shift": regime_shift,
    "delayed_cross_variable_effect": delayed_effect,
    "phase_break": phase_break,
}


def generate_samples(
    scenario: str,
    count: int,
    seq_len: int,
    pred_len: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    samples: list[dict[str, object]] = []
    oracles: list[np.ndarray] = []
    maker = SCENARIOS[scenario]
    for _ in range(count):
        output = maker(rng, seq_len, pred_len)
        samples.append(output["sample"])
        oracles.append(np.asarray(output["oracle"], dtype=float))
    return samples, oracles


def candidate_predictions(candidate: AnchorCandidate, sample: dict[str, object], beta: float = 1.0) -> np.ndarray:
    values = candidate_forecast_scaled(
        candidate=candidate,
        history_times=sample["history_times"],
        history_scaled=sample["history_scaled"],
        target_times=sample["target_times"],
        lookback_span=float(len(sample["history_scaled"])),
    )
    return beta * np.asarray(values, dtype=float)


def collect_candidate_values(samples: list[dict[str, object]], candidate: AnchorCandidate) -> tuple[list[float], list[float]]:
    preds: list[float] = []
    targets: list[float] = []
    for sample in samples:
        preds.extend(candidate_predictions(candidate, sample))
        targets.extend(sample["actual_scaled"])
    return preds, targets


def score_predictions(samples: list[dict[str, object]], predictions: list[np.ndarray]) -> tuple[float, float]:
    actual = np.concatenate([np.asarray(sample["actual_scaled"], dtype=float) for sample in samples])
    predicted = np.concatenate(predictions)
    residual = actual - predicted
    return float(np.mean(np.abs(residual))), float(np.mean(residual**2))


def select_erm_candidate(samples: list[dict[str, object]]) -> tuple[AnchorCandidate, float]:
    best: tuple[float, float, str, AnchorCandidate, float] | None = None
    for candidate in AUTO_ANCHOR_CANDIDATES:
        base_values, target_values = collect_candidate_values(samples, candidate)
        beta_options = [1.0]
        if base_values:
            fitted = fit_beta(base_values, target_values)
            if abs(fitted - 1.0) > 1e-9:
                beta_options.append(fitted)
        for beta in beta_options:
            predictions = [candidate_predictions(candidate, sample, beta) for sample in samples]
            mae, mse = score_predictions(samples, predictions)
            key = (mse, mae, candidate.name, candidate, beta)
            if best is None or key[:3] < best[:3]:
                best = key
    assert best is not None
    return best[3], best[4]


def evaluate_scenario(
    scenario: str,
    n_train: int,
    n_test: int,
    seq_len: int,
    pred_len: int,
    seed: int,
) -> list[dict[str, object]]:
    train, _train_oracles = generate_samples(scenario, n_train, seq_len, pred_len, seed)
    test, test_oracles = generate_samples(scenario, n_test, seq_len, pred_len, seed + 1000)
    erm_candidate, erm_beta = select_erm_candidate(train)
    fixed = {
        "NaiveAnchor": (AUTO_ANCHOR_BY_NAME["last"], 1.0),
        "ExpoAnchor": (AUTO_ANCHOR_BY_NAME["ema03"], 1.0),
        "ERMAnchor": (erm_candidate, erm_beta),
        "AutoAnchor": (erm_candidate, erm_beta),
    }

    rows: list[dict[str, object]] = []
    for method in METHODS:
        if method == "Oracle":
            predictions = test_oracles
            anchor_rule = "scenario_oracle"
            beta = 1.0
        else:
            candidate, beta = fixed[method]
            predictions = [candidate_predictions(candidate, sample, beta) for sample in test]
            anchor_rule = candidate.name
        mae, mse = score_predictions(test, predictions)
        rows.append(
            {
                "Scenario": scenario,
                "Method": method,
                "Anchor_rule": anchor_rule,
                "Beta": beta,
                "MAE_scaled": mae,
                "MSE_scaled": mse,
                "Predictions": n_test * pred_len,
            }
        )
    return rows


def markdown(frame: pd.DataFrame) -> str:
    lines = [
        "# Anchor Synthetic Stress Tests",
        "",
        "These CPU-only tests intentionally include regimes where univariate history anchors should fail.",
        "",
        "| Scenario | Method | Rule | MAE | MSE |",
        "|---|---|---|---:|---:|",
    ]
    for _, row in frame.iterrows():
        lines.append(
            "| "
            f"{row['Scenario']} | {row['Method']} | {row['Anchor_rule']} | "
            f"{row['MAE_scaled']:.4f} | {row['MSE_scaled']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for offset, scenario in enumerate(SCENARIOS):
        rows.extend(
            evaluate_scenario(
                scenario=scenario,
                n_train=args.n_train,
                n_test=args.n_test,
                seq_len=args.seq_len,
                pred_len=args.pred_len,
                seed=args.seed + offset * 17,
            )
        )
    frame = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "anchor_synthetic_stress.csv"
    md_path = args.output_dir / "anchor_synthetic_stress.md"
    frame.to_csv(csv_path, index=False)
    md_path.write_text(markdown(frame), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
