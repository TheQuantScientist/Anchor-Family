"""Generate Anchor-family CPU efficiency tables from a completed summary CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = PROJECT_ROOT / "anchor_results_family" / "anchor_summary.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "anchor_results_family" / "ablation"

METHOD_ORDER = [
    "NaiveAnchor",
    "ExpoAnchor",
    "SparseAnchor",
    "ERMAnchor",
    "AutoAnchor",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Anchor-family CPU throughput tables from anchor_summary.csv."
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def efficiency_frame(summary_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(summary_path)
    required = {"Dataset", "Method", "Predictions", "Fallback", "Time_s"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns in summary CSV: {sorted(missing)}")

    frame = frame.copy()
    frame["Method"] = pd.Categorical(frame["Method"], categories=METHOD_ORDER, ordered=True)
    frame = frame.dropna(subset=["Method"]).sort_values(["Dataset", "Method"])
    frame["Predictions"] = frame["Predictions"].astype(float)
    frame["Fallback"] = frame["Fallback"].astype(float)
    frame["CPU_Total_s"] = frame["Time_s"].astype(float)
    frame["CPU_ms_per_pred"] = 1000.0 * frame["CPU_Total_s"] / frame["Predictions"]
    frame["CPU_pred_per_s"] = frame["Predictions"] / frame["CPU_Total_s"]
    return frame


def markdown_efficiency(frame: pd.DataFrame) -> str:
    lines = [
        "# Anchor Family CPU Efficiency",
        "",
        "CPU total includes calibration plus forecasting wall time from `anchor_summary.csv`.",
        "",
        "| Dataset | Method | Predictions | Fallback | CPU total (s) | ms/pred | pred/s |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in frame.iterrows():
        lines.append(
            "| "
            f"{row['Dataset']} | {row['Method']} | "
            f"{int(round(row['Predictions']))} | "
            f"{int(round(row['Fallback']))} | "
            f"{row['CPU_Total_s']:.4f} | "
            f"{row['CPU_ms_per_pred']:.4f} | "
            f"{row['CPU_pred_per_s']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def write_outputs(frame: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "Dataset",
        "Method",
        "Predictions",
        "Fallback",
        "CPU_Total_s",
        "CPU_ms_per_pred",
        "CPU_pred_per_s",
    ]
    csv_path = output_dir / "anchor_family_efficiency.csv"
    md_path = output_dir / "anchor_family_efficiency.md"
    frame[columns].to_csv(csv_path, index=False)
    md_path.write_text(markdown_efficiency(frame), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def main() -> None:
    args = parse_args()
    write_outputs(efficiency_frame(args.summary), args.output_dir)


if __name__ == "__main__":
    main()
