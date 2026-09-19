#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN="${CONDA_BIN:-/home/nckh2/miniconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-mimic37}"
RAW_DIR="${RAW_DIR:-/home/nckh2/qa/ChronoLM/APN/data/physionet.org/files/mimiciii/1.4}"
PYOMNITS_DIR="${PYOMNITS_DIR:-/home/nckh2/qa/PyOmniTS}"
TARGET_DIR="${TARGET_DIR:-$HOME/.tsdm/rawdata/MIMIC_III_DeBrouwer2019}"
EXPECTED_SHA="8106f64292771956f70ccd0ca1a4f7a0a4563fe63d6eff6ee2ef27dc6fdb614a"
EXPECTED_ROWS="3082224"
EXPECTED_COLS="7"

if [[ ! -x "$CONDA_BIN" ]]; then
  echo "conda not found at $CONDA_BIN" >&2
  exit 1
fi

if [[ ! -d "$PYOMNITS_DIR/data/dependencies/MIMIC_III/preprocess" ]]; then
  echo "PyOmniTS preprocessing scripts not found under $PYOMNITS_DIR" >&2
  exit 1
fi

if [[ ! -f "$RAW_DIR/ADMISSIONS.csv.gz" ]]; then
  echo "Raw MIMIC-III files not found under $RAW_DIR" >&2
  exit 1
fi

if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  "$CONDA_BIN" create -n "$ENV_NAME" python=3.7 -y
fi

"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python -m pip install numpy==1.21.6 pandas==1.3.5
mkdir -p "$TARGET_DIR"

PREPROCESS_DIR="$PYOMNITS_DIR/data/dependencies/MIMIC_III/preprocess"
for step in 1_Admissions.py 2_Outputs.py 3_LabEvents.py 4_Prescriptions.py 5_DataMerging.py; do
  echo "Running $PREPROCESS_DIR/$step"
  "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python "$PREPROCESS_DIR/$step" "$RAW_DIR/"
done

"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python -c "import hashlib; import pandas as pd; from pathlib import Path; path = Path(r'$TARGET_DIR') / 'complete_tensor.csv'; assert path.exists(), f'missing {path}'; df = pd.read_csv(path, index_col=0); print('complete_tensor shape:', df.shape); assert df.shape == (int('$EXPECTED_ROWS'), int('$EXPECTED_COLS')), f'bad shape {df.shape}'; sha = hashlib.sha256(path.read_bytes()).hexdigest(); print('complete_tensor sha256:', sha); assert sha == '$EXPECTED_SHA', f'bad sha256 {sha}'; print('MIMIC-III De Brouwer preprocessing verified.')"
