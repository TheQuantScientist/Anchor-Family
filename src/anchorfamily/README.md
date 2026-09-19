# AnchorFamily Package

Active project code outside the vendored benchmark is intentionally small:

- `experiments/anchor_baseline.py`: anchor-family forecasting with upstream benchmark data splits and metrics.
- `benchmark.py`: path helper for importing vendored benchmark modules without changing directories.

Run from the repository root:

```bash
python scripts/experiments/run_anchor_baseline.py --dataset P12
```
