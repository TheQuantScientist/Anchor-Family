# Anchor family architecture figure

The figure is drawn with Matplotlib as a vector illustration. LaTeX includes
its PDF through `graphicx`; TikZ, shell escape, and image conversion during
paper compilation are unnecessary. The PNG is a preview, not the paper asset.

## Files

- `anchor_family_architecture.pdf` — vector figure with embedded TrueType fonts.
- `anchor_family_architecture.svg` — editable shapes and text for a vector editor.
- `anchor_family_architecture.png` — 300 dpi preview.
- `anchor_family_architecture_print_proof.png` — 5.5-inch-width proof at 150 dpi.
- `anchor_family_figure.tex` — figure environment, caption, and reference label.
- `anchor_family_preview.tex` — standalone LaTeX inclusion example.
- `draw_anchor_family.py` — editable, reproducible drawing source.
- `anchor_family_source_map.json` — code hash, candidate counts, and function locations.

The main diagram is 7.00 × 6.67 inches natively. At a 5.5-inch paper width it
is 5.24 inches tall. Main text is approximately 6.8–8.9 pt at that width, with
slightly smaller numbers inside the three rule badges. Use the full text width;
the complete diagram is too dense for a narrow two-column column.

## Insert in the existing paper

When compiling from `docs/paper/`, add this under Proposed Architecture:

```tex
% In the preamble (already present in iclr2027_anchor.tex):
\usepackage{graphicx}

% In the Proposed Architecture section:
\input{../figures/anchor_family_figure.tex}
```

For Overleaf, upload `anchor_family_architecture.pdf` and
`anchor_family_figure.tex` to a `figures/` folder, then use:

```tex
% In the preamble:
\usepackage{graphicx}
\newcommand{\anchorfiguredir}{figures}

% In the Proposed Architecture section:
\input{figures/anchor_family_figure.tex}
```

Reference the figure with `Figure~\ref{fig:anchor-family-architecture}`.
The existing manuscript and old TikZ file are left intact. For a two-column
paper, change the fragment's `figure` environment to `figure*`.

For just the image, the minimal insertion is:

```tex
\includegraphics[width=\linewidth]{figures/anchor_family_architecture.pdf}
```

## Regenerate

From the repository root, using the existing workspace environment:

```sh
/home/nckh2/qa/.venv/bin/python docs/figures/draw_anchor_family.py
```

Any Python environment with NumPy and Matplotlib will work. No model run or
dataset access is needed. Miniature candidate plots use a synthetic history and
the repository's actual pure forecast functions, loaded without importing APN.
The script checks that labels stay within their specified boxes and canvas.

Edit coordinates, labels, and palette in the Python file for reproducible
changes. Alternatively, edit the SVG in Inkscape or another vector editor and
export a PDF. The SVG retains text; use DejaVu Sans to preserve its layout.
Regeneration replaces exported assets, including manual SVG changes.

## What the diagram means

**(a) Calibration.** The detailed branch depicts default AutoAnchor. Every
variable has its own selected `(candidate, beta)`. Selection happens once on
pooled train and validation samples, before test evaluation. The selected
candidate's numerical output still depends on each new sample's history.
ERMAnchor is the same minimum-risk selector before the structural override.

**(b) Forecasting.** Each variant evaluates its fixed candidate at the requested
future times using the current observed history. It applies the scale
coefficient, converts to raw units, and clips using that history. A solid arrow
is a computation or input; the dashed arrow passes the frozen specification.
The final miniature forecast is schematic, not an experimental result.

**(c) Variants.** The five cards are alternative methods sharing panel (b), not
five modules composed in series. NaiveAnchor fixes the last-value rule;
ExpoAnchor fixes an EMA using window metadata; SparseAnchor chooses among its
sparse rules; ERMAnchor searches the library; AutoAnchor adds the ordered prior.

## Implementation correspondence

All references below are to `src/chronolm/experiments/anchor_baseline.py`.
Exact line numbers and the source SHA-256 are in `anchor_family_source_map.json`.

| Figure element | Implementation | Detail |
| --- | --- | --- |
| Shared library | `build_auto_anchor_candidates` | 85 candidates: 26 primitive rules, 40 two-component mixtures, 19 three-component mixtures. |
| Primitive rules | `component_forecast_scaled` | Last; recent means over 2/3/5/8 points; trimmed means over 3/5/8; mode; nine EMA coefficients; four trend windows; four phase configurations. |
| Fixed mixtures | `candidate_forecast_scaled` | Prespecified convex weights, not learned mixture weights. |
| ERM score | `select_erm_spec`, `score_candidate_on_dataset`, `anchor_predictions_scaled` | Pooled masked scaled MSE after inverse scaling, recent-history clipping, and re-scaling. |
| Candidate scale | `candidate_beta_options`, `fit_beta` | For each candidate compare beta = 1 against a closed-form coefficient clipped to [0, 1.25]. |
| Prior statistics | `variable_history_statistics` | Aggregate rounded raw-value mode frequency and cardinality; mean absolute disagreements among phase, trend, and last forecasts at calibration query times. |
| Ordered prior | `choose_structural_prior_method` | Repeated-value rules first, then long-window/short-horizon EMA, then window-gated phase mixtures. |
| Override | `build_family_spec` | If the prior returns an available candidate, replace the ERM specification and reset beta to 1. Otherwise retain ERM. |
| One specification per variable | `calibrate_anchor_family` | `calibration_samples = fit_samples + validation_samples`; store an `AnchorSpec` for each variable. |
| Inference | `build_forecast_anchors`, `run` | Evaluate the candidate, multiply by beta, inverse-scale, round raw anchors to four decimals, then clip. |
| Final bound | `recent_history_bounds` | Use the last up to 10 raw values: mean ± 3 max(std, 0.1); no bound for histories shorter than 3. |

### Exact rules behind the abbreviated prior panel

Let `p` be the pooled frequency of the most common raw value after rounding to
four decimals, `u` the number of distinct rounded raw values, `L = seq_len`, and
`K = pred_len`. The following are evaluated in order, and the first match wins:

1. If `p > 0.95`, choose **mode**.
2. If `p > 0.50` and `u > 150`, choose **mode**.
3. If `p > 0.50`, choose **0.80 last + 0.20 phase033k5**.
4. If `L >= 1000` and `K/L <= 0.20`, choose **EMA** with
   `alpha = clip(round(10 sqrt(K/L))/10, 0.1, 0.9)`.
5. If `L >= 100` and phase diagnostics are available, compare mean absolute
   disagreements. If `gap(phase, trend) <= gap(phase, last)`, choose
   **0.65 phase033k5 + 0.25 trend8 + 0.10 last**; otherwise choose
   **0.25 last + 0.75 phase033k5**.
6. Otherwise retain the ERM choice.

SparseAnchor uses only rules 1–3 and otherwise chooses the last value. Its beta
is always 1. ExpoAnchor uses the same alpha formula as rule 4 for every window
length. NaiveAnchor also has beta = 1.

`phase033k5` uses period `L/3`, at most five neighbors by circular phase distance,
and weights `1/(distance + 0.25)`. The code does not fit a period or run a
statistical seasonality test. `trend8` fits a line to the last up to eight
observations, evaluates it at the query time, and bounds the extrapolation.

### Scope and precision

- The diagram depicts the full, default library and AutoAnchor strategy.
  Candidate-exclusion and `erm_only`/`rules_only` ablations are omitted.
- The prior is based on aggregate calibration histories and query-time
  diagnostics. It does not run again for each test sample and does not use
  future target values. The code uses `seq_len`, not measured history density,
  to gate its long-window and phase branches.
- Beta is called a **scale coefficient** here: its allowed maximum is 1.25,
  so it is not always a shrinkage coefficient. The fitted value is a bounded
  least-squares proposal; selection compares it with 1 after postprocessing.
- The risk minimization shorthand in the image means a finite search over
  each candidate's identity/fitted options, not continuous optimization.
  Ties use calibration MAE, validation MSE, candidate name, and beta source.
- The evaluator masks missing targets and normally keeps variable histories
  with at least three observations. Query times and masks define evaluated
  forecast positions. These protocol details are not additional learned blocks.
- The final inference path rounds the raw anchor to four decimals before
  clipping; calibration scoring does not do this intermediate rounding. The
  figure abstracts that serialization detail but shows the shared mathematical
  operations in their actual order.
- The current manuscript uses the word “dynamic” for the selector in places.
  The implementation freezes the rule per variable; only candidate evaluation
  changes with each input history. Match the architecture prose to this behavior.

## Validation

The generation script checks label bounds and reads the actual candidate
builder, and the generated PNG was visually inspected at intended paper width.
PDF inspection confirmed one page, all five variant names, embedded fonts, and
no raster images. SVG inspection confirmed editable text and no raster images.
See the source map for counts and provenance. The environment does not currently
provide a LaTeX executable or the manuscript's ICLR style file, so the full paper
has not been compiled here. The supplied standalone preview can be compiled in
any ordinary LaTeX installation with `graphicx`, `geometry`, and AMS packages.
