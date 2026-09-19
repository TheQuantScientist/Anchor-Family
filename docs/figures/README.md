# Anchor family architecture — revised figure

The figure presents **one candidate space and five alternative selection
policies**. AutoAnchor extends empirical selection with an ordered structural
prior. This relationship is the main panel; a second panel illustrates how
any of the five policies produces a forecast through the same operator.

The drawing uses a restrained serif design, thin rules, and one blue accent
for AutoAnchor and the selected forecast. Its PDF/SVG are entirely vector.
The high-resolution PNG is 4800 × 4000 pixels at 600 dpi.

## Update the existing Overleaf project

Replace **both** files in your Overleaf `figures/` folder:

- `anchor_family_architecture.pdf`
- `anchor_family_figure.tex`

The second file supplies the rewritten caption for the new two-panel layout.
Keep the same input command as before:

```tex
% Preamble:
\usepackage{graphicx}
\newcommand{\anchorfiguredir}{figures}

% Inside Proposed Architecture:
\input{figures/anchor_family_figure.tex}
```

Do not duplicate preamble commands that are already present. The reference
label remains `fig:anchor-family-architecture`.

When compiling locally from `docs/paper/`, the fragment's default image path
already points to `../figures/`; use
`\input{../figures/anchor_family_figure.tex}`. For a two-column template, use
`figure*` instead of `figure` in the fragment to span both columns.

## Exports and editable sources

- `anchor_family_architecture.pdf` — paper asset; vector with embedded fonts.
- `anchor_family_architecture.svg` — editable vector geometry and text.
- `anchor_family_architecture.png` — 600 dpi export, 4800 × 4000 pixels.
- `anchor_family_architecture_preview.png` — smaller preview for browsing.
- `anchor_family_architecture_print_proof.png` — intended 5.5-inch width at 180 dpi.
- `anchor_family_figure.tex` — figure environment and updated caption.
- `anchor_family_preview.tex` — standalone LaTeX inclusion example.
- `draw_anchor_family.py` — reproducible drawing source.
- `anchor_family_source_map.json` — source hash and implementation references.
- `previous_versions/v1/` — preserved original design.

The native figure is 8.00 × 6.67 inches. At a 5.5-inch paper width it is
approximately 4.58 inches tall. The smallest main label is approximately
7.2 pt at that width; body labels are generally 7.5–9.6 pt. Use full text width.
PDF resolution is independent of the PNG export resolution.

## Reading the new figure

**(a) Selection policies.** Rows are alternative methods. Every method chooses
its own specification `theta_v = (a_v, beta_v)` per variable. The first three
rows expose fixed or prescribed rules. The ERM row searches the common candidate
space. The AutoAnchor row reuses the sparse and EMA rules within a broader
ordered prior; an explicit arrow from ERM shows its empirical fallback. The
blue prior arrow carries beta = 1, making the override semantics visible.

**(b) Shared forecast operator.** Gray curves illustrate alternatives available
in the candidate library, while the blue curve shows one fixed candidate.
The selected candidate alone is evaluated for each new test history; deployment
does not rescore the entire library. Filled dots are history observations;
open dots are the selected predictions at query times. The dashed vertical line
separates observed history from the forecast interval. The displayed selection
is an illustrative fixed mixture, not a calibration result or benchmark result.

`P_{b,v}` in the equation denotes inverse scaling and recent-history clipping.
The candidate consumes history values in benchmark scale. The caption defines
this notation as well as the candidate-dependent scaling set `B_{a,v}`, the
empirical risk, lookback `L`, and horizon `K`.

## Regenerate or edit

From the repository root:

```sh
/home/nckh2/qa/.venv/bin/python docs/figures/draw_anchor_family.py
```

Any Python environment with NumPy and Matplotlib will work. No LaTeX
installation, APN import, or dataset access is needed. The illustrative
trajectories use the repository's actual candidate functions on synthetic data.
The generator checks label bounds and pairwise text intersections.

Edit typography, coordinates, or content in the Python file and regenerate.
The SVG can also be edited directly in a vector editor. It uses STIXGeneral
text and STIX math; install those fonts when editing to preserve the layout.
The PDF embeds its fonts and needs no font installation. Regeneration overwrites
exports, including manual SVG changes.

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

- Visually inspected both the main preview and the 5.5-inch print proof.
- All 36 labels fit; automated rendered-text checks found no intersections.
- PDF inspection confirmed a single vector page, all five variant names,
  embedded fonts, and no raster images.
- SVG inspection confirmed editable text and no raster images.
- PNG dimensions and its 600 dpi metadata were checked.
- The caption and panel labels now match the redesigned figure. The image and
  fragment retain their filenames and reference label for the existing Overleaf
  integration, which the user has already compiled successfully.

The revised full LaTeX document has not been compiled locally because this
environment has no LaTeX executable or manuscript ICLR style file. The supplied
`anchor_family_preview.tex` remains available for local or Overleaf compilation.
