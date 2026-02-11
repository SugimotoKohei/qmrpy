# Validation Suite

`qmrpy` provides a reproducible validation suite to quantify implementation
consistency across `T1`, `T2`, `T2*`, `B0`, `B1`, `QSM`, and `Simulation` domains.

The suite is designed to run **without external dependencies** (no qMRLab/Octave required)
for the core claims in the JOSS paper.

## Run Core Validation

```bash
uv run --locked -- python scripts/summarize_parity.py \
  --suite core \
  --formats csv,markdown,json \
  --config configs/exp/validation_core.toml \
  --out-dir output/reports/parity_summary
```

Main outputs:

- `output/reports/parity_summary/core_validation.csv`
- `output/reports/parity_summary/core_validation_metrics.csv`
- `output/reports/parity_summary/core_validation.json`
- `output/reports/parity_summary/summary.md`
- `output/reports/parity_summary/summary.json`

## Config Structure

Validation cases and thresholds are defined in:

- `configs/exp/validation_core.toml`

This file includes:

- Global seed (`[meta].seed`) for deterministic runs
- Default sample count (`[run].n_samples`)
- Per-model case settings (`[core.<model>]`)
- Thresholds used for pass/fail decision

## Output Semantics

`core_validation.csv`:

- One row per validation case
- Includes primary metric, threshold, and pass/fail (`pass` = `0` or `1`)

`core_validation_metrics.csv`:

- One row per metric
- Includes threshold and pass/fail (`pass` = `0` or `1`)

Pass/fail rule:

- A case passes only when **all metrics** in that case satisfy `value <= threshold`.

## Threshold Rationale

Current thresholds are practical acceptance limits aligned with existing test behavior:

- Tight enough to catch regressions in fitting/simulation consistency
- Loose enough to avoid brittle failures from controlled synthetic noise

When updating thresholds, adjust `configs/exp/validation_core.toml` and keep
the rationale in sync with this guide and `paper.md`.

## Additional Suites

The same CLI supports additional suites:

```bash
# DECAES fixed-vector parity
uv run --locked -- python scripts/summarize_parity.py --suite decaes --formats csv,json

# Collect pre-generated qMRLab parity reports (if present)
uv run --locked -- python scripts/summarize_parity.py --suite qmrlab --formats csv,json
```

`qmrlab` suite only summarizes existing `report.json` files and does not require
Octave unless you generate those reports separately.

## Correction Effect Report (T1/T2/T2* + B1/B0)

To quantify correction gain (`without correction` vs `with B1/B0 correction`) on synthetic data:

```bash
uv run --locked -- python scripts/report_b0_b1_correction_effect.py \
  --seed 20260211 \
  --n-samples 300 \
  --json-out output/reports/b0_b1_correction_report.json
```

Output:

- `output/reports/b0_b1_correction_report.json`

This report contains:

- median relative error for `T1`, `T2`, `T2*` (uncorrected/corrected)
- `B1` MAE and `B0` RMSE
- fraction of samples improved by correction
- threshold checks aligned with the roadmap acceptance targets
