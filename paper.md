---
title: 'qmrpy: A verification-first Python reference implementation for quantitative MRI modeling'
tags:
  - Python
  - MRI
  - quantitative MRI
  - relaxometry
  - T1 mapping
  - T2 mapping
  - myelin water imaging
authors:
  - name: Kohei Sugimoto
    orcid: 0000-0003-2702-5235
    corresponding: true
    affiliation: 1
affiliations:
  - name: Independent Researcher, Japan
    index: 1
date: 15 January 2026
bibliography: paper.bib
---

# Summary

`qmrpy` is a verification-first Python reference implementation for
quantitative MRI (qMRI) modeling, fitting, simulation, and real-data I/O. The
package provides T1, T1rho, magnetization transfer, MRF, T2/T2*, B0/B1, QSM,
noise, and simulation components under a unified API and pairs those
implementations with reproducible validation artifacts generated from fixed
configurations and deterministic seeds.

Rather than framing contribution primarily as "Python reimplementation", `qmrpy`
focuses on making qMRI model behavior auditable and comparable across domains in
one environment. This includes a common model interface (`forward`, `fit`,
`fit_image`), consistent `FitResult` output conventions, optional NIfTI/DICOM
and minimal qMRI-BIDS helpers, a command-line entry point, and a validation
suite that produces machine-readable pass/fail summaries.

# Statement of need

qMRI pipelines are often implemented in ecosystem-specific tools (e.g., MATLAB,
Julia), and many cross-method workflows in Python rely on ad hoc wrappers or
single-model scripts. A key gap is not only language availability but
**verification consistency**: users need a practical way to evaluate whether
multiple model families remain numerically stable under one reproducible setup.

`qmrpy` addresses this gap by providing:

- Cross-domain qMRI models in a single Python package
- A shared API contract for voxel-wise and image-wise fitting
- Real-data I/O helpers that preserve spatial metadata for result-map export
- A lightweight command-line interface for fitting and validation
- Reproducible validation configurations (`configs/exp/validation_core.toml`)
- Scripted summaries (`scripts/summarize_parity.py`) that emit CSV/Markdown/JSON

This targets MRI researchers, neuroscientists, and medical physicists who need
software that is both easy to integrate into Python workflows and defensible in
terms of reproducibility and regression detection.

# State of the field

qMRLab [@Karakuzu2020] provides a comprehensive MATLAB framework for qMRI model
fitting and simulation. DECAES [@Doucette2020] provides a Julia implementation
for multicomponent T2 mapping with regularized NNLS.

`qmrpy` is complementary to these tools. Its core differentiation is a
verification-oriented packaging of multiple qMRI domains in Python with a
single interface and explicit validation outputs. Implemented methods include:

- DESPOT1-style variable flip angle T1 mapping [@Deoni2005]
- Inversion recovery T1 fitting [@Barral2010]
- DESPOT1-HIFI, MP2RAGE, and spin-lock T1rho mapping
- MTR and MTsat magnetization transfer mapping
- Dictionary-based MR fingerprinting for simultaneous T1-T2 estimation
- Mono-exponential, EPG-corrected, EMC, and two-pool water/fat T2 fitting
- Multi-component T2 / myelin water fraction analysis [@MacKay1994]
- B0/B1 mapping and T2* / R2* fitting
- MPPCA denoising [@Veraart2016]
- Split-Bregman-based QSM reconstruction [@Goldstein2009]

# Software design

`qmrpy` is organized around two layers:

1. Model layer (`qmrpy.models`): domain-specific implementations with a common
   API (`forward`, `fit`, `fit_image`).
2. I/O and CLI layer (`qmrpy.io`, `qmrpy.cli`): TIFF core I/O, optional
   NIfTI/DICOM/qMRI-BIDS helpers, NIfTI result-map export, and thin
   command-line wrappers around the public Python API.
3. Validation layer (`scripts/summarize_parity.py` +
   `configs/exp/validation_core.toml`): reproducible case definitions,
   thresholds, and machine-readable summaries.

The validation script supports suite selection (`core`, `decaes`, `qmrlab`,
`all`) and output format selection (`csv`, `markdown`, `json`). The
external-dependency-free core suite validates T1, T1rho, MT, MRF, T2/T2*,
B0/B1, QSM, and simulation behavior without requiring qMRLab/Octave.

A standard run is:

```bash
uv run --locked -- python scripts/summarize_parity.py \
  --suite core \
  --formats csv,markdown,json \
  --config configs/exp/validation_core.toml \
  --out-dir output/reports/parity_summary
```

# Research impact statement

The main impact of `qmrpy` is enabling reproducible cross-domain verification of
qMRI implementations within a single Python environment. Using the default core
validation configuration, all 21 validation cases passed with deterministic
outputs across nine domains (B0, B1, MRF, MT, QSM, Simulation, T1, T2, T2*).

Primary metrics from `core_validation.csv` are:

| Domain | Model | Primary metric | Value | Threshold |
|---|---|---|---:|---:|
| B0 | `b0_dual_echo` | `b0_mae_hz` | 1.594 | 2.0 |
| B0 | `b0_multi_echo` | `b0_mae_hz` | 0.1707 | 1.5 |
| B1 | `b1_bloch_siegert` | `b1_mae_abs` | 0.0102 | 0.06 |
| B1 | `b1_dam` | `b1_mae_abs` | 0.00595 | 0.08 |
| MRF | `mrf_dictionary` | `t1_error_rate` | 0.0 | 0.0 |
| MT | `mtr` | `mtr_mae_abs` | 0.00208 | 0.02 |
| MT | `mtsat` | `mtsat_mae_abs` | 0.00440 | 0.02 |
| QSM | `qsm` | `chi_l2_repro_rmse` | 0.0 | 1e-12 |
| Simulation | `simulation` | `t2_rel_mae` | 0.00701 | 0.10 |
| T1 | `despot1_hifi` | `t1_rel_mae` | 0.00555 | 0.10 |
| T1 | `inversion_recovery` | `t1_rel_mae` | 0.0604 | 0.08 |
| T1 | `mp2rage` | `t1_rel_mae` | 5.87e-12 | 0.12 |
| T1 | `t1rho` | `t1rho_rel_mae` | 0.00350 | 0.08 |
| T1 | `vfa_t1` | `t1_rel_mae` | 0.0296 | 0.08 |
| T2 | `emc_t2` | `t2_rel_mae` | 0.0124 | 0.12 |
| T2 | `epg_t2` | `t2_rel_mae` | 0.00320 | 0.10 |
| T2 | `mono_t2` | `t2_rel_mae` | 0.00782 | 0.06 |
| T2 | `mwf` | `mwf_mae_abs` | 0.0101 | 0.06 |
| T2 | `t2_water_fat` | `fat_fraction_mae` | 5.76e-16 | 1e-12 |
| T2* | `r2star_complex` | `t2star_rel_mae` | 0.00455 | 0.10 |
| T2* | `r2star_mono` | `t2star_rel_mae` | 0.00536 | 0.08 |

These artifacts support regression monitoring and method comparison workflows,
while preserving interoperability with established qMRI conventions.

# AI usage disclosure

A generative AI assistant was used to draft portions of this manuscript and to outline documentation
changes. All content was reviewed, validated, and edited by the author.

# Acknowledgements

We thank the qMRLab and DECAES communities for open-source reference implementations that enabled this
reimplementation effort.

# References
