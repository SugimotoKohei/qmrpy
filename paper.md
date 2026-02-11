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
quantitative MRI (qMRI) modeling, fitting, and simulation. The package provides
T1/T2/B1/QSM/noise/simulation components under a unified API and pairs those
implementations with reproducible validation artifacts generated from fixed
configurations and deterministic seeds.

Rather than framing contribution primarily as "Python reimplementation", `qmrpy`
focuses on making qMRI model behavior auditable and comparable across domains in
one environment. This includes a common model interface (`forward`, `fit`,
`fit_image`), consistent output conventions, and a validation suite that
produces machine-readable pass/fail summaries.

# Statement of need

qMRI pipelines are often implemented in ecosystem-specific tools (e.g., MATLAB,
Julia), and many cross-method workflows in Python rely on ad hoc wrappers or
single-model scripts. A key gap is not only language availability but
**verification consistency**: users need a practical way to evaluate whether
multiple model families remain numerically stable under one reproducible setup.

`qmrpy` addresses this gap by providing:

- Cross-domain qMRI models in a single Python package
- A shared API contract for voxel-wise and image-wise fitting
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
- Mono-exponential and EPG-corrected T2 fitting
- Multi-component T2 / myelin water fraction analysis [@MacKay1994]
- MPPCA denoising [@Veraart2016]
- Split-Bregman-based QSM reconstruction [@Goldstein2009]

# Software design

`qmrpy` is organized around two layers:

1. Model layer (`qmrpy.models`): domain-specific implementations with a common
   API (`forward`, `fit`, `fit_image`).
2. Validation layer (`scripts/summarize_parity.py` +
   `configs/exp/validation_core.toml`): reproducible case definitions,
   thresholds, and machine-readable summaries.

The validation script supports suite selection (`core`, `decaes`, `qmrlab`,
`all`) and output format selection (`csv`, `markdown`, `json`). The
external-dependency-free core suite validates T1/T2/B1/QSM/simulation behavior
without requiring qMRLab/Octave.

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
validation configuration, all eight validation cases passed with deterministic
outputs across five domains (T1, T2, B1, QSM, Simulation).

Primary metrics from `core_validation.csv` are:

| Domain | Model | Primary metric | Value | Threshold |
|---|---|---|---:|---:|
| T1 | `vfa_t1` | `t1_rel_mae` | 0.0232 | 0.08 |
| T1 | `inversion_recovery` | `t1_rel_mae` | 0.0642 | 0.08 |
| T2 | `mono_t2` | `t2_rel_mae` | 0.00782 | 0.06 |
| T2 | `epg_t2` | `t2_rel_mae` | 0.00555 | 0.10 |
| T2 | `mwf` | `mwf_mae_abs` | 0.0129 | 0.06 |
| B1 | `b1_dam` | `b1_mae_abs` | 0.00681 | 0.08 |
| QSM | `qsm` | `chi_l2_repro_rmse` | 0.0 | 1e-12 |
| Simulation | `simulation` | `t2_rel_mae` | 0.0152 | 0.10 |

These artifacts support regression monitoring and method comparison workflows,
while preserving interoperability with established qMRI conventions.

# AI usage disclosure

A generative AI assistant was used to draft portions of this manuscript and to outline documentation
changes. All content was reviewed, validated, and edited by the author.

# Acknowledgements

We thank the qMRLab and DECAES communities for open-source reference implementations that enabled this
reimplementation effort.

# References
