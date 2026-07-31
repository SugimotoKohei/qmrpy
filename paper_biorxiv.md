---
title: "qmrpy: a verification-first Python reference implementation for quantitative MRI modelling and cross-domain validation"
bibliography: paper.bib
link-citations: true
geometry: margin=1in
fontsize: 11pt
linestretch: 1.15
colorlinks: true
---

**Kohei Sugimoto**^1,\*^

^1^ Independent Researcher, Japan. ORCID: 0000-0003-2702-5235

\* Correspondence: sugimotokouhei@gmail.com

**Running title:** Verification-first qMRI modelling in Python

**Keywords:** quantitative MRI; relaxometry; myelin water imaging; magnetization
transfer; reproducibility; open-source software; Python

---

# Abstract

Quantitative MRI (qMRI) converts image contrast into physical tissue parameters
such as T1, T2, T1rho, magnetization transfer indices, magnetic susceptibility
and myelin water fraction. Reference implementations of these models are
distributed across separate software ecosystems — most notably MATLAB (qMRLab)
and Julia (DECAES) — which makes it difficult to check, in one place, whether
several model families remain numerically consistent under a single reproducible
setup. We present qmrpy, an open-source Python package that implements 24 qMRI
model classes across nine domains behind a single API contract (`forward`,
`fit`, `fit_image`) and a common result schema, and that ships the verification
evidence together with the code. Verification is defined declaratively: a
configuration file fixes each case, its random seed, its primary metric and its
pass threshold, and a summary script emits machine-readable CSV, Markdown and
JSON reports. In the dependency-free core suite, all 21 cases pass across B0,
B1, MRF, MT, QSM, simulation, T1, T2 and T2\* mapping; the tightest case still
sits 20% below its threshold, and two cases recover the reference exactly.
Parameter
recovery over physiological ranges gives mean relative errors of 2.6% (T1),
0.37% (T1rho) and 0.77% (T2), and a mean absolute myelin water fraction error of
0.011. Against fixed DECAES reference vectors, T2 distributions agree to
$2.2 \times 10^{-15}$ without regularization. qmrpy makes cross-domain qMRI model
behaviour auditable, regression-tested and directly usable from Python.

# Introduction

Quantitative MRI (qMRI) replaces qualitative image intensity with estimates of
physical parameters, and those estimates are increasingly used as biophysical
markers of tissue microstructure. Longitudinal and transverse relaxation times
[@Deoni2005; @Barral2010], the rotating-frame relaxation time T1rho,
magnetization transfer indices [@Helms2008], multi-component T2 distributions
and the derived myelin water fraction [@MacKay1994; @Whittall1989], magnetic
susceptibility, and simultaneous multi-parameter estimation by MR fingerprinting
[@Ma2013] are all now routine research measurements.

The corresponding reference implementations, however, are spread across software
ecosystems. qMRLab [@Karakuzu2020] provides a broad MATLAB framework for qMRI
fitting and simulation, and DECAES [@Doucette2020] provides a fast Julia
implementation of regularized non-negative least squares for multi-component T2
analysis. Both are mature and well validated, but a Python user who wants to
combine several qMRI model families in one analysis typically ends up with ad
hoc wrappers, language bridges, or single-model scripts.

We argue that the more consequential gap is not language availability but
*verification consistency*. Individual qMRI implementations are usually
validated in isolation, with study-specific simulation settings that are
described in prose rather than encoded in a re-runnable artefact. As a result it
is hard to answer a simple operational question: under one fixed, reproducible
setup, do all of the model families in a toolbox still recover known ground
truth within stated tolerances, and would a regression be detected
automatically?

This paper describes qmrpy, a Python package designed around that question.
Its contributions are:

1. **Cross-domain coverage under one API.** 24 model classes spanning T1, T1rho,
   MT, MRF, T2, T2\*, B0, B1 and QSM share the same `forward` /
   `fit` / `fit_image` contract and the same result schema.
2. **Declarative, machine-readable verification.** Validation cases, seeds,
   metrics and pass thresholds live in a version-controlled configuration file;
   a single command regenerates CSV, Markdown and JSON evidence.
3. **Verification as a continuous-integration gate.** The core suite runs
   without MATLAB, Octave or Julia, and a failing case fails the build.
4. **Cross-implementation parity checks.** Fixed reference vectors exported from
   DECAES, and archived qMRLab comparison reports, quantify agreement with the
   established implementations.
5. **Practical integration.** Optional NIfTI, DICOM and qMRI-BIDS
   [@Gorgolewski2016; @Karakuzu2022] helpers preserve spatial metadata for
   result-map export, and a thin command-line interface wraps the public API.

# Methods

## Software architecture

qmrpy (version 2.0.0) is organized in three layers.

The **model layer** (`qmrpy.models`) contains the domain-specific
implementations. Every model class exposes the same three operations:
`forward(...)` generates a noise-free signal from model parameters,
`fit(signal, ...)` estimates parameters for a single voxel or signal vector, and
`fit_image(data, ...)` applies the same estimator over an image array with
optional automatic masking (`mask="otsu"`) and multi-core execution
(`n_jobs=-1`). Results are returned through a common schema in which parameter
estimates are accessed by name (for example `result["t2_ms"]`), while goodness
of fit and solver diagnostics are attached as `result.quality` and
`result.diagnostics`. A functional API (`qmrpy.functional`) exposes the same
estimators as plain functions for users who prefer not to instantiate model
objects.

The **I/O and interface layer** (`qmrpy.io`, `qmrpy.cli`) provides TIFF
read/write in the core installation, and — under the optional `qmrpy[io]`
extra — NIfTI loading and saving, DICOM series loading, qMRI-BIDS relaxometry
loading, and result-map export that inherits the affine and header of the source
volume. The `qmrpy` console script wraps the public Python API for inspection
(`qmrpy info`), NIfTI-in/NIfTI-out fitting (`qmrpy fit ...`) and validation
(`qmrpy validate`).

The **validation layer** (`configs/exp/validation_core.toml` and
`scripts/summarize_parity.py`) defines the verification cases and emits the
reports described below.

The package requires Python 3.11 or newer and depends on NumPy [@Harris2020] and
SciPy [@Virtanen2020] for numerics, joblib for parallel voxel-wise fitting, and
pypulseq [@Layton2017] for pulse-sequence definition. Bloch-based sequence
simulation through MRzeroCore [@Loktyushin2021] is available as an opt-in extra
(`qmrpy[mrzero]`) rather than a default dependency, because MRzeroCore is
licensed under AGPL-3.0 and installing it places the combined environment under
AGPL-3.0 terms. qmrpy itself is distributed under the MIT licence. Modules that
are Python translations of qMRLab or DECAES.jl routines carry an attribution
header, and the correspondence between qmrpy modules and their upstream sources
is recorded in `THIRD_PARTY_NOTICES.md`.

## Implemented models

Table 1 lists the implemented model classes grouped by qMRI domain. Multi-echo
spin-echo models that need stimulated-echo correction are built on an extended
phase graph (EPG) engine [@Hennig1988], which is shared by the EPG-corrected T2
fit, the EMC-style Bloch-simulation reconstruction [@BenEliezer2015], the
stimulated-echo-corrected multi-component analysis [@Prasloski2012] and the
MR fingerprinting dictionary generator.

Table: Model classes implemented in qmrpy 2.0.0.

| Domain | Classes | Method summary |
|---|---|---|
| T1 | `T1VFA`, `T1InversionRecovery`, `T1DESPOT1HIFI`, `T1MP2RAGE` | DESPOT1-style variable flip angle [@Deoni2005], inversion recovery [@Barral2010], joint T1/B1 estimation, MP2RAGE lookup [@Marques2010] |
| T1rho | `T1Rho` | Mono-exponential spin-lock decay |
| MT | `MTR`, `MTsat` | Magnetization transfer ratio; T1- and B1-corrected saturation [@Helms2008] |
| MRF | `MRFDictionary` | Spoiled-FISP-style dictionary generation and normalized inner-product matching [@Ma2013] |
| T2 | `T2Mono`, `T2EPG`, `T2EMC`, `T2MultiComponent`, `T2WaterFat`, `T2DECAESMap`, `T2DECAESPart` | Mono-exponential fit; EPG-corrected fit [@Hennig1988]; Bloch-simulation reconstruction [@BenEliezer2015]; regularized NNLS multi-component analysis [@Whittall1989; @Prasloski2012]; two-pool water/fat separation; DECAES-compatible T2 distribution and partitioning [@Doucette2020] |
| T2\* | `T2StarMonoR2`, `T2StarComplexR2`, `T2StarESTATICS` | Magnitude and complex-signal R2\* estimation; ESTATICS-style joint fit |
| B0 | `B0DualEcho`, `B0MultiEcho` | Phase-difference and linear multi-echo phase field mapping |
| B1 | `B1DAM`, `B1AFI`, `B1BlochSiegert` | Double-angle method; actual flip-angle imaging [@Yarnykh2007]; Bloch-Siegert shift [@Sacolick2010] |
| QSM | `QSMSplitBregman` | Split-Bregman L1-regularized dipole inversion [@Goldstein2009] |

Signal simulation utilities (`qmrpy.sim`) provide Gaussian and Rician noise
models, digital phantoms, Pulseq sequence templates [@Layton2017] and an MRzero
interface [@Loktyushin2021] for Bloch-based sequence simulation.

## Verification framework

Verification in qmrpy is declarative. The file
`configs/exp/validation_core.toml` defines, for each case: a global base seed,
the number of synthetic samples, the acquisition settings (echo times, flip
angles, spin-lock times, repetition time, and so on), the ground-truth parameter
range, the noise model and noise level, and one or more acceptance thresholds.
Per-case seeds are derived deterministically from the base seed, so the entire
suite is reproducible bit-for-bit.

`scripts/summarize_parity.py` executes the cases, computes the metrics and
writes the evidence:

```bash
uv run --locked -- python scripts/summarize_parity.py \
  --suite core \
  --formats csv,markdown,json \
  --config configs/exp/validation_core.toml \
  --out-dir output/reports/parity_summary
```

Each case yields one row in `core_validation.csv` (domain, model, case, seed,
sample count, primary metric, value, threshold, unit, pass flag) and one row per
metric in `core_validation_metrics.csv`. A case passes only when *every* metric
in that case satisfies `value <= threshold`; a failing case makes the script
exit with a non-zero status, so the suite acts as a gate rather than as a report.

Three suites are available. The **core** suite is free of external dependencies
and is the one used for the claims in this paper. The **decaes** suite compares
qmrpy against fixed reference vectors exported from DECAES [@Doucette2020] and
stored in the repository, so it is reproducible without a Julia installation.
The **qmrlab** suite aggregates previously generated qMRLab comparison reports;
regenerating those reports requires a local qMRLab/Octave installation.

## Reproducibility and quality control

The Python environment is pinned with uv and a committed lockfile, so
`uv sync --locked` reproduces the exact dependency set. Continuous integration
runs on Ubuntu and macOS and enforces, in order: lockfile consistency
(`uv lock --check`), formatting and linting (ruff), static typing (mypy), the
142-test automated test suite with coverage reporting (pytest), the
documentation build (mkdocs) and the core validation suite. Because the
validation suite is part of the same gate, a numerical regression in any model
family blocks the build in the same way a failing unit test does.

# Results

## Cross-domain validation

All 21 cases of the core validation suite pass, covering nine domains (B0, B1,
MRF, MT, QSM, simulation, T1, T2 and T2\*). Table 2 lists the primary metric,
its value and its threshold for every case.

Figure 1 shows the same result as a margin: each case's primary metric divided
by its pass threshold, on a logarithmic axis. The largest margin ratio is 0.80
(dual-echo B0 mapping, 1.59 Hz against a 2.0 Hz threshold) and the smallest
non-zero ratio is $4.9 \times 10^{-11}$ (MP2RAGE); the MR fingerprinting and QSM
reproducibility cases recover the reference exactly. In other words, no case
passes marginally by accident, and cases whose expected behaviour is exact
recovery are verified as exact rather than as merely small.

![Normalized validation margin for all 21 core validation cases. Each point is the primary metric of one case divided by its pass threshold, on a logarithmic axis; the dashed line marks the threshold, so every point to its left is a pass. Circles denote cases that recover the reference exactly (ratio 0, drawn at the axis floor and annotated); diamonds denote cases with a finite margin. Colour encodes the qMRI domain.](output/paper_figures/fig1_validation_margin.png)

## Parameter recovery over physiological ranges

To show behaviour across a parameter range rather than at a single operating
point, four representative estimators were evaluated on synthetic signals
generated with the acquisition and noise settings of the corresponding
validation cases, sweeping the ground truth over a physiologically relevant
interval (Figure 2). Mean relative errors were 2.6% for variable-flip-angle T1
over 500–2000 ms, 0.37% for spin-lock T1rho over 30–120 ms and 0.77% for
mono-exponential T2 over 30–150 ms. Myelin water fraction was recovered with a
mean absolute error of 0.011 over the range 0.08–0.25, i.e. about one
percentage point of myelin water fraction, consistent with the 0.010 reported by
the corresponding validation case.

![Parameter recovery for four representative models. Points are qmrpy estimates against ground truth; the dashed line is identity. Signals were generated with the acquisition and Gaussian-noise settings of the corresponding core validation cases.](output/paper_figures/fig2_parameter_recovery.png)

## Image-level fitting

`fit_image` was applied to a synthetic $64 \times 64$ phantom containing four
disks with T2 values of 30, 60, 90 and 130 ms, sampled at eight echo times
between 10 and 120 ms with additive Gaussian noise ($\sigma$ = 15 on a signal
amplitude of 1000–1300). The recovered map (Figure 3) has a mean absolute error
of 1.64 ms, corresponding to a mean relative error of 2.2%, and preserves the
disk boundaries without spatial regularization.

![Image-level mono-exponential T2 mapping on a synthetic phantom. Left: ground-truth T2. Right: voxel-wise qmrpy estimate obtained with `fit_image`. Both panels share the colour scale.](output/paper_figures/fig3_t2_phantom_map.png)

## Parity with established implementations

Against the fixed DECAES reference vectors stored in the repository (16 echo
times, 30 T2 basis values), the recovered T2 distribution agreed to a maximum
absolute difference of $2.2 \times 10^{-15}$ with a fixed 180° flip angle and no
regularization. With flip-angle optimization enabled, the maximum absolute
distribution difference was $1.1 \times 10^{-6}$ for generalized cross-validation
and below $5 \times 10^{-12}$ for the L-curve, chi-squared and discrepancy-principle
criteria, with flip-angle differences of $2.0 \times 10^{-10}$ degrees in all four
cases. These residuals are consistent with accumulated floating-point
differences between the two implementations rather than with a difference in the
estimator itself.

For multi-component analysis, 135 archived qMRLab comparison reports were
aggregated. In the 87 reports generated with a simulated noise level of
$\sigma = 0.001$ or $\sigma = 0.002$, the myelin water fraction differed from
qMRLab by at most 0.71 percentage points, and the short- and long-T2 component
times by at most 0.61 ms and 0.40 ms respectively. The remaining 48 reports — 46
generated without added noise, one with $\sigma = 2$, and one with no recorded
noise setting — differed by up to 10.5 percentage points, indicating that the
two implementations diverge mainly in the noiseless limit, where the
regularization search is least constrained. This qMRLab comparison depends on
locally generated reports and therefore is reported as supporting evidence
rather than as part of the dependency-free core suite.

| Domain | Model | Primary metric | Value | Threshold | Unit |
|:-----------|:---------------------|:--------------------|-------------:|-----------:|:-----------|
| B0 | b0_dual_echo | b0_mae_hz | 1.594 | 2 | Hz |
| B0 | b0_multi_echo | b0_mae_hz | 0.1707 | 1.5 | Hz |
| B1 | b1_bloch_siegert | b1_mae_abs | 0.01025 | 0.06 | ratio |
| B1 | b1_dam | b1_mae_abs | 0.005951 | 0.08 | ratio |
| MRF | mrf_dictionary | t1_error_rate | 0 | 0 | error rate |
| MT | mtr | mtr_mae_abs | 0.002085 | 0.02 | ratio |
| MT | mtsat | mtsat_mae_abs | 0.004401 | 0.02 | ratio |
| QSM | qsm | chi_l2_repro_rmse | 0 | 1e-12 | a.u. |
| Simulation | simulation | t2_rel_mae | 0.007012 | 0.1 | ratio |
| T1 | despot1_hifi | t1_rel_mae | 0.005551 | 0.1 | ratio |
| T1 | inversion_recovery | t1_rel_mae | 0.06042 | 0.08 | ratio |
| T1 | mp2rage | t1_rel_mae | 5.867e-12 | 0.12 | ratio |
| T1 | t1rho | t1rho_rel_mae | 0.003499 | 0.08 | ratio |
| T1 | vfa_t1 | t1_rel_mae | 0.0296 | 0.08 | ratio |
| T2 | emc_t2 | t2_rel_mae | 0.01238 | 0.12 | ratio |
| T2 | epg_t2 | t2_rel_mae | 0.003199 | 0.1 | ratio |
| T2 | mono_t2 | t2_rel_mae | 0.00782 | 0.06 | ratio |
| T2 | mwf | mwf_mae_abs | 0.01007 | 0.06 | fraction |
| T2 | t2_water_fat | fat_fraction_mae | 5.759e-16 | 1e-12 | fraction |
| T2\* | r2star_complex | t2star_rel_mae | 0.004553 | 0.1 | ratio |
| T2\* | r2star_mono | t2star_rel_mae | 0.005362 | 0.08 | ratio |

Table: Core validation results for qmrpy 2.0.0. All 21 cases pass.

# Discussion

qmrpy packages a set of qMRI estimators together with the evidence that they
behave as intended, in a form that a reader can re-execute. The design choice
that makes this practical is the separation between *what is verified* (a
version-controlled configuration of cases, seeds, metrics and thresholds) and
*how it is verified* (a single script that emits machine-readable evidence). The
resulting artefacts serve three purposes at once: they document expected
accuracy, they act as regression detectors in continuous integration, and they
give a downstream user a concrete tolerance to reason about.

Relative to existing tools, qmrpy is complementary rather than competitive.
qMRLab [@Karakuzu2020] covers a broader set of models and includes a graphical
interface and extensive protocol tooling; DECAES [@Doucette2020] is
substantially faster for large-scale multi-component T2 analysis. qmrpy's
distinguishing features are that its models live natively in the Python
scientific stack, that they share one API contract and one result schema across
nine domains, and that the cross-domain verification evidence is generated by
the repository itself rather than described in prose. The agreement with DECAES
reference vectors at the level of floating-point noise, and with qMRLab within
0.71 percentage points of myelin water fraction whenever noise is present,
supports the claim that this consistency is not obtained at the cost of
correctness.

Several limitations should be stated plainly. First, the entire core validation
suite uses synthetic ground truth; it establishes numerical self-consistency and
regression control, not in vivo accuracy, and it does not substitute for
phantom or scanner validation. Second, some implementations are deliberately
minimal: the water/fat separation uses a two-pool T2 grid rather than a full
multi-peak spectral fat model, the MR fingerprinting module uses a
spoiled-FISP-style approximation, and the QSM module implements a single
split-Bregman dipole inversion rather than a family of inversion algorithms.
Third, the qMRLab parity evidence depends on locally generated reports and is
therefore weaker, in reproducibility terms, than the DECAES parity and the core
suite. Fourth, the thresholds in the validation configuration are practical
acceptance limits chosen to catch regressions without being brittle under
controlled synthetic noise; they are not derived from a formal error model.
Fifth, real-data I/O has been exercised on synthetic NIfTI, DICOM and BIDS
structures rather than on a multi-vendor clinical corpus.

Future work follows directly from these limitations: evaluation on physical
relaxometry phantoms and in vivo data, extension of the parity suites to
additional qMRLab models with an automated Octave-based regeneration path, and a
broader set of dipole-inversion and fat-model implementations. Because the
verification layer is declarative, each of these extensions adds cases to a
configuration file rather than a new bespoke validation script.

# Conclusion

qmrpy provides 24 quantitative MRI model classes across nine domains behind a
single Python API, together with a declarative, machine-readable and
continuously enforced verification suite in which all 21 core cases pass. It
gives MRI researchers, neuroscientists and medical physicists a way to run
cross-domain qMRI analysis inside the Python ecosystem while keeping the
numerical behaviour of each model auditable and regression-tested.

# Data and code availability

qmrpy is open-source under the MIT licence. Source code, validation
configurations, figure-generation scripts and documentation are available at
<https://github.com/SugimotoKohei/qmrpy>, and releases are distributed on PyPI
(<https://pypi.org/project/qmrpy/>). All results reported here were produced
with version 2.0.0. Third-party attributions and the licences of all optional
dependencies are listed in `THIRD_PARTY_NOTICES.md` in the repository. No human or animal data were used: every result in this
paper is generated from synthetic signals and digital phantoms by the scripts
`scripts/summarize_parity.py` and `scripts/generate_paper_figures.py` included
in the repository.

# Declarations

**Competing interests.** The author declares no competing interests.

**Funding.** This work received no specific grant from any funding agency.

**Ethics approval and consent.** Not applicable. No human participants, human
data or animal subjects were involved; all data are synthetic.

**Author contributions.** K.S. designed and implemented the software, designed
and executed the validation experiments, and wrote the manuscript.

**Generative AI disclosure.** A generative AI assistant was used to draft
portions of this manuscript and to outline documentation. All text, code,
numerical results and references were reviewed, verified and edited by the
author, who takes full responsibility for the content.

# References
