---
title: 'qmrpy: A Python package for quantitative MRI modeling'
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

`qmrpy` is a Python package for quantitative MRI (qMRI) modeling, fitting, and
simulation. It reimplements core models and workflows originating from qMRLab
[@Karakuzu2020] and DECAES [@Doucette2020] in a unified, Python-native API,
while maintaining outputs compatible with established conventions. The library
provides both object-oriented model classes and functional entry points to
support single-voxel fitting, image/volume fitting, and protocol-driven
simulation workflows.

Quantitative MRI enables the measurement of tissue properties such as T1 and T2
relaxation times, which provide valuable biomarkers for clinical diagnosis and
neuroscience research. `qmrpy` brings these capabilities to the Python
ecosystem, enabling seamless integration with modern data science workflows.

# Statement of need

Many qMRI pipelines and reference implementations are distributed in MATLAB or
Julia, which can hinder integration with Python-based analysis stacks and
reproducible workflows. `qmrpy` fills this gap by offering Python
implementations of widely used qMRI models, with consistent parameter naming,
structured return keys, and testable interfaces. This enables researchers to
embed qMRI modeling directly into Python data pipelines and to compare methods
under a single, reproducible environment.

The package targets MRI researchers, neuroscientists, and medical physicists
who need accessible tools for relaxometry analysis. It has been designed to
lower the barrier for students learning qMRI techniques while providing
production-ready implementations for experienced users.

# State of the field

Several tools exist for quantitative MRI analysis. qMRLab [@Karakuzu2020]
provides a comprehensive MATLAB framework for qMRI model fitting and simulation.
DECAES [@Doucette2020] offers a Julia implementation for multi-component T2
mapping with non-negative least squares and regularization strategies.

`qmrpy` was built to bring these capabilities to the Python ecosystem, where a
unified, tested, and reproducible implementation remains limited. The package
bridges these ecosystems while preserving the behavior and output conventions
expected by users of the upstream tools. Key implemented models include:

- DESPOT1-style variable flip angle T1 mapping [@Deoni2005]
- Inversion recovery T1 fitting [@Barral2010]
- Multi-component T2 analysis for myelin water imaging [@MacKay1994]
- EPG-corrected T2 fitting with B1 inhomogeneity correction
- MPPCA denoising [@Veraart2016]
- Split-Bregman regularization for QSM inverse problems [@Goldstein2009]

# Software design

`qmrpy`'s design philosophy centers on providing a consistent, user-friendly API
across all model types. The package exposes model classes with a common
interface:

- `forward()` for signal simulation given tissue parameters
- `fit()` for single-voxel parameter estimation
- `fit_image()` for volume-wise fitting with optional parallelization

A functional API (`qmrpy.functional`) mirrors these model interfaces for
lightweight, stateless use. The codebase separates model logic
(`qmrpy.models`), simulation utilities (`qmrpy.sim`), and I/O operations,
allowing both direct model usage and composable workflows.

All `fit_image()` methods support parallel execution via the `n_jobs` parameter,
Otsu-based automatic masking, and progress bars for long-running fits. Numerical
parity is verified against reference data where available, and a comprehensive
test suite covers model outputs, shape consistency, and error handling.

# Research impact statement

`qmrpy` reduces the barrier to reuse and comparison of established qMRI models
in Python-based workflows. Parity checks against DECAES reference data show
small absolute discrepancies across multiple regularization strategies,
indicating faithful reproduction of upstream behavior.

The project provides broad test coverage across T1/T2/B1/QSM/noise/simulation
components, helping maintain numerical consistency over time and enabling
reproducible method comparisons within a single ecosystem. By providing
accessible Python implementations, `qmrpy` enables researchers to rapidly
prototype and validate qMRI analysis pipelines.

# AI usage disclosure

A generative AI assistant was used to draft portions of this manuscript and to outline documentation
changes. All content was reviewed, validated, and edited by the author.

# Acknowledgements

We thank the qMRLab and DECAES communities for open-source reference implementations that enabled this
reimplementation effort.

# References
