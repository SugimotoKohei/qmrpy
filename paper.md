---
title: "qmrpy: Python reimplementation of quantitative MRI models with parity checks"
tags:
  - MRI
  - quantitative MRI
  - relaxometry
  - Python
authors:
  - name: Kohei Sugimoto
    affiliation: 1
    orcid: 0000-0003-2702-5235
affiliations:
  - name: Independent researcher
    index: 1
date: 2026-01-15
bibliography: paper.bib
---

# Summary

qmrpy is a Python toolkit for quantitative MRI (qMRI) modeling, fitting, and simulation. It reimplements
core models and workflows originating from qMRLab and DECAES in a unified, Python-native API, while keeping
outputs compatible with established conventions. The library provides both object-oriented models and
functional entry points to support single-voxel fitting, image/volume fitting, and protocol-driven
simulation workflows.

A citable software archive will be provided via Zenodo before submission; the reference entry
`@qmrpy_zenodo` should be updated with the final DOI.

# Statement of need

Many qMRI pipelines and reference implementations are distributed in MATLAB or Julia, which can hinder
integration with Python-based analysis stacks and reproducible workflows. qmrpy fills this gap by offering
Python implementations of widely used qMRI models, with consistent parameter naming, structured return
keys, and testable interfaces. This enables researchers to embed qMRI modeling directly into Python data
pipelines and to compare methods under a single, reproducible environment.

# State of the field

qMRLab provides a comprehensive MATLAB framework for qMRI model fitting and simulation, and DECAES offers
a Julia implementation for multi-component T2 mapping and regularization strategies. qmrpy focuses on
bringing these capabilities to the Python ecosystem, where a unified, tested, and reproducible
implementation remains limited. The package bridges these ecosystems while preserving the behavior and
output conventions expected by users of the upstream tools [@Karakuzu2020; @Doucette2020]. It includes
DESPOT1-style variable flip angle T1 mapping [@Deoni2005], inversion recovery fitting [@Barral2010],
multi-component T2 analysis for myelin water imaging [@MacKay1994], MPPCA denoising [@Veraart2016], and
split-Bregman regularization for inverse problems [@Goldstein2009].

# Software design

qmrpy exposes model classes with a common interface: `forward` for simulation and `fit`/`fit_image` for
parameter estimation. A functional API mirrors these model interfaces for lightweight use. The codebase
separates model logic, simulation utilities, and high-level pipelines, allowing both direct model usage
and composable workflows. Numerical parity is checked against reference data where available, and a
comprehensive test suite covers model outputs, shape consistency, and error handling.

# Research impact statement

qmrpy reduces the barrier to reuse and comparison of established qMRI models in Python-based workflows.
Parity checks against DECAES reference data show small absolute discrepancies across multiple
regularization strategies, indicating faithful reproduction of upstream behavior. The project also
provides broad test coverage across T1/T2/B1/QSM/noise/simulation components, helping maintain numerical
consistency over time and enabling reproducible method comparisons within a single ecosystem.

# AI usage disclosure

A generative AI assistant was used to draft portions of this manuscript and to outline documentation
changes. All content was reviewed, validated, and edited by the author.

# Acknowledgements

We thank the qMRLab and DECAES communities for open-source reference implementations that enabled this
reimplementation effort.

# References
