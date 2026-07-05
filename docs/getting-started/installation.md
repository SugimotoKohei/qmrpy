# Installation

## Requirements

- Python 3.11
- NumPy, SciPy

## Install from PyPI

```bash
pip install qmrpy
```

## Install from Source

```bash
git clone https://github.com/SugimotoKohei/qmrpy.git
cd qmrpy
uv sync --locked --extra dev --extra io --group docs
```

## Using uv (recommended for development)

```bash
git clone https://github.com/SugimotoKohei/qmrpy.git
cd qmrpy
uv sync --locked --extra dev --extra io --group docs
```

## Optional Dependencies

TIFF I/O is available in the core package through Pillow.

For NIfTI, DICOM, and minimal qMRI-BIDS helpers:

```bash
pip install "qmrpy[io]"
```

For local development quality gates:

```bash
uv run --locked pre-commit run --all-files
uv run --locked mypy src/qmrpy
uv run --locked -m pytest
uv run --locked mkdocs build
```
