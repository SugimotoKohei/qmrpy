# Installation

## Requirements

- Python 3.11+
- NumPy, SciPy

## Install from PyPI

```bash
pip install qmrpy
```

## Install from Source

```bash
git clone https://github.com/SugimotoKohei/qmrpy.git
cd qmrpy
pip install -e .
```

## Using uv (recommended for development)

```bash
git clone https://github.com/SugimotoKohei/qmrpy.git
cd qmrpy
uv sync
```

## Optional Dependencies

For TIFF I/O:
```bash
pip install pillow
```

For parallel processing (included by default):
```bash
pip install joblib tqdm
```
