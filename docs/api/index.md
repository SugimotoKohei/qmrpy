# API Reference

This section provides detailed API documentation for all qmrpy modules.

## Models

| Module | Description |
|--------|-------------|
| [T1 Models](t1.md) | VFA T1, Inversion Recovery, DESPOT1-HIFI, T1MP2RAGE |
| [T2 Models](t2.md) | T2Mono, T2EPG, EMC, MWF, DECAES |
| [T2* Models](t2star.md) | R2Star mono/complex, ESTATICS |
| [B0 Models](b0.md) | Dual-echo, Multi-echo phase regression |
| [B1 Models](b1.md) | DAM, AFI, Bloch-Siegert |

## Utilities

| Module | Description |
|--------|-------------|
| [Functional API](functional.md) | Convenience functions |
| [I/O Utilities](io.md) | TIFF save/load |
| [Simulation](simulation.md) | Phantoms, Bloch simulation |

## Common Interface

All fitting models follow a consistent interface:

### Model Creation

```python
model = ModelClass(acquisition_parameters...)
```

### Single Voxel Fit

```python
result = model.fit(signal, **options)
# Returns: dict with fitted parameters
```

### Image Fit

```python
maps = model.fit_image(
    signal,           # ND array, last dim = time/echoes
    mask=None,        # None, "otsu", or boolean array
    n_jobs=1,         # Parallel jobs (-1 = all CPUs)
    verbose=False,    # Show progress bar
    **options
)
# Returns: dict of parameter maps
```

### Forward Model

```python
signal = model.forward(parameters...)
# Returns: simulated signal array
```
