# API Reference

This section provides detailed API documentation for all qmrpy modules.

## Models

| Module | Description |
|--------|-------------|
| [T1 Models](t1.md) | VFA T1, Inversion Recovery |
| [T2 Models](t2.md) | MonoT2, EpgT2, MWF, DECAES |
| [B1 Models](b1.md) | DAM, AFI |

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
