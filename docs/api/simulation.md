# Simulation

qmrpy provides tools for simulating MRI signals and phantoms.

## Overview

The simulation module (`qmrpy.sim`) provides:

- **Phantoms**: 2D/3D/4D numerical phantoms with tissue parameters
- **Bloch simulation**: Time-domain signal simulation
- **EPG**: Extended Phase Graph simulation
- **Noise**: Gaussian and Rician noise models
- **CRLB**: Cramér-Rao lower bound analysis

## Quick Examples

### Generate Phantom

```python
from qmrpy.sim import generate_4d_phantom

# Create 4D phantom with T2 decay
phantom, params = generate_4d_phantom(
    size=(64, 64, 10),
    n_echoes=32,
    te_ms=10.0,
    snr=50.0,
)
print(phantom.shape)  # (64, 64, 10, 32)
```

### Simulate Single Voxel

```python
from qmrpy.sim import simulate_single_voxel

signal = simulate_single_voxel(
    t1_ms=1000.0,
    t2_ms=50.0,
    te_ms=[10, 20, 30, 40, 50],
    model="mono_t2",
)
```

### Add Noise

```python
from qmrpy.sim import add_rician_noise

noisy_signal = add_rician_noise(signal, snr=30.0)
```

## Available Functions

For detailed API documentation, see the source code in `qmrpy.sim`.

### Phantoms

- `generate_4d_phantom()`: Generate 4D phantom with decay
- `generate_shepp_logan()`: Shepp-Logan phantom

### Simulation

- `simulate_single_voxel()`: Simulate signal for one voxel
- `simulate_bloch()`: Bloch equation simulation
- `epg_decay()`: EPG simulation

### Noise

- `add_gaussian_noise()`: Add Gaussian noise
- `add_rician_noise()`: Add Rician noise

### Analysis

- `sim_crlb`: Cramér-Rao lower bound calculator
- `sim_fisher_matrix`: Fisher information matrix
