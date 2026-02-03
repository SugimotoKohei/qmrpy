# EPG Simulation

The `qmrpy.epg` module provides Extended Phase Graph (EPG) simulations for
MRI pulse sequences.

## Overview

EPG is a formalism for computing MRI signal evolution through RF pulses,
relaxation, and gradient events. It tracks magnetization in terms of
phase states (F+, F-, Z).

## Submodules

| Module | Description |
|--------|-------------|
| `epg_se` | Spin Echo sequences (CPMG, TSE/FSE) |
| `epg_gre` | Gradient Echo sequences (FLASH, bSSFP) |
| `core` | Core EPG engine (advanced users) |

## Spin Echo Sequences

### CPMG

```python
from qmrpy.epg import epg_se

# Standard CPMG with 32 echoes
signal = epg_se.cpmg(
    t2_ms=80,
    t1_ms=1000,
    te_ms=10,
    n_echoes=32,
    b1=1.0,  # B1 scaling factor
)
```

### TSE (Variable Flip Angle)

```python
# Turbo Spin Echo with variable refocusing angles
angles = [180, 160, 140, 120, 100, 80, 60, 40]
signal = epg_se.tse(
    t2_ms=80,
    t1_ms=1000,
    te_ms=10,
    etl=8,
    refocus_angles_deg=angles,
)
```

## Gradient Echo Sequences

### FLASH (Spoiled Gradient Echo)

```python
from qmrpy.epg import epg_gre

# FLASH signal approaching steady state
signal = epg_gre.flash(
    t1_ms=1000,
    tr_ms=10,
    fa_deg=15,
    n_pulses=100,
)

# Analytical steady-state (Ernst equation)
ss = epg_gre.flash_steady_state(t1_ms=1000, tr_ms=10, fa_deg=15)

# Optimal flip angle for maximum signal
ernst = epg_gre.ernst_angle(t1_ms=1000, tr_ms=10)
```

### bSSFP (Balanced SSFP)

```python
# bSSFP with off-resonance
signal = epg_gre.bssfp(
    t1_ms=1000,
    t2_ms=80,
    tr_ms=5,
    fa_deg=45,
    n_pulses=100,
    off_resonance_hz=0.0,
)
```

### SSFP-FID and SSFP-Echo

```python
# SSFP-FID (signal after RF pulse)
signal_fid = epg_gre.ssfp_fid(
    t1_ms=1000, t2_ms=80, tr_ms=10, fa_deg=30, n_pulses=50
)

# SSFP-Echo (signal before RF pulse)
signal_echo = epg_gre.ssfp_echo(
    t1_ms=1000, t2_ms=80, tr_ms=10, fa_deg=30, n_pulses=50
)
```

## B1 Inhomogeneity

All functions support a `b1` parameter for simulating B1 field inhomogeneity:

```python
# B1 = 0.8 means flip angles are 80% of nominal
signal_b1_low = epg_se.cpmg(t2_ms=80, t1_ms=1000, te_ms=10, n_echoes=32, b1=0.8)
```

## API Reference

::: qmrpy.epg.epg_se
    options:
      show_root_heading: true
      members:
        - cpmg
        - mese
        - tse
        - decay_curve

::: qmrpy.epg.epg_gre
    options:
      show_root_heading: true
      members:
        - flash
        - flash_steady_state
        - ernst_angle
        - bssfp
        - bssfp_steady_state
        - ssfp_fid
        - ssfp_echo

::: qmrpy.epg.core.EPGSimulator
    options:
      show_root_heading: true
