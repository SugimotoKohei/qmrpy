from __future__ import annotations

from typing import Any


def add_gaussian_noise(signal: Any, *, sigma: float, rng: Any) -> Any:
    """Add i.i.d. Gaussian noise to a real-valued signal.

    Parameters
    ----------
    signal:
        Array-like.
    sigma:
        Standard deviation of the additive noise.
    rng:
        NumPy Generator-compatible object with `.normal`.
    """
    import numpy as np

    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    x = np.asarray(signal, dtype=np.float64)
    if sigma == 0:
        return x
    return x + rng.normal(loc=0.0, scale=float(sigma), size=x.shape)


def add_rician_noise(signal: Any, *, sigma: float, rng: Any) -> Any:
    """Add Rician noise (magnitude of complex Gaussian) to a real-valued signal.

    This matches the common qMRI magnitude noise model:
        y = sqrt( (s + n1)^2 + n2^2 ),  n1,n2 ~ N(0, sigma)
    """
    import numpy as np

    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    s = np.asarray(signal, dtype=np.float64)
    if sigma == 0:
        return s
    n1 = rng.normal(loc=0.0, scale=float(sigma), size=s.shape)
    n2 = rng.normal(loc=0.0, scale=float(sigma), size=s.shape)
    return np.sqrt((s + n1) ** 2 + (n2**2))

