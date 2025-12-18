from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
else:
    ArrayLike = Any  # type: ignore[misc,assignment]


def _as_1d_float_array(values: ArrayLike, *, name: str):
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={array.shape}")
    return array


@dataclass(frozen=True, slots=True)
class VfaT1:
    """Variable flip angle T1 model on SPGR (qMRLab: vfa_t1).

    Signal model (SPGR steady-state):
        S = M0 * sin(a) * (1 - E) / (1 - E * cos(a))
        E = exp(-TR / T1)

    Units (aligned to qMRLab protocol):
        - flip_angle_deg: degrees
        - tr_s: seconds
        - t1_s: seconds
    """

    flip_angle_deg: Any
    tr_s: float
    b1: float = 1.0

    def __post_init__(self) -> None:
        import numpy as np

        fa = _as_1d_float_array(self.flip_angle_deg, name="flip_angle_deg")
        if np.any(fa <= 0):
            raise ValueError("flip_angle_deg must be > 0")
        if self.tr_s <= 0:
            raise ValueError("tr_s must be > 0")
        if self.b1 <= 0:
            raise ValueError("b1 must be > 0")
        object.__setattr__(self, "flip_angle_deg", fa)

    def forward(self, *, m0: float, t1_s: float) -> Any:
        import numpy as np

        if t1_s <= 0:
            raise ValueError("t1_s must be > 0")
        alpha = np.deg2rad(self.flip_angle_deg) * float(self.b1)
        e = np.exp(-float(self.tr_s) / float(t1_s))
        return m0 * np.sin(alpha) * (1.0 - e) / (1.0 - e * np.cos(alpha))

    def fit_linear(self, signal: ArrayLike) -> dict[str, float]:
        """Fit by linearized SPGR relation (matches qMRLab approach).

        Linearization:
            y = S / sin(a)
            x = S / tan(a)
            y = intercept + slope * x
            slope = E, intercept = M0 * (1 - E)
        """
        import numpy as np

        y = _as_1d_float_array(signal, name="signal")
        if y.shape != self.flip_angle_deg.shape:
            raise ValueError(
                f"signal shape {y.shape} must match flip_angle_deg shape {self.flip_angle_deg.shape}"
            )

        alpha = np.deg2rad(self.flip_angle_deg) * float(self.b1)
        sin_a = np.sin(alpha)
        tan_a = np.tan(alpha)
        if np.any(sin_a == 0) or np.any(tan_a == 0):
            raise ValueError("invalid flip angles leading to sin/tan = 0")

        xdata = y / tan_a
        ydata = y / sin_a

        a = np.vstack([np.ones_like(xdata), xdata]).T
        intercept, slope = np.linalg.lstsq(a, ydata, rcond=None)[0]
        slope = float(slope)
        intercept = float(intercept)

        slope = min(max(slope, 1e-12), 1.0 - 1e-12)
        t1_s = -float(self.tr_s) / float(np.log(slope))
        m0 = intercept / (1.0 - slope)
        return {"m0": float(m0), "t1_s": float(t1_s)}

