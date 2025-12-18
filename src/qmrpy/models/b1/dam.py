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
class B1Dam:
    """Double-Angle Method (DAM) for B1+ mapping (qMRLab: b1_dam).

    qMRLab formula:
        B1 = abs( acos(S(2α) / (2 S(α))) / α )

    where α is the nominal flip angle in radians and B1 is a multiplicative factor:
        FA_actual = B1 * FA_nominal

    Notes:
    - This implementation is a voxel-level core. qMRLab also supports smoothing/filtering
      and spurious-value handling; those are intentionally kept out of the core model.
    """

    alpha_deg: float

    def __post_init__(self) -> None:
        if self.alpha_deg <= 0:
            raise ValueError("alpha_deg must be > 0")

    def forward(self, *, m0: float, b1: float) -> Any:
        """Forward model returning [S(alpha), S(2*alpha)] under a simple sine model.

        This is a simplified signal model (TR→∞ assumption) used for synthetic runs/tests.
        """
        import numpy as np

        a1 = np.deg2rad(self.alpha_deg) * float(b1)
        a2 = np.deg2rad(self.alpha_deg * 2.0) * float(b1)
        return np.array([m0 * np.sin(a1), m0 * np.sin(a2)], dtype=np.float64)

    def fit_raw(self, signal: ArrayLike) -> dict[str, float]:
        """Fit B1 from [S(alpha), S(2*alpha)].

        Returns:
            - b1_raw: raw DAM estimate (may be <0.5 in very noisy data)
            - spurious: 1.0 if b1_raw < 0.5 else 0.0 (qMRLab convention)
        """
        import numpy as np

        y = _as_1d_float_array(signal, name="signal")
        if y.shape != (2,):
            raise ValueError("signal must be shape (2,) as [S(alpha), S(2*alpha)]")

        s1, s2 = float(y[0]), float(y[1])
        if abs(s1) < 1e-12:
            return {"b1_raw": float("nan"), "spurious": 1.0}

        ratio = s2 / (2.0 * s1)
        ratio = float(max(min(ratio, 1.0), -1.0))

        alpha_nom = float(np.deg2rad(self.alpha_deg))
        alpha_act = float(np.arccos(ratio))
        b1_raw = abs(alpha_act / alpha_nom)

        spurious = 1.0 if (not np.isfinite(b1_raw) or b1_raw < 0.5) else 0.0
        return {"b1_raw": float(b1_raw), "spurious": float(spurious)}

