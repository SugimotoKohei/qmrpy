from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import ArrayLike, NDArray
else:
    ArrayLike = Any  # type: ignore[misc,assignment]
    NDArray = Any  # type: ignore[misc,assignment]


def _as_phase(values: ArrayLike) -> NDArray[np.float64]:
    import numpy as np

    arr = np.asarray(values)
    if np.iscomplexobj(arr):
        return np.angle(arr).astype(np.float64)
    return np.asarray(arr, dtype=np.float64)


def _wrap_phase(phi_rad: NDArray[np.float64]) -> NDArray[np.float64]:
    import numpy as np

    return np.angle(np.exp(1j * phi_rad)).astype(np.float64)


@dataclass(frozen=True, slots=True)
class B1BlochSiegert:
    """Bloch-Siegert phase-shift B1 mapping.

    For two acquisitions with opposite off-resonance pulse frequency:
        dphi = phi_plus - phi_minus = 2 * k_bs * b1^2

    where ``k_bs`` is a calibration constant.
    """

    k_bs_rad_per_b1sq: float = 1.0

    def __post_init__(self) -> None:
        if float(self.k_bs_rad_per_b1sq) <= 0:
            raise ValueError("k_bs_rad_per_b1sq must be > 0")

    def forward(self, *, b1: float, phase0_rad: float = 0.0) -> NDArray[np.float64]:
        import numpy as np

        if b1 < 0:
            raise ValueError("b1 must be >= 0")
        delta = float(self.k_bs_rad_per_b1sq) * float(b1) ** 2
        return _wrap_phase(np.asarray([phase0_rad + delta, phase0_rad - delta], dtype=np.float64))

    def fit(self, signal: ArrayLike) -> dict[str, float]:
        import numpy as np

        phase = _as_phase(signal)
        if phase.ndim != 1 or phase.shape != (2,):
            raise ValueError("signal must be shape (2,) as [plus, minus] phase/complex values")

        dphi = float(_wrap_phase(np.asarray(phase[0] - phase[1], dtype=np.float64)))
        b1_raw = float(np.sqrt(np.abs(dphi) / (2.0 * float(self.k_bs_rad_per_b1sq))))
        spurious = 0.0 if np.isfinite(b1_raw) else 1.0
        if b1_raw < 1e-6:
            spurious = 1.0

        return {"b1_raw": b1_raw, "spurious": float(spurious)}

    def fit_image(
        self,
        signal: ArrayLike,
        *,
        mask: ArrayLike | str | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
    ) -> dict[str, Any]:
        import numpy as np

        from qmrpy._mask import resolve_mask

        arr = np.asarray(signal)
        if arr.ndim == 1:
            if mask is not None:
                raise ValueError("mask must be None for 1D data")
            return self.fit(arr)
        if arr.shape[-1] != 2:
            raise ValueError("data must have last dim=2 as [plus, minus]")

        spatial_shape = arr.shape[:-1]
        phase = _as_phase(arr)

        dphi = _wrap_phase(np.asarray(phase[..., 0] - phase[..., 1], dtype=np.float64))
        b1 = np.sqrt(np.abs(dphi) / (2.0 * float(self.k_bs_rad_per_b1sq)))

        resolved_mask = resolve_mask(mask, np.asarray(arr, dtype=np.float64) if np.isrealobj(arr) else np.abs(arr))
        if resolved_mask is None:
            valid = np.ones(spatial_shape, dtype=bool)
        else:
            if resolved_mask.shape != spatial_shape:
                raise ValueError(f"mask shape {resolved_mask.shape} must match spatial shape {spatial_shape}")
            valid = resolved_mask.astype(bool)

        out_b1 = np.full(spatial_shape, np.nan, dtype=np.float64)
        out_spurious = np.ones(spatial_shape, dtype=np.float64)

        finite = valid & np.isfinite(b1)
        out_b1[finite] = b1[finite]
        out_spurious[finite] = np.where(b1[finite] >= 1e-6, 0.0, 1.0)

        _ = n_jobs
        _ = verbose
        return {"b1_raw": out_b1, "spurious": out_spurious}
