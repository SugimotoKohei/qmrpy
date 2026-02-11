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
class B0DualEcho:
    """Dual-echo phase-difference B0 mapping.

    The model is:
        phi(TE) = phase0 + 2*pi*b0_hz*TE

    where ``TE`` is in seconds.
    """

    te1_ms: float
    te2_ms: float
    unwrap_phase: bool = False

    def __post_init__(self) -> None:
        if float(self.te1_ms) <= 0 or float(self.te2_ms) <= 0:
            raise ValueError("te1_ms and te2_ms must be > 0")
        if float(self.te1_ms) == float(self.te2_ms):
            raise ValueError("te1_ms and te2_ms must be different")

    def forward(self, *, b0_hz: float, phase0_rad: float = 0.0) -> NDArray[np.float64]:
        import numpy as np

        te_s = np.array([self.te1_ms, self.te2_ms], dtype=np.float64) / 1000.0
        return _wrap_phase(phase0_rad + 2.0 * np.pi * float(b0_hz) * te_s)

    def fit(self, signal: ArrayLike) -> dict[str, float]:
        import numpy as np

        phase = _as_phase(signal)
        if phase.ndim != 1 or phase.shape != (2,):
            raise ValueError("signal must be shape (2,) with dual-echo phase/complex values")

        dt_s = (float(self.te2_ms) - float(self.te1_ms)) / 1000.0
        dphi = _wrap_phase(np.asarray(phase[1] - phase[0], dtype=np.float64))
        b0_hz = float(dphi / (2.0 * np.pi * dt_s))

        te1_s = float(self.te1_ms) / 1000.0
        te2_s = float(self.te2_ms) / 1000.0
        phase0_rad = float(_wrap_phase(np.asarray(phase[0] - 2.0 * np.pi * b0_hz * te1_s, dtype=np.float64)))

        pred = phase0_rad + 2.0 * np.pi * b0_hz * np.array([te1_s, te2_s], dtype=np.float64)
        residual = float(np.sqrt(np.mean(_wrap_phase(pred - phase) ** 2)))
        return {
            "b0_hz": b0_hz,
            "phase0_rad": phase0_rad,
            "residual": residual,
        }

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
            raise ValueError("data last dim must be 2 for dual-echo input")

        spatial_shape = arr.shape[:-1]
        phase = _as_phase(arr)

        if self.unwrap_phase and len(spatial_shape) == 3:
            from qmrpy.models.qsm import unwrap_phase_laplacian

            phase0 = unwrap_phase_laplacian(np.asarray(phase[..., 0], dtype=np.float64))
            phase1 = unwrap_phase_laplacian(np.asarray(phase[..., 1], dtype=np.float64))
            dphi = phase1 - phase0
            phase_e1 = phase0
            phase_e2 = phase1
        else:
            phase_e1 = np.asarray(phase[..., 0], dtype=np.float64)
            phase_e2 = np.asarray(phase[..., 1], dtype=np.float64)
            dphi = _wrap_phase(phase_e2 - phase_e1)

        dt_s = (float(self.te2_ms) - float(self.te1_ms)) / 1000.0
        te1_s = float(self.te1_ms) / 1000.0
        te2_s = float(self.te2_ms) / 1000.0

        b0_hz = dphi / (2.0 * np.pi * dt_s)
        phase0 = _wrap_phase(phase_e1 - 2.0 * np.pi * b0_hz * te1_s)

        pred1 = phase0 + 2.0 * np.pi * b0_hz * te1_s
        pred2 = phase0 + 2.0 * np.pi * b0_hz * te2_s
        residual = np.sqrt(0.5 * (_wrap_phase(pred1 - phase_e1) ** 2 + _wrap_phase(pred2 - phase_e2) ** 2))

        resolved_mask = resolve_mask(mask, np.asarray(arr, dtype=np.float64) if np.isrealobj(arr) else np.abs(arr))
        if resolved_mask is None:
            valid = np.ones(spatial_shape, dtype=bool)
        else:
            if resolved_mask.shape != spatial_shape:
                raise ValueError(f"mask shape {resolved_mask.shape} must match spatial shape {spatial_shape}")
            valid = resolved_mask.astype(bool)

        out_b0 = np.full(spatial_shape, np.nan, dtype=np.float64)
        out_phase0 = np.full(spatial_shape, np.nan, dtype=np.float64)
        out_res = np.full(spatial_shape, np.nan, dtype=np.float64)

        finite = valid & np.isfinite(phase_e1) & np.isfinite(phase_e2)
        out_b0[finite] = b0_hz[finite]
        out_phase0[finite] = phase0[finite]
        out_res[finite] = residual[finite]

        _ = n_jobs
        _ = verbose
        return {"b0_hz": out_b0, "phase0_rad": out_phase0, "residual": out_res}
