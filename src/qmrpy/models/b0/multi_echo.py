from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import ArrayLike, NDArray
else:
    ArrayLike = Any  # type: ignore[misc,assignment]
    NDArray = Any  # type: ignore[misc,assignment]


def _as_1d_float_array(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={array.shape}")
    return array


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
class B0MultiEcho:
    """Multi-echo B0 mapping by linear phase regression."""

    te_ms: ArrayLike
    unwrap_phase: bool = True

    def __post_init__(self) -> None:
        import numpy as np

        te = _as_1d_float_array(self.te_ms, name="te_ms")
        if te.size < 2:
            raise ValueError("te_ms must contain at least 2 echoes")
        if np.any(te <= 0):
            raise ValueError("te_ms must be > 0")
        if np.any(np.diff(te) <= 0):
            raise ValueError("te_ms must be strictly increasing")
        object.__setattr__(self, "te_ms", te)

    def fit(self, signal: ArrayLike) -> dict[str, float]:
        import numpy as np

        phase = _as_phase(signal)
        if phase.ndim != 1 or phase.shape != self.te_ms.shape:
            raise ValueError(f"signal shape {phase.shape} must match te_ms shape {self.te_ms.shape}")

        te_s = self.te_ms / 1000.0
        phase_fit = np.unwrap(phase) if self.unwrap_phase else phase

        a = np.column_stack([te_s, np.ones_like(te_s)])
        slope, intercept = np.linalg.lstsq(a, phase_fit, rcond=None)[0]
        pred = slope * te_s + intercept
        residual = float(np.sqrt(np.mean((pred - phase_fit) ** 2)))

        b0_hz = float(slope / (2.0 * np.pi))
        return {
            "b0_hz": b0_hz,
            "phase0_rad": float(_wrap_phase(np.asarray(intercept, dtype=np.float64))),
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
        from qmrpy._parallel import parallel_fit

        arr = np.asarray(signal)
        if arr.ndim == 1:
            if mask is not None:
                raise ValueError("mask must be None for 1D data")
            return self.fit(arr)
        if arr.shape[-1] != self.te_ms.shape[0]:
            raise ValueError(
                f"data last dim {arr.shape[-1]} must match te_ms length {self.te_ms.shape[0]}"
            )

        phase = _as_phase(arr)
        spatial_shape = phase.shape[:-1]
        flat = phase.reshape((-1, phase.shape[-1]))

        resolved_mask = resolve_mask(mask, np.asarray(arr, dtype=np.float64) if np.isrealobj(arr) else np.abs(arr))
        if resolved_mask is None:
            mask_flat = np.ones((flat.shape[0],), dtype=bool)
        else:
            if resolved_mask.shape != spatial_shape:
                raise ValueError(f"mask shape {resolved_mask.shape} must match spatial shape {spatial_shape}")
            mask_flat = resolved_mask.reshape((-1,))

        def fit_func(signal_1d: NDArray[Any]) -> dict[str, float]:
            return self.fit(signal_1d)

        return parallel_fit(
            fit_func,
            flat,
            mask_flat,
            ["b0_hz", "phase0_rad", "residual"],
            spatial_shape,
            n_jobs=n_jobs,
            verbose=verbose,
            desc="B0MultiEcho",
        )
