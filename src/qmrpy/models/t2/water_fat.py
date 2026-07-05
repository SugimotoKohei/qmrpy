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
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


@dataclass(frozen=True, slots=True)
class T2WaterFat:
    """Two-pool dictionary T2 water/fat separation.

    Signal model:
        S(TE) = W * exp(-TE / T2w) + F * exp(-TE / T2f)

    For each candidate ``(T2w, T2f)`` pair, non-negative amplitudes ``W`` and
    ``F`` are estimated by NNLS and the pair with the lowest residual is kept.
    This is a compact two-pool approximation intended as the first practical
    water/fat separation layer; full multi-peak fat spectra and B0-dependent
    complex modeling are intentionally left for later extensions.

    References
    ----------
    .. [1] Graham SJ, Stanchev PL, Bronskill MJ (1996). Criteria for analysis
           of multicomponent tissue T2 relaxation data. Magnetic Resonance in
           Medicine, 35(3):370-378.
    .. [2] Poon CS, Henkelman RM (1992). Practical T2 quantitation for clinical
           applications. Journal of Magnetic Resonance Imaging, 2(5):541-553.
    """

    te_ms: ArrayLike

    def __post_init__(self) -> None:
        import numpy as np

        te = _as_1d_float_array(self.te_ms, name="te_ms")
        if np.any(te < 0):
            raise ValueError("te_ms must be non-negative")
        object.__setattr__(self, "te_ms", te)

    def forward(
        self,
        *,
        water_amplitude: float,
        fat_amplitude: float,
        water_t2_ms: float,
        fat_t2_ms: float,
    ) -> NDArray[np.float64]:
        import numpy as np

        if water_amplitude < 0 or fat_amplitude < 0:
            raise ValueError("water_amplitude and fat_amplitude must be >= 0")
        if water_t2_ms <= 0 or fat_t2_ms <= 0:
            raise ValueError("water_t2_ms and fat_t2_ms must be > 0")
        return float(water_amplitude) * np.exp(-self.te_ms / float(water_t2_ms)) + float(
            fat_amplitude
        ) * np.exp(-self.te_ms / float(fat_t2_ms))

    def fit(
        self,
        signal: ArrayLike,
        *,
        water_t2_grid_ms: ArrayLike,
        fat_t2_grid_ms: ArrayLike,
    ) -> dict[str, float]:
        """Fit two-pool water/fat amplitudes and T2 values."""
        import numpy as np
        from scipy.optimize import nnls

        y = _as_1d_float_array(signal, name="signal")
        if y.shape != self.te_ms.shape:
            raise ValueError(f"signal shape {y.shape} must match te_ms shape {self.te_ms.shape}")
        water_grid = _positive_grid(water_t2_grid_ms, name="water_t2_grid_ms")
        fat_grid = _positive_grid(fat_t2_grid_ms, name="fat_t2_grid_ms")

        best: dict[str, float] | None = None
        for water_t2 in water_grid:
            water_basis = np.exp(-self.te_ms / float(water_t2))
            for fat_t2 in fat_grid:
                fat_basis = np.exp(-self.te_ms / float(fat_t2))
                A = np.column_stack([water_basis, fat_basis])
                weights, _ = nnls(A, y)
                pred = A @ weights
                resid_l2 = float(np.linalg.norm(pred - y))
                if best is None or resid_l2 < best["resid_l2"]:
                    water_amp = float(weights[0])
                    fat_amp = float(weights[1])
                    total = water_amp + fat_amp
                    fat_fraction = fat_amp / total if total > 0 else float("nan")
                    best = {
                        "water_amplitude": water_amp,
                        "fat_amplitude": fat_amp,
                        "water_t2_ms": float(water_t2),
                        "fat_t2_ms": float(fat_t2),
                        "fat_fraction": float(fat_fraction),
                        "resid_l2": resid_l2,
                    }
        if best is None:
            raise ValueError("dictionary grid produced no valid entries")
        return best

    def fit_image(
        self,
        signal: ArrayLike,
        *,
        water_t2_grid_ms: ArrayLike,
        fat_t2_grid_ms: ArrayLike,
        mask: ArrayLike | str | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Voxel-wise two-pool T2 water/fat fitting."""
        import numpy as np

        from qmrpy._mask import resolve_mask
        from qmrpy._parallel import parallel_fit

        arr = np.asarray(signal, dtype=np.float64)
        if arr.ndim == 1:
            if mask is not None:
                raise ValueError("mask must be None for 1D data")
            return self.fit(arr, water_t2_grid_ms=water_t2_grid_ms, fat_t2_grid_ms=fat_t2_grid_ms)
        if arr.shape[-1] != self.te_ms.shape[0]:
            raise ValueError(
                f"data last dim {arr.shape[-1]} must match te_ms length {self.te_ms.shape[0]}"
            )

        spatial_shape = arr.shape[:-1]
        flat = arr.reshape((-1, arr.shape[-1]))
        resolved_mask = resolve_mask(mask, arr)
        if resolved_mask is None:
            mask_flat = np.ones((flat.shape[0],), dtype=bool)
        else:
            if resolved_mask.shape != spatial_shape:
                raise ValueError(
                    f"mask shape {resolved_mask.shape} must match spatial shape {spatial_shape}"
                )
            mask_flat = resolved_mask.reshape((-1,))

        def fit_func(signal_1d: NDArray[Any]) -> dict[str, float]:
            return self.fit(
                signal_1d,
                water_t2_grid_ms=water_t2_grid_ms,
                fat_t2_grid_ms=fat_t2_grid_ms,
            )

        return parallel_fit(
            fit_func,
            flat,
            mask_flat,
            [
                "water_amplitude",
                "fat_amplitude",
                "water_t2_ms",
                "fat_t2_ms",
                "fat_fraction",
                "resid_l2",
            ],
            spatial_shape,
            n_jobs=n_jobs,
            verbose=verbose,
            desc="T2WaterFat",
        )


def _positive_grid(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    import numpy as np

    grid = _as_1d_float_array(values, name=name)
    if np.any(grid <= 0):
        raise ValueError(f"{name} must be > 0")
    return grid
