from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .decaes_t2 import epg_decay_curve

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


@dataclass(frozen=True, slots=True)
class T2EMC:
    """EMC-like T2 fitting using an EPG dictionary."""

    n_te: int
    te_ms: float
    t1_ms: float = 1000.0
    alpha_deg: float = 180.0
    beta_deg: float = 180.0
    t2_grid_ms: ArrayLike | None = None
    b1_grid: ArrayLike | None = None

    def __post_init__(self) -> None:
        import numpy as np

        if int(self.n_te) < 2:
            raise ValueError("n_te must be >= 2")
        if float(self.te_ms) <= 0:
            raise ValueError("te_ms must be > 0")
        if float(self.t1_ms) <= 0:
            raise ValueError("t1_ms must be > 0")

        if self.t2_grid_ms is None:
            t2_grid = np.logspace(np.log10(10.0), np.log10(300.0), 80, dtype=np.float64)
        else:
            t2_grid = _as_1d_float_array(self.t2_grid_ms, name="t2_grid_ms")
        if np.any(t2_grid <= 0):
            raise ValueError("t2_grid_ms must be > 0")

        if self.b1_grid is None:
            b1_grid = np.linspace(0.7, 1.3, 31, dtype=np.float64)
        else:
            b1_grid = _as_1d_float_array(self.b1_grid, name="b1_grid")
        if np.any(b1_grid <= 0):
            raise ValueError("b1_grid must be > 0")

        object.__setattr__(self, "t2_grid_ms", t2_grid)
        object.__setattr__(self, "b1_grid", b1_grid)

    def _curve(self, *, t2_ms: float, b1: float) -> NDArray[np.float64]:
        return epg_decay_curve(
            etl=int(self.n_te),
            alpha_deg=float(self.alpha_deg) * float(b1),
            te_ms=float(self.te_ms),
            t2_ms=float(t2_ms),
            t1_ms=float(self.t1_ms),
            beta_deg=float(self.beta_deg),
            backend="decaes",
        )

    def forward(self, *, m0: float, t2_ms: float, b1: float = 1.0) -> NDArray[np.float64]:
        return float(m0) * self._curve(t2_ms=float(t2_ms), b1=float(b1))

    def fit(
        self,
        signal: ArrayLike,
        *,
        b1: float | None = None,
        estimate_b1: bool = False,
        b0_hz: float | None = None,
    ) -> dict[str, float]:
        import numpy as np

        y = _as_1d_float_array(signal, name="signal")
        if y.shape != (int(self.n_te),):
            raise ValueError(f"signal must be shape ({int(self.n_te)},)")

        _ = b0_hz

        y_abs = np.abs(y)
        if np.max(y_abs) <= 0:
            return {
                "m0": 0.0,
                "t2_ms": float("nan"),
                "b1": float("nan"),
                "res_rmse": float("nan"),
            }

        if b1 is not None and estimate_b1:
            raise ValueError("use either fixed b1 or estimate_b1=True")

        if estimate_b1:
            b1_candidates = np.asarray(self.b1_grid, dtype=np.float64)
        else:
            b1_fixed = 1.0 if b1 is None else float(b1)
            if b1_fixed <= 0:
                raise ValueError("b1 must be > 0")
            b1_candidates = np.array([b1_fixed], dtype=np.float64)

        best_rmse = np.inf
        best_m0 = float(np.max(y_abs))
        best_t2 = float(np.asarray(self.t2_grid_ms, dtype=np.float64)[0])
        best_b1 = float(b1_candidates[0])

        for b1_try in b1_candidates:
            for t2_try in np.asarray(self.t2_grid_ms, dtype=np.float64):
                curve = self._curve(t2_ms=float(t2_try), b1=float(b1_try))
                den = float(np.dot(curve, curve))
                if den <= 0:
                    continue
                m0_try = float(np.dot(y_abs, curve) / den)
                pred = m0_try * curve
                rmse = float(np.sqrt(np.mean((pred - y_abs) ** 2)))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_m0 = m0_try
                    best_t2 = float(t2_try)
                    best_b1 = float(b1_try)

        return {
            "m0": float(best_m0),
            "t2_ms": float(best_t2),
            "b1": float(best_b1),
            "res_rmse": float(best_rmse),
        }

    def fit_image(
        self,
        signal: ArrayLike,
        *,
        mask: ArrayLike | str | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        import numpy as np

        from qmrpy._mask import resolve_mask
        from qmrpy._parallel import parallel_fit

        arr = np.asarray(signal, dtype=np.float64)
        if arr.ndim == 1:
            if mask is not None:
                raise ValueError("mask must be None for 1D data")
            return self.fit(arr, **kwargs)

        if arr.shape[-1] != int(self.n_te):
            raise ValueError(f"data last dim {arr.shape[-1]} must match n_te {int(self.n_te)}")

        spatial_shape = arr.shape[:-1]
        flat = arr.reshape((-1, arr.shape[-1]))

        resolved_mask = resolve_mask(mask, arr)
        if resolved_mask is None:
            mask_flat = np.ones((flat.shape[0],), dtype=bool)
        else:
            if resolved_mask.shape != spatial_shape:
                raise ValueError(f"mask shape {resolved_mask.shape} must match spatial shape {spatial_shape}")
            mask_flat = resolved_mask.reshape((-1,))

        def fit_func(signal_1d: NDArray[Any]) -> dict[str, float]:
            return self.fit(signal_1d, **kwargs)

        return parallel_fit(
            fit_func,
            flat,
            mask_flat,
            ["m0", "t2_ms", "b1", "res_rmse"],
            spatial_shape,
            n_jobs=n_jobs,
            verbose=verbose,
            desc="T2EMC",
        )
