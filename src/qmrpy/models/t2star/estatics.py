from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .r2star import T2StarMonoR2

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
class T2StarESTATICS:
    """Simplified ESTATICS-like joint R2* fitting.

    Input can be:
      - (n_echo,) for single-contrast magnitude data
      - (n_contrast, n_echo) for multi-contrast joint fitting
    """

    te_ms: ArrayLike

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

    def fit(self, signal: ArrayLike) -> dict[str, Any]:
        import numpy as np
        from scipy.optimize import least_squares

        arr = np.asarray(signal)

        if arr.ndim == 1:
            return T2StarMonoR2(te_ms=self.te_ms).fit(arr)

        if arr.ndim != 2 or arr.shape[-1] != self.te_ms.shape[0]:
            raise ValueError(
                "signal must be shape (n_echo,) or (n_contrast, n_echo) with matching n_echo"
            )

        y = np.abs(np.asarray(arr, dtype=np.complex128))
        n_contrast = y.shape[0]

        mono = T2StarMonoR2(te_ms=self.te_ms)
        t2_init_vals: list[float] = []
        s0_init_vals: list[float] = []
        for c in range(n_contrast):
            fit_c = mono.fit(y[c])
            t2_c = fit_c.get("t2star_ms")
            if isinstance(t2_c, float) and np.isfinite(t2_c):
                t2_init_vals.append(float(t2_c))
            s0_init_vals.append(float(fit_c.get("s0", float(np.max(y[c])))))

        t2_init = float(np.median(t2_init_vals)) if t2_init_vals else 30.0
        x0 = np.array([t2_init, *s0_init_vals], dtype=np.float64)

        te = self.te_ms

        def residuals(params: NDArray[np.float64]) -> NDArray[np.float64]:
            t2_val = float(params[0])
            s0_vals = params[1:]
            decay = np.exp(-te / t2_val)
            pred = s0_vals[:, None] * decay[None, :]
            return (pred - y).ravel()

        lower = np.array([1e-6] + [0.0] * n_contrast, dtype=np.float64)
        upper = np.array([np.inf] * (n_contrast + 1), dtype=np.float64)
        result = least_squares(residuals, x0=x0, bounds=(lower, upper))

        t2_hat = float(result.x[0])
        s0_hat = np.asarray(result.x[1:], dtype=np.float64)
        pred = s0_hat[:, None] * np.exp(-te / t2_hat)[None, :]
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))

        return {
            "s0": float(np.mean(s0_hat)),
            "s0_per_contrast": s0_hat,
            "t2star_ms": t2_hat,
            "r2star_hz": float(1000.0 / t2_hat),
            "res_rmse": rmse,
        }

    def fit_image(
        self,
        signal: ArrayLike,
        *,
        mask: ArrayLike | str | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
        multi_contrast: bool = False,
    ) -> dict[str, Any]:
        import numpy as np

        from qmrpy._mask import resolve_mask
        from qmrpy._parallel import parallel_fit

        arr = np.asarray(signal)
        if arr.ndim == 1:
            if mask is not None:
                raise ValueError("mask must be None for 1D data")
            return self.fit(arr)

        if not multi_contrast:
            return T2StarMonoR2(te_ms=self.te_ms).fit_image(
                arr, mask=mask, n_jobs=n_jobs, verbose=verbose
            )

        if arr.ndim < 3:
            raise ValueError("multi_contrast=True requires data shape (..., n_contrast, n_echo)")
        if arr.shape[-1] != self.te_ms.shape[0]:
            raise ValueError(
                f"data last dim {arr.shape[-1]} must match te_ms length {self.te_ms.shape[0]}"
            )

        spatial_shape = arr.shape[:-2]
        flat = arr.reshape((-1, arr.shape[-2], arr.shape[-1]))

        resolved_mask = resolve_mask(mask, np.mean(np.abs(np.asarray(arr, dtype=np.complex128)), axis=-2))
        if resolved_mask is None:
            mask_flat = np.ones((flat.shape[0],), dtype=bool)
        else:
            if resolved_mask.shape != spatial_shape:
                raise ValueError(f"mask shape {resolved_mask.shape} must match spatial shape {spatial_shape}")
            mask_flat = resolved_mask.reshape((-1,))

        def fit_func(signal_1d: NDArray[Any]) -> dict[str, float]:
            res = self.fit(signal_1d)
            return {
                "s0": float(res["s0"]),
                "t2star_ms": float(res["t2star_ms"]),
                "r2star_hz": float(res["r2star_hz"]),
                "res_rmse": float(res["res_rmse"]),
            }

        return parallel_fit(
            fit_func,
            flat,
            mask_flat,
            ["s0", "t2star_ms", "r2star_hz", "res_rmse"],
            spatial_shape,
            n_jobs=n_jobs,
            verbose=verbose,
            desc="T2StarESTATICS",
        )
