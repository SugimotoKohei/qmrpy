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


def _init_t2star(te_ms: NDArray[np.float64], y_abs: NDArray[np.float64]) -> float:
    import numpy as np

    if y_abs.size < 2:
        return 30.0
    y0 = float(y_abs[0])
    y1 = float(y_abs[-1])
    dt = float(te_ms[-1] - te_ms[0])
    if y0 <= 0 or y1 <= 0 or dt <= 0:
        return 30.0
    ratio = y1 / y0
    if ratio <= 0 or ratio == 1:
        return 30.0
    t2 = -dt / float(np.log(ratio))
    if not np.isfinite(t2) or t2 <= 0:
        return 30.0
    return float(t2)


@dataclass(frozen=True, slots=True)
class T2StarMonoR2:
    """Mono-exponential T2* (R2*) mapping from magnitude data."""

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

    def forward(self, *, s0: float, t2star_ms: float, offset: float = 0.0) -> NDArray[np.float64]:
        import numpy as np

        if t2star_ms <= 0:
            raise ValueError("t2star_ms must be > 0")
        return float(s0) * np.exp(-self.te_ms / float(t2star_ms)) + float(offset)

    def fit(
        self,
        signal: ArrayLike,
        *,
        offset_term: bool = False,
        s0_init: float | None = None,
        t2star_init_ms: float | None = None,
    ) -> dict[str, float]:
        import numpy as np
        from scipy.optimize import least_squares

        y = np.asarray(signal)
        if y.ndim != 1 or y.shape != self.te_ms.shape:
            raise ValueError(f"signal shape {y.shape} must match te_ms shape {self.te_ms.shape}")

        y_abs = np.abs(np.asarray(y, dtype=np.complex128)).astype(np.float64)
        if np.max(y_abs) <= 0:
            return {
                "s0": 0.0,
                "t2star_ms": float("nan"),
                "r2star_hz": float("nan"),
                "res_rmse": float("nan"),
            }

        if s0_init is None:
            s0_init = float(y_abs[0])
        if t2star_init_ms is None:
            t2star_init_ms = _init_t2star(self.te_ms, y_abs)

        def residuals(params: NDArray[np.float64]) -> NDArray[np.float64]:
            s0_val = float(params[0])
            t2_val = float(params[1])
            if offset_term:
                off = float(params[2])
                pred = s0_val * np.exp(-self.te_ms / t2_val) + off
            else:
                pred = s0_val * np.exp(-self.te_ms / t2_val)
            return pred - y_abs

        if offset_term:
            x0 = np.array([s0_init, t2star_init_ms, 0.0], dtype=np.float64)
            lower = np.array([0.0, 1e-6, -np.inf], dtype=np.float64)
            upper = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
            result = least_squares(residuals, x0=x0, bounds=(lower, upper))
            s0_hat, t2_hat, off_hat = result.x
            pred = s0_hat * np.exp(-self.te_ms / t2_hat) + off_hat
            rmse = float(np.sqrt(np.mean((pred - y_abs) ** 2)))
            return {
                "s0": float(s0_hat),
                "t2star_ms": float(t2_hat),
                "r2star_hz": float(1000.0 / t2_hat),
                "offset": float(off_hat),
                "res_rmse": rmse,
            }

        x0 = np.array([s0_init, t2star_init_ms], dtype=np.float64)
        lower = np.array([0.0, 1e-6], dtype=np.float64)
        upper = np.array([np.inf, np.inf], dtype=np.float64)
        result = least_squares(residuals, x0=x0, bounds=(lower, upper))
        s0_hat, t2_hat = result.x
        pred = s0_hat * np.exp(-self.te_ms / t2_hat)
        rmse = float(np.sqrt(np.mean((pred - y_abs) ** 2)))
        return {
            "s0": float(s0_hat),
            "t2star_ms": float(t2_hat),
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
        **kwargs: Any,
    ) -> dict[str, Any]:
        import numpy as np

        from qmrpy._mask import resolve_mask
        from qmrpy._parallel import parallel_fit

        arr = np.asarray(signal)
        if arr.ndim == 1:
            if mask is not None:
                raise ValueError("mask must be None for 1D data")
            return self.fit(arr, **kwargs)
        if arr.shape[-1] != self.te_ms.shape[0]:
            raise ValueError(
                f"data last dim {arr.shape[-1]} must match te_ms length {self.te_ms.shape[0]}"
            )

        spatial_shape = arr.shape[:-1]
        flat = arr.reshape((-1, arr.shape[-1]))

        resolved_mask = resolve_mask(mask, np.abs(np.asarray(arr, dtype=np.complex128)))
        if resolved_mask is None:
            mask_flat = np.ones((flat.shape[0],), dtype=bool)
        else:
            if resolved_mask.shape != spatial_shape:
                raise ValueError(f"mask shape {resolved_mask.shape} must match spatial shape {spatial_shape}")
            mask_flat = resolved_mask.reshape((-1,))

        offset_term = bool(kwargs.get("offset_term", False))
        output_keys = ["s0", "t2star_ms", "r2star_hz", "res_rmse"]
        if offset_term:
            output_keys.append("offset")

        def fit_func(signal_1d: NDArray[Any]) -> dict[str, float]:
            return self.fit(signal_1d, **kwargs)

        return parallel_fit(
            fit_func,
            flat,
            mask_flat,
            output_keys,
            spatial_shape,
            n_jobs=n_jobs,
            verbose=verbose,
            desc="T2StarMonoR2",
        )


@dataclass(frozen=True, slots=True)
class T2StarComplexR2:
    """Complex-valued T2* model with off-resonance estimation."""

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

    def forward(
        self,
        *,
        s0: float,
        t2star_ms: float,
        delta_f_hz: float = 0.0,
        phi0_rad: float = 0.0,
    ) -> NDArray[np.complex128]:
        import numpy as np

        if t2star_ms <= 0:
            raise ValueError("t2star_ms must be > 0")
        te_s = self.te_ms / 1000.0
        return (
            float(s0)
            * np.exp(-self.te_ms / float(t2star_ms))
            * np.exp(1j * (float(phi0_rad) + 2.0 * np.pi * float(delta_f_hz) * te_s))
        )

    def fit(self, signal: ArrayLike) -> dict[str, float]:
        import numpy as np
        from scipy.optimize import least_squares

        y = np.asarray(signal, dtype=np.complex128)
        if y.ndim != 1 or y.shape != self.te_ms.shape:
            raise ValueError(f"signal shape {y.shape} must match te_ms shape {self.te_ms.shape}")

        y_abs = np.abs(y)
        if np.max(y_abs) <= 0:
            return {
                "s0": 0.0,
                "t2star_ms": float("nan"),
                "r2star_hz": float("nan"),
                "delta_f_hz": float("nan"),
                "phi0_rad": float("nan"),
                "res_rmse": float("nan"),
            }

        s0_init = float(y_abs[0])
        t2_init = _init_t2star(self.te_ms, y_abs)
        te_s = self.te_ms / 1000.0
        phase_u = np.unwrap(np.angle(y))
        slope, intercept = np.linalg.lstsq(
            np.column_stack([te_s, np.ones_like(te_s)]), phase_u, rcond=None
        )[0]
        delta_f_init = float(slope / (2.0 * np.pi))
        phi0_init = float(intercept)

        def residuals(params: NDArray[np.float64]) -> NDArray[np.float64]:
            s0_val = float(params[0])
            t2_val = float(params[1])
            df_val = float(params[2])
            ph0_val = float(params[3])
            pred = self.forward(s0=s0_val, t2star_ms=t2_val, delta_f_hz=df_val, phi0_rad=ph0_val)
            return np.concatenate([(pred.real - y.real), (pred.imag - y.imag)])

        x0 = np.array([s0_init, t2_init, delta_f_init, phi0_init], dtype=np.float64)
        lower = np.array([0.0, 1e-6, -np.inf, -np.inf], dtype=np.float64)
        upper = np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float64)
        result = least_squares(residuals, x0=x0, bounds=(lower, upper))
        s0_hat, t2_hat, df_hat, ph0_hat = result.x

        pred = self.forward(
            s0=float(s0_hat),
            t2star_ms=float(t2_hat),
            delta_f_hz=float(df_hat),
            phi0_rad=float(ph0_hat),
        )
        rmse = float(np.sqrt(np.mean(np.abs(pred - y) ** 2)))

        return {
            "s0": float(s0_hat),
            "t2star_ms": float(t2_hat),
            "r2star_hz": float(1000.0 / t2_hat),
            "delta_f_hz": float(df_hat),
            "phi0_rad": float(ph0_hat),
            "res_rmse": rmse,
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

        spatial_shape = arr.shape[:-1]
        flat = arr.reshape((-1, arr.shape[-1]))

        resolved_mask = resolve_mask(mask, np.abs(np.asarray(arr, dtype=np.complex128)))
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
            ["s0", "t2star_ms", "r2star_hz", "delta_f_hz", "phi0_rad", "res_rmse"],
            spatial_shape,
            n_jobs=n_jobs,
            verbose=verbose,
            desc="T2StarComplexR2",
        )
