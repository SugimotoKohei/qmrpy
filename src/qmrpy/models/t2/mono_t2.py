from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

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
class MonoT2:
    """Mono-exponential T2 relaxometry model.

    Signal model:
        S(TE) = m0 * exp(-TE / T2)

    Units
    -----
    te_ms : milliseconds
    t2_ms : milliseconds
    """

    te_ms: ArrayLike

    def __post_init__(self) -> None:
        import numpy as np

        te_array = _as_1d_float_array(self.te_ms, name="te_ms")
        if np.any(te_array < 0):
            raise ValueError("te_ms must be non-negative")
        object.__setattr__(self, "te_ms", te_array)

    def forward(self, *, m0: float, t2_ms: float) -> NDArray[np.float64]:
        import numpy as np

        if t2_ms <= 0:
            raise ValueError("t2_ms must be > 0")
        return m0 * np.exp(-self.te_ms / t2_ms)

    def fit(
        self,
        signal: ArrayLike,
        *,
        fit_type: str = "exponential",
        drop_first_echo: bool = False,
        offset_term: bool = False,
        m0_init: float | None = None,
        t2_init_ms: float | None = None,
        bounds_ms: tuple[tuple[float, float], tuple[float, float]] | None = None,
    ) -> dict[str, float]:
        """Fit m0 and T2 using non-linear least squares.

        Parameters
        ----------
        signal : array-like
            1D signal samples at the model's ``te_ms``.
        m0_init : float, optional
            Initial guess for m0.
        t2_init_ms : float, optional
            Initial guess for T2 in milliseconds.
        bounds_ms : tuple of tuple, optional
            Bounds as ``((m0_min, t2_min_ms), (m0_max, t2_max_ms))``.

        Returns
        -------
        dict
            Fit results with keys ``m0`` and ``t2_ms``.
        """
        import numpy as np
        from scipy.optimize import least_squares

        y = _as_1d_float_array(signal, name="signal")
        if y.shape != self.te_ms.shape:
            raise ValueError(f"signal shape {y.shape} must match te_ms shape {self.te_ms.shape}")

        x = self.te_ms
        if drop_first_echo:
            if y.size <= 2:
                raise ValueError("drop_first_echo is not valid for <=2 echoes")
            x = x[1:]
            y = y[1:]

        fit_type_norm = fit_type.lower().strip()
        if fit_type_norm not in {"exponential", "linear"}:
            raise ValueError("fit_type must be 'exponential' or 'linear'")

        y_abs = np.abs(y)
        if np.max(y_abs) > 0:
            y_norm = y_abs / np.max(y_abs)
        else:
            y_norm = y_abs

        if m0_init is None:
            m0_init = float(y_norm[0]) * 1.5 if y_norm.size else 1.0
        if t2_init_ms is None:
            if x.size >= 2 and y_norm.size >= 2 and y_norm[0] > 0 and y_norm[-1] > 0:
                # qMRLab-like heuristic: use end-1 and first, robust to last-point noise
                ref_idx = -2 if y_norm.size >= 3 else -1
                dt = float(x[0] - x[ref_idx])
                ratio = float(y_norm[ref_idx] / y_norm[0])
                if ratio > 0 and ratio != 1:
                    t2_init_ms = dt / float(np.log(ratio))
                else:
                    t2_init_ms = 30.0
            else:
                t2_init_ms = 30.0
            if t2_init_ms <= 0 or np.isnan(t2_init_ms):
                t2_init_ms = 30.0
        if bounds_ms is None:
            lower = (0.0, 1e-6)
            upper = (np.inf, np.inf)
        else:
            lower, upper = bounds_ms

        if fit_type_norm == "linear":
            if np.any(y_abs <= 0):
                raise ValueError("linear fit requires strictly positive signal")
            # log(y) = log(m0) - x/t2
            a = np.vstack([np.ones_like(x), x]).T
            beta0, beta1 = np.linalg.lstsq(a, np.log(y_abs), rcond=None)[0]
            beta1 = float(beta1)
            if beta1 == 0:
                beta1 = float(np.finfo(float).eps)
            t2_hat = -1.0 / beta1
            m0_hat = float(np.exp(float(beta0)))
            return {"m0": m0_hat, "t2_ms": float(t2_hat)}

        def residuals(params: NDArray[np.float64]) -> NDArray[np.float64]:
            m0_value = float(params[0])
            t2_value = float(params[1])
            if offset_term:
                offset_value = float(params[2])
                return (m0_value * np.exp(-x / t2_value) + offset_value) - y_norm
            return (m0_value * np.exp(-x / t2_value)) - y_norm

        if offset_term:
            x0 = np.array([m0_init, t2_init_ms, 0.0], dtype=np.float64)
            lower3 = (float(lower[0]), float(lower[1]), -np.inf)
            upper3 = (float(upper[0]), float(upper[1]), np.inf)
            result = least_squares(
                residuals,
                x0=x0,
                bounds=(np.asarray(lower3, dtype=np.float64), np.asarray(upper3, dtype=np.float64)),
            )
            m0_hat, t2_hat, offset_hat = result.x
            return {"m0": float(m0_hat), "t2_ms": float(t2_hat), "offset": float(offset_hat)}

        result = least_squares(
            residuals,
            x0=np.array([m0_init, t2_init_ms], dtype=np.float64),
            bounds=(np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)),
        )
        m0_hat, t2_hat = result.x
        return {"m0": float(m0_hat), "t2_ms": float(t2_hat)}

    def fit_image(
        self,
        data: ArrayLike,
        *,
        mask: ArrayLike | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Voxel-wise fit on an image/volume.

        Parameters
        ----------
        data : array-like
            Input array with last dim as echoes.
        mask : array-like, optional
            Spatial mask.
        **kwargs
            Passed to ``fit``.

        Returns
        -------
        dict
            Dict of parameter maps.
        """
        import numpy as np

        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim == 1:
            return self.fit(arr, **kwargs)
        if arr.shape[-1] != self.te_ms.shape[0]:
            raise ValueError(
                f"data last dim {arr.shape[-1]} must match te_ms length {self.te_ms.shape[0]}"
            )

        spatial_shape = arr.shape[:-1]
        flat = arr.reshape((-1, arr.shape[-1]))

        if mask is None:
            mask_flat = np.ones((flat.shape[0],), dtype=bool)
        else:
            m = np.asarray(mask, dtype=bool)
            if m.shape != spatial_shape:
                raise ValueError(f"mask shape {m.shape} must match spatial shape {spatial_shape}")
            mask_flat = m.reshape((-1,))

        offset_term = bool(kwargs.get("offset_term", False))
        out: dict[str, Any] = {
            "m0": np.full(spatial_shape, np.nan, dtype=np.float64),
            "t2_ms": np.full(spatial_shape, np.nan, dtype=np.float64),
        }
        if offset_term:
            out["offset"] = np.full(spatial_shape, np.nan, dtype=np.float64)

        for idx in np.flatnonzero(mask_flat):
            res = self.fit(flat[idx], **kwargs)
            out["m0"].flat[idx] = float(res["m0"])
            out["t2_ms"].flat[idx] = float(res["t2_ms"])
            if offset_term and "offset" in res:
                out["offset"].flat[idx] = float(res["offset"])

        return out
