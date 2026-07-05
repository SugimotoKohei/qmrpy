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
class T1Rho:
    """Mono-exponential T1rho spin-lock relaxometry model.

    Signal model:
        S(TSL) = m0 * exp(-TSL / T1rho)

    Units
    -----
    tsl_ms : milliseconds
    t1rho_ms : milliseconds

    References
    ----------
    .. [1] Witschey WRT, Borthakur A, Elliott MA, et al. (2007).
           T1rho-prepared balanced gradient echo for rapid 3D T1rho MRI.
           Journal of Magnetic Resonance Imaging, 26(6):1481-1488.
    .. [2] Wang L, Regatte RR (2015). T1rho MRI of human musculoskeletal
           system. Journal of Magnetic Resonance Imaging, 41(3):586-600.
    """

    tsl_ms: ArrayLike

    def __post_init__(self) -> None:
        import numpy as np

        tsl_array = _as_1d_float_array(self.tsl_ms, name="tsl_ms")
        if np.any(tsl_array < 0):
            raise ValueError("tsl_ms must be non-negative")
        object.__setattr__(self, "tsl_ms", tsl_array)

    def forward(self, *, m0: float, t1rho_ms: float) -> NDArray[np.float64]:
        import numpy as np

        if not np.isfinite(m0):
            raise ValueError("m0 must be finite")
        if not np.isfinite(t1rho_ms) or t1rho_ms <= 0:
            raise ValueError("t1rho_ms must be finite and > 0")
        return float(m0) * np.exp(-self.tsl_ms / float(t1rho_ms))

    def fit(
        self,
        signal: ArrayLike,
        *,
        m0_init: float | None = None,
        t1rho_init_ms: float | None = None,
        bounds_ms: tuple[tuple[float, float], tuple[float, float]] | None = None,
    ) -> dict[str, float]:
        """Fit m0 and T1rho using non-linear least squares.

        Parameters
        ----------
        signal : array-like
            1D signal samples at the model's ``tsl_ms``.
        m0_init : float, optional
            Initial guess for m0.
        t1rho_init_ms : float, optional
            Initial guess for T1rho in milliseconds.
        bounds_ms : tuple of tuple, optional
            Bounds as ``((m0_min, t1rho_min_ms), (m0_max, t1rho_max_ms))``.

        Returns
        -------
        dict
            Fit results with keys ``m0`` and ``t1rho_ms``.
        """
        import numpy as np
        from scipy.optimize import least_squares

        y = _as_1d_float_array(signal, name="signal")
        if y.shape != self.tsl_ms.shape:
            raise ValueError(f"signal shape {y.shape} must match tsl_ms shape {self.tsl_ms.shape}")

        y_abs = np.abs(y)
        scale = float(np.max(y_abs)) if np.max(y_abs) > 0 else 1.0
        y_norm = y_abs / scale

        if m0_init is None:
            m0_init = float(y_norm[0]) if y_norm.size else 1.0
        if t1rho_init_ms is None:
            t1rho_init_ms = _initial_t1rho_ms(self.tsl_ms, y_norm)
        if not np.isfinite(m0_init) or m0_init < 0:
            m0_init = max(float(y_norm[0]), 0.0)
        if not np.isfinite(t1rho_init_ms) or t1rho_init_ms <= 0:
            t1rho_init_ms = 50.0

        if bounds_ms is None:
            lower = (0.0, 1e-6)
            upper = (np.inf, np.inf)
        else:
            lower, upper = bounds_ms

        lower_arr = np.asarray(lower, dtype=np.float64)
        upper_arr = np.asarray(upper, dtype=np.float64)
        if lower_arr.shape != (2,) or upper_arr.shape != (2,):
            raise ValueError("bounds_ms must be ((m0_min, t1rho_min_ms), (m0_max, t1rho_max_ms))")
        x0 = np.array([m0_init, t1rho_init_ms], dtype=np.float64)
        x0 = np.minimum(np.maximum(x0, lower_arr), upper_arr)

        def residuals(params: NDArray[np.float64]) -> NDArray[np.float64]:
            m0_value = float(params[0])
            t1rho_value = float(params[1])
            return (m0_value * np.exp(-self.tsl_ms / t1rho_value)) - y_norm

        result = least_squares(residuals, x0=x0, bounds=(lower_arr, upper_arr))
        m0_hat, t1rho_hat = result.x
        return {"m0": float(m0_hat) * scale, "t1rho_ms": float(t1rho_hat)}

    def fit_image(
        self,
        signal: ArrayLike,
        *,
        mask: ArrayLike | str | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Voxel-wise T1rho fitting for image data.

        Parameters
        ----------
        signal : array-like
            Input array with last dim as spin-lock times.
        mask : array-like, "otsu", or None
            Spatial mask. If "otsu", Otsu thresholding is applied.
        n_jobs : int, default=1
            Number of parallel jobs. -1 uses all CPUs.
        verbose : bool, default=False
            If True, show progress bar and log info.
        **kwargs
            Passed to ``fit``.

        Returns
        -------
        dict
            Dict of parameter maps.
        """
        import numpy as np

        from qmrpy._mask import resolve_mask
        from qmrpy._parallel import parallel_fit

        arr = np.asarray(signal, dtype=np.float64)
        if arr.ndim == 1:
            if mask is not None:
                raise ValueError("mask must be None for 1D data")
            return self.fit(arr, **kwargs)
        if arr.shape[-1] != self.tsl_ms.shape[0]:
            raise ValueError(
                f"data last dim {arr.shape[-1]} must match tsl_ms length {self.tsl_ms.shape[0]}"
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
            return self.fit(signal_1d, **kwargs)

        return parallel_fit(
            fit_func,
            flat,
            mask_flat,
            ["m0", "t1rho_ms"],
            spatial_shape,
            n_jobs=n_jobs,
            verbose=verbose,
            desc="T1Rho",
        )


def _initial_t1rho_ms(tsl_ms: NDArray[np.float64], signal_norm: NDArray[np.float64]) -> float:
    import numpy as np

    positive = np.flatnonzero(signal_norm > 0)
    if positive.size < 2:
        return 50.0
    first = int(positive[0])
    last = int(positive[-1])
    if first == last or signal_norm[last] == signal_norm[first]:
        return 50.0
    delta_t = float(tsl_ms[last] - tsl_ms[first])
    ratio = float(signal_norm[last] / signal_norm[first])
    if delta_t <= 0 or ratio <= 0 or ratio >= 1:
        return 50.0
    return -delta_t / float(np.log(ratio))
