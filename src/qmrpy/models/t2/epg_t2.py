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
class EpgT2:
    """EPG-corrected mono-exponential T2 model for multi-echo spin-echo data.

    Signal model:
        S(TE) = m0 * EPG(TE, T2, T1, alpha, beta) + offset

    Units
    -----
    te_ms : milliseconds
    t2_ms : milliseconds
    t1_ms : milliseconds
    """

    n_te: int
    te_ms: float
    t1_ms: float = 1000.0
    alpha_deg: float = 180.0
    beta_deg: float = 180.0
    b1: float = 1.0
    epg_backend: str = "decaes"

    def __post_init__(self) -> None:
        if int(self.n_te) < 2:
            raise ValueError("n_te must be >= 2")
        if float(self.te_ms) <= 0:
            raise ValueError("te_ms must be > 0")
        if float(self.t1_ms) <= 0:
            raise ValueError("t1_ms must be > 0")
        if float(self.alpha_deg) <= 0:
            raise ValueError("alpha_deg must be > 0")
        if float(self.beta_deg) <= 0:
            raise ValueError("beta_deg must be > 0")
        if float(self.b1) <= 0:
            raise ValueError("b1 must be > 0")
        backend = str(self.epg_backend).lower().strip()
        if backend != "decaes":
            raise ValueError("epg_backend must be 'decaes'")
        object.__setattr__(self, "epg_backend", backend)

    def echotimes_ms(self) -> NDArray[np.float64]:
        import numpy as np

        return float(self.te_ms) * np.arange(1, int(self.n_te) + 1, dtype=np.float64)

    def _decay_curve(self, *, t2_ms: float, b1: float | None = None) -> NDArray[np.float64]:
        if t2_ms <= 0:
            raise ValueError("t2_ms must be > 0")
        b1_eff = float(self.b1 if b1 is None else b1)
        if b1_eff <= 0:
            raise ValueError("b1 must be > 0")
        return epg_decay_curve(
            etl=int(self.n_te),
            alpha_deg=float(self.alpha_deg) * b1_eff,
            te_ms=float(self.te_ms),
            t2_ms=float(t2_ms),
            t1_ms=float(self.t1_ms),
            beta_deg=float(self.beta_deg),
            backend=self.epg_backend,
        )

    def forward(
        self,
        *,
        m0: float,
        t2_ms: float,
        offset: float = 0.0,
        b1: float | None = None,
    ) -> NDArray[np.float64]:
        """Simulate EPG-corrected multi-echo signal."""
        curve = self._decay_curve(t2_ms=t2_ms, b1=b1)
        return float(m0) * curve + float(offset)

    def fit(
        self,
        signal: ArrayLike,
        *,
        drop_first_echo: bool = False,
        offset_term: bool = False,
        b1: float | None = None,
        m0_init: float | None = None,
        t2_init_ms: float | None = None,
        bounds_ms: tuple[tuple[float, float], tuple[float, float]] | None = None,
        max_nfev: int | None = None,
    ) -> dict[str, float]:
        """Fit m0 and T2 using non-linear least squares on EPG decay."""
        import numpy as np
        from scipy.optimize import least_squares

        y = _as_1d_float_array(signal, name="signal")
        if y.shape != (int(self.n_te),):
            raise ValueError(f"signal must be shape ({int(self.n_te)},), got {y.shape}")

        x = self.echotimes_ms()
        y_abs = np.abs(y)

        if drop_first_echo:
            if y_abs.size <= 2:
                raise ValueError("drop_first_echo is not valid for <=2 echoes")
            x = x[1:]
            y_abs = y_abs[1:]

        if np.max(y_abs) > 0:
            y_norm = y_abs / np.max(y_abs)
        else:
            y_norm = y_abs

        if m0_init is None:
            m0_init = float(y_abs[0]) if y_abs.size else 1.0
            if m0_init <= 0:
                m0_init = float(np.max(y_abs)) if y_abs.size else 1.0

        if t2_init_ms is None:
            if x.size >= 2 and y_norm.size >= 2 and y_norm[0] > 0 and y_norm[-1] > 0:
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

        if b1 is not None and float(b1) <= 0:
            raise ValueError("b1 must be > 0")

        def residuals(params: NDArray[np.float64]) -> NDArray[np.float64]:
            m0_value = float(params[0])
            t2_value = float(params[1])
            curve = self._decay_curve(t2_ms=t2_value, b1=b1)
            if drop_first_echo:
                curve = curve[1:]
            if offset_term:
                offset_value = float(params[2])
                return (m0_value * curve + offset_value) - y_abs
            return (m0_value * curve) - y_abs

        if offset_term:
            x0 = np.array([m0_init, float(t2_init_ms), 0.0], dtype=np.float64)
            lower3 = (float(lower[0]), float(lower[1]), -np.inf)
            upper3 = (float(upper[0]), float(upper[1]), np.inf)
            result = least_squares(
                residuals,
                x0=x0,
                bounds=(np.asarray(lower3, dtype=np.float64), np.asarray(upper3, dtype=np.float64)),
                max_nfev=max_nfev,
            )
            m0_hat, t2_hat, offset_hat = result.x
            return {"m0": float(m0_hat), "t2_ms": float(t2_hat), "offset": float(offset_hat)}

        result = least_squares(
            residuals,
            x0=np.array([m0_init, float(t2_init_ms)], dtype=np.float64),
            bounds=(np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)),
            max_nfev=max_nfev,
        )
        m0_hat, t2_hat = result.x
        return {"m0": float(m0_hat), "t2_ms": float(t2_hat)}

    def fit_image(
        self,
        signal: ArrayLike,
        *,
        mask: ArrayLike | str | None = None,
        b1_map: ArrayLike | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Voxel-wise fit on an image/volume.

        Parameters
        ----------
        signal : array-like
            Input array with last dim as echoes.
        mask : array-like, "otsu", or None
            Spatial mask. If "otsu", Otsu thresholding is applied.
        b1_map : array-like, optional
            B1 inhomogeneity map.
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
        import logging

        import numpy as np

        from qmrpy._mask import resolve_mask
        from qmrpy._parallel import parallel_fit

        logger = logging.getLogger("qmrpy")

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

        if b1_map is not None and "b1" in kwargs:
            raise ValueError("use either b1_map or b1, not both")
        b1_flat = None
        if b1_map is not None:
            b1_arr = np.asarray(b1_map, dtype=np.float64)
            if b1_arr.shape != spatial_shape:
                raise ValueError(f"b1_map shape {b1_arr.shape} must match spatial shape {spatial_shape}")
            b1_flat = b1_arr.reshape((-1,))

        offset_term = bool(kwargs.get("offset_term", False))
        output_keys = ["m0", "t2_ms"]
        if offset_term:
            output_keys.append("offset")

        # If b1_map is provided, we need custom parallel logic
        if b1_flat is not None:
            # Custom parallel fitting for b1_map
            from joblib import Parallel, delayed

            out: dict[str, Any] = {
                key: np.full(spatial_shape, np.nan, dtype=np.float64)
                for key in output_keys
            }
            indices = np.flatnonzero(mask_flat)
            n_voxels = len(indices)

            if n_voxels == 0:
                return out

            if verbose:
                logger.info("EpgT2: %d voxels, n_jobs=%s, shape=%s", n_voxels, n_jobs, spatial_shape)

            def fit_with_b1(idx: int) -> tuple[int, dict[str, float]]:
                return idx, self.fit(flat[idx], b1=float(b1_flat[idx]), **kwargs)

            if n_jobs == 1:
                iterator = indices
                if verbose:
                    from tqdm import tqdm
                    iterator = tqdm(indices, desc="EpgT2", unit="voxel")

                for idx in iterator:
                    res = self.fit(flat[idx], b1=float(b1_flat[idx]), **kwargs)
                    for key in output_keys:
                        if key in res:
                            out[key].flat[idx] = float(res[key])
            else:
                if verbose:
                    from tqdm import tqdm
                    results = Parallel(n_jobs=n_jobs)(
                        delayed(fit_with_b1)(idx)
                        for idx in tqdm(indices, desc="EpgT2", unit="voxel")
                    )
                else:
                    results = Parallel(n_jobs=n_jobs)(
                        delayed(fit_with_b1)(idx) for idx in indices
                    )
                for idx, res in results:
                    for key in output_keys:
                        if key in res:
                            out[key].flat[idx] = float(res[key])

            if verbose:
                logger.info("EpgT2 complete: %d voxels processed", n_voxels)

            return out
        else:
            # Use standard parallel_fit
            def fit_func(signal: NDArray[Any]) -> dict[str, float]:
                return self.fit(signal, **kwargs)

            return parallel_fit(
                fit_func, flat, mask_flat, output_keys, spatial_shape,
                n_jobs=n_jobs, verbose=verbose, desc="EpgT2"
            )
