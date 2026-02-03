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


@dataclass(frozen=True, slots=True)
class VFAT1:
    """Variable flip angle T1 model using FLASH/SPGR (qMRLab: vfa_t1).

    Signal model (FLASH/SPGR steady-state):
        S = M0 * sin(a) * (1 - E) / (1 - E * cos(a))
        E = exp(-TR / T1)

    Note: FLASH (Siemens) and SPGR (GE) refer to the same spoiled gradient echo sequence.

    Units (aligned to qMRLab protocol):
        - flip_angle_deg: degrees
        - tr_ms: milliseconds
        - t1_ms: milliseconds

    References
    ----------
    .. [1] Fram EK, et al. (1987). Rapid calculation of T1 using variable flip
           angle gradient refocused imaging. Magn Reson Imaging, 5(3):201-208.
    .. [2] Deoni SC, et al. (2003). Rapid combined T1 and T2 mapping using
           gradient recalled acquisition in the steady state. Magn Reson Med,
           49(3):515-526.
    .. [3] Boudreau M, et al. (2017). B1 mapping for bias-correction in
           quantitative T1 imaging of the brain at 3T using standard pulse
           sequences. J Magn Reson Imaging, 46(6):1673-1682.
    """

    flip_angle_deg: ArrayLike
    tr_ms: float
    b1: ArrayLike | float = 1.0

    def __post_init__(self) -> None:
        import numpy as np

        fa = _as_1d_float_array(self.flip_angle_deg, name="flip_angle_deg")
        if np.any(fa <= 0):
            raise ValueError("flip_angle_deg must be > 0")
        if self.tr_ms <= 0:
            raise ValueError("tr_ms must be > 0")
        b1 = np.asarray(self.b1, dtype=np.float64)
        if b1.ndim == 0:
            if float(b1) <= 0:
                raise ValueError("b1 must be > 0")
        elif b1.ndim == 1:
            if b1.shape != fa.shape:
                raise ValueError("b1 must be scalar or same shape as flip_angle_deg")
            if np.any(b1 <= 0):
                raise ValueError("b1 must be > 0")
        else:
            raise ValueError("b1 must be scalar or 1D")

        object.__setattr__(self, "flip_angle_deg", fa)
        object.__setattr__(self, "b1", b1)

    def forward(self, *, m0: float, t1_ms: float) -> NDArray[np.float64]:
        """Simulate FLASH/SPGR signal.

        Parameters
        ----------
        m0 : float
            Proton density scale.
        t1_ms : float
            T1 in milliseconds.

        Returns
        -------
        ndarray
            Simulated signal array.

        Raises
        ------
        ValueError
            If ``t1_ms`` <= 0.
        """
        import numpy as np

        if t1_ms <= 0:
            raise ValueError("t1_ms must be > 0")
        alpha = np.deg2rad(self.flip_angle_deg) * self.b1
        e = np.exp(-float(self.tr_ms) / float(t1_ms))
        return m0 * np.sin(alpha) * (1.0 - e) / (1.0 - e * np.cos(alpha))

    def fit(
        self,
        signal: ArrayLike,
        *,
        mask: ArrayLike | None = None,
        robust: bool = False,
        huber_k: float = 1.345,
        outlier_reject: bool = False,
        max_iter: int = 50,
        min_points: int = 2,
    ) -> dict[str, float]:
        """Fit by linearized FLASH/SPGR relation (alias of ``fit_linear``)."""
        return self.fit_linear(
            signal,
            mask=mask,
            robust=robust,
            huber_k=huber_k,
            outlier_reject=outlier_reject,
            max_iter=max_iter,
            min_points=min_points,
        )

    def fit_linear(
        self,
        signal: ArrayLike,
        *,
        mask: ArrayLike | None = None,
        robust: bool = False,
        huber_k: float = 1.345,
        outlier_reject: bool = False,
        max_iter: int = 50,
        min_points: int = 2,
    ) -> dict[str, float]:
        """Fit by linearized FLASH/SPGR relation (matches qMRLab approach).

        Linearization:
            y = S / sin(a)
            x = S / tan(a)
            y = intercept + slope * x
            slope = E, intercept = M0 * (1 - E)

        Parameters
        ----------
        signal : array-like
            Observed signal array.
        mask : array-like, optional
            Mask for valid points.
        robust : bool, optional
            Use robust regression if True.
        huber_k : float, optional
            Huber threshold for robust regression.
        outlier_reject : bool, optional
            Enable iterative outlier rejection.
        max_iter : int, optional
            Max iterations for outlier rejection.
        min_points : int, optional
            Minimum points required to fit.

        Returns
        -------
        dict
            Fit results dict including ``t1_ms`` and ``m0``.
        """
        import numpy as np

        y = _as_1d_float_array(signal, name="signal")
        if y.shape != self.flip_angle_deg.shape:
            raise ValueError(
                f"signal shape {y.shape} must match flip_angle_deg shape {self.flip_angle_deg.shape}"
            )

        alpha = np.deg2rad(self.flip_angle_deg) * self.b1
        sin_a = np.sin(alpha)
        tan_a = np.tan(alpha)
        if np.any(sin_a == 0) or np.any(tan_a == 0):
            raise ValueError("invalid flip angles leading to sin/tan = 0")

        xdata = y / tan_a
        ydata = y / sin_a

        valid = np.isfinite(xdata) & np.isfinite(ydata) & np.isfinite(y) & (y > 0)
        if mask is not None:
            m = np.asarray(mask)
            if m.shape != y.shape:
                raise ValueError("mask must have same shape as signal")
            valid = valid & (m.astype(bool))

        xdata_all = xdata
        ydata_all = ydata
        intercept, slope, final_valid = _fit_line_with_outlier_rejection(
            x=xdata_all,
            y=ydata_all,
            valid=valid,
            robust=robust,
            huber_k=huber_k,
            max_iter=max_iter,
            outlier_reject=outlier_reject,
            min_points=min_points,
        )

        slope = min(max(slope, 1e-12), 1.0 - 1e-12)
        t1_ms = -float(self.tr_ms) / float(np.log(slope))
        m0 = intercept / (1.0 - slope)
        return {
            "m0": float(m0),
            "t1_ms": float(t1_ms),
            "n_points": int(np.sum(final_valid)),
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
        """Voxel-wise linear VFA fit on an image/volume.

        Expects `signal` shape (..., n_fa) where n_fa == len(self.flip_angle_deg).

        Parameters
        ----------
        signal : array-like
            Input array with last dim as flip angles.
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
        if arr.shape[-1] != np.asarray(self.flip_angle_deg).shape[0]:
            raise ValueError(
                "data last dim must match flip_angle_deg length "
                f"({arr.shape[-1]} != {np.asarray(self.flip_angle_deg).shape[0]})"
            )

        spatial_shape = arr.shape[:-1]
        flat = arr.reshape((-1, arr.shape[-1]))

        resolved_mask = resolve_mask(mask, arr)
        if resolved_mask is None:
            mask_flat = np.ones((flat.shape[0],), dtype=bool)
        else:
            if resolved_mask.shape != spatial_shape:
                raise ValueError(f"mask shape {resolved_mask.shape} must match spatial shape {spatial_shape}")
            mask_flat = resolved_mask.reshape((-1,))

        output_keys = ["m0", "t1_ms", "n_points"]

        def fit_func(signal: NDArray[Any]) -> dict[str, float]:
            return self.fit(signal, **kwargs)

        return parallel_fit(
            fit_func, flat, mask_flat, output_keys, spatial_shape,
            n_jobs=n_jobs, verbose=verbose, desc="VFAT1"
        )


def _fit_line(
    x: Any,
    y: Any,
    *,
    robust: bool,
    huber_k: float,
    max_iter: int,
) -> tuple[float, float]:
    """Fit y = a + b x. If robust, use IRLS with Huber weights."""
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    a = np.vstack([np.ones_like(x), x]).T

    if not robust:
        intercept, slope = np.linalg.lstsq(a, y, rcond=None)[0]
        return float(intercept), float(slope)

    # Robust initialization (Theil–Sen): median of pairwise slopes
    slopes: list[float] = []
    n = int(x.size)
    for i in range(n):
        for j in range(i + 1, n):
            dx = float(x[i] - x[j])
            if dx == 0:
                continue
            slopes.append(float((y[i] - y[j]) / dx))
    if slopes:
        slope = float(np.median(np.asarray(slopes, dtype=np.float64)))
        intercept = float(np.median(y - slope * x))
    else:
        intercept, slope = np.linalg.lstsq(a, y, rcond=None)[0]
        intercept, slope = float(intercept), float(slope)

    w = np.ones_like(x, dtype=np.float64)
    for _ in range(max_iter):
        aw = a * w[:, None]
        yw = y * w
        intercept_new, slope_new = np.linalg.lstsq(aw, yw, rcond=None)[0]
        intercept_new = float(intercept_new)
        slope_new = float(slope_new)

        r = y - (intercept_new + slope_new * x)
        mad = float(np.median(np.abs(r - np.median(r))))
        scale = 1.4826 * mad if mad > 0 else float(np.std(r) + 1e-12)
        if scale <= 0:
            break
        c = float(huber_k) * scale
        abs_r = np.abs(r)
        w_new = np.ones_like(w)
        big = abs_r > c
        w_new[big] = c / abs_r[big]

        if np.allclose(w, w_new, rtol=0, atol=1e-6) and np.isclose(slope, slope_new, atol=1e-9):
            intercept, slope = intercept_new, slope_new
            break
        w = w_new
        intercept, slope = intercept_new, slope_new

    return float(intercept), float(slope)


def _fit_line_with_outlier_rejection(
    *,
    x: Any,
    y: Any,
    valid: Any,
    robust: bool,
    huber_k: float,
    max_iter: int,
    outlier_reject: bool,
    min_points: int,
) -> tuple[float, float, Any]:
    """Fit y = a + b x with optional outlier rejection.

    Outlier rejection is implemented with a small-n friendly exhaustive subset search:
    - enumerate all subsets with size >= min_points
    - enforce physical constraint: 0 < slope < 1 (E = exp(-TR/T1))
    - choose the largest subset that minimizes MAD residuals
    - refit using the selected subset (optionally robust)
    """
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid_mask = np.asarray(valid, dtype=bool).copy()

    def require_points() -> None:
        if int(np.sum(valid_mask)) < int(min_points):
            raise ValueError("not enough valid points for VFA linear fit")

    require_points()
    if not outlier_reject:
        x_fit = x[valid_mask]
        y_fit = y[valid_mask]
        intercept, slope = _fit_line(
            x_fit,
            y_fit,
            robust=robust,
            huber_k=huber_k,
            max_iter=max_iter,
        )
        return float(intercept), float(slope), valid_mask

    # Exhaustive subset search (small-n) with a physical constraint:
    # slope = E must satisfy 0 < slope < 1 for valid T1.
    # Choose the largest subset that yields a valid slope and minimizes MAD residuals.
    idx = np.flatnonzero(valid_mask)
    x_fit = x[idx]
    y_fit = y[idx]
    n = int(x_fit.size)
    if n > 20:
        # fallback: no rejection for large n
        intercept, slope = _fit_line(
            x_fit,
            y_fit,
            robust=robust,
            huber_k=huber_k,
            max_iter=max_iter,
        )
        return float(intercept), float(slope), valid_mask

    best_subset_mask: Any | None = None
    best_size = -1
    best_obj = float("inf")

    # iterate subsets by bitmask
    # note: exclude subsets smaller than min_points
    for bits in range(1 << n):
        k = int(bits.bit_count())
        if k < int(min_points):
            continue
        subset = np.array([(bits >> i) & 1 for i in range(n)], dtype=bool)
        x_sub = x_fit[subset]
        y_sub = y_fit[subset]
        # quick check for degenerate x
        if float(np.std(x_sub)) == 0.0:
            continue
        intercept_sub, slope_sub = _fit_line(
            x_sub,
            y_sub,
            robust=False,
            huber_k=huber_k,
            max_iter=max_iter,
        )
        if not (0.0 < float(slope_sub) < 1.0):
            continue
        r = y_sub - (float(intercept_sub) + float(slope_sub) * x_sub)
        mad = float(np.median(np.abs(r - np.median(r))))
        obj = mad
        if k > best_size or (k == best_size and obj < best_obj):
            best_size = k
            best_obj = obj
            best_subset_mask = subset

    if best_subset_mask is not None and best_size < n:
        # apply subset
        valid_mask[idx[~best_subset_mask]] = False
        require_points()

    # final fit after last rejection
    x_fit = x[valid_mask]
    y_fit = y[valid_mask]
    intercept, slope = _fit_line(
        x_fit,
        y_fit,
        robust=robust,
        huber_k=huber_k,
        max_iter=max_iter,
    )
    return float(intercept), float(slope), valid_mask
