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
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


@dataclass(frozen=True, slots=True)
class MTR:
    """Magnetization transfer ratio model.

    Signal model:
        MTR = (S0 - Smt) / S0

    References
    ----------
    .. [1] Wolff SD, Balaban RS (1989). Magnetization transfer contrast (MTC)
           and tissue water proton relaxation in vivo. Magnetic Resonance in
           Medicine, 10(1):135-144.
    .. [2] Henkelman RM, Stanisz GJ, Graham SJ (2001). Magnetization transfer
           in MRI: a review. NMR in Biomedicine, 14(2):57-64.
    """

    def forward(self, *, s0: float, mtr: float) -> NDArray[np.float64]:
        import numpy as np

        if not np.isfinite(s0) or s0 <= 0:
            raise ValueError("s0 must be finite and > 0")
        if not np.isfinite(mtr):
            raise ValueError("mtr must be finite")
        return np.array([float(s0), float(s0) * (1.0 - float(mtr))], dtype=np.float64)

    def fit(self, signal: ArrayLike) -> dict[str, float]:
        """Fit MTR from ``[S0, Smt]``.

        Parameters
        ----------
        signal : array-like
            1D signal array ``[S0, Smt]``.

        Returns
        -------
        dict
            Result with ``mtr`` as a fraction and ``mtr_percent`` in percent.
        """
        y = _as_1d_float_array(signal, name="signal")
        if y.shape != (2,):
            raise ValueError("signal must be shape (2,) as [S0, Smt]")
        s0, smt = float(y[0]), float(y[1])
        if s0 <= 0:
            raise ValueError("S0 must be > 0")
        mtr = (s0 - smt) / s0
        return {"mtr": float(mtr), "mtr_percent": float(100.0 * mtr)}

    def fit_image(
        self,
        signal: ArrayLike,
        *,
        mask: ArrayLike | str | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Voxel-wise MTR calculation.

        Parameters
        ----------
        signal : array-like
            Input array with last dim 2 as ``[S0, Smt]``.
        mask : array-like, "otsu", or None
            Spatial mask. If "otsu", Otsu thresholding is applied.
        n_jobs : int, default=1
            Number of parallel jobs. -1 uses all CPUs.
        verbose : bool, default=False
            If True, show progress bar.
        """
        import numpy as np

        from qmrpy._mask import resolve_mask
        from qmrpy._parallel import parallel_fit

        _ = verbose
        arr = np.asarray(signal, dtype=np.float64)
        if arr.ndim == 1:
            if mask is not None:
                raise ValueError("mask must be None for 1D data")
            return self.fit(arr)
        if arr.shape[-1] != 2:
            raise ValueError("data must have last dim=2 as [S0, Smt]")

        spatial_shape = arr.shape[:-1]
        flat = arr.reshape((-1, 2))
        resolved_mask = resolve_mask(mask, arr)
        if resolved_mask is None:
            mask_flat = np.ones((flat.shape[0],), dtype=bool)
        else:
            if resolved_mask.shape != spatial_shape:
                raise ValueError(
                    f"mask shape {resolved_mask.shape} must match spatial shape {spatial_shape}"
                )
            mask_flat = resolved_mask.reshape((-1,))

        return parallel_fit(
            self.fit,
            flat,
            mask_flat,
            ["mtr", "mtr_percent"],
            spatial_shape,
            n_jobs=n_jobs,
            verbose=verbose,
            desc="MTR",
        )


@dataclass(frozen=True, slots=True)
class MTsat:
    """Magnetization transfer saturation using a FLASH T1/FA/TR correction.

    This model uses the spoiled-GRE steady-state signal without MT preparation
    as the reference and estimates the additional fractional saturation in the
    MT-weighted image:

        MTsat = 1 - Smt / S_GRE(M0, T1, alpha, TR)

    The reference signal follows the Ernst equation and corresponds to the
    first-order correction used by Helms-style MTsat mapping workflows.

    Units
    -----
    flip_angle_deg : degrees
    tr_ms : milliseconds
    t1_ms : milliseconds

    References
    ----------
    .. [1] Helms G, Dathe H, Kallenberg K, Dechent P (2008). High-resolution
           maps of magnetization transfer with inherent correction for RF
           inhomogeneity and T1 relaxation obtained from 3D FLASH MRI.
           Magnetic Resonance in Medicine, 60(6):1396-1407.
    .. [2] Weiskopf N, et al. (2013). Quantitative multi-parameter mapping of
           R1, PD*, MT, and R2* at 3T. Frontiers in Neuroscience, 7:95.
    """

    flip_angle_deg: float
    tr_ms: float

    def __post_init__(self) -> None:
        import numpy as np

        if not np.isfinite(self.flip_angle_deg) or self.flip_angle_deg <= 0:
            raise ValueError("flip_angle_deg must be finite and > 0")
        if not np.isfinite(self.tr_ms) or self.tr_ms <= 0:
            raise ValueError("tr_ms must be finite and > 0")

    def forward(self, *, m0: float, t1_ms: float, mtsat: float) -> NDArray[np.float64]:
        import numpy as np

        if not np.isfinite(mtsat):
            raise ValueError("mtsat must be finite")
        reference = _spoiled_gre_signal(
            m0=m0,
            t1_ms=t1_ms,
            flip_angle_deg=self.flip_angle_deg,
            tr_ms=self.tr_ms,
        )
        return np.array([reference * (1.0 - float(mtsat))], dtype=np.float64)

    def fit(self, signal: ArrayLike, *, m0: float, t1_ms: float) -> dict[str, float]:
        """Fit MTsat from an MT-weighted signal and T1/M0 estimates.

        Parameters
        ----------
        signal : array-like
            1D MT-weighted signal with shape ``(1,)``.
        m0 : float
            Proton-density-like equilibrium signal scale.
        t1_ms : float
            T1 value in milliseconds.

        Returns
        -------
        dict
            Result with ``mtsat`` as a fraction and ``mtsat_percent`` in percent.
        """
        y = _as_1d_float_array(signal, name="signal")
        if y.shape != (1,):
            raise ValueError("signal must be shape (1,) as [Smt]")
        reference = _spoiled_gre_signal(
            m0=m0,
            t1_ms=t1_ms,
            flip_angle_deg=self.flip_angle_deg,
            tr_ms=self.tr_ms,
        )
        if reference <= 0:
            raise ValueError("reference signal must be > 0")
        mtsat = 1.0 - float(y[0]) / reference
        return {"mtsat": float(mtsat), "mtsat_percent": float(100.0 * mtsat)}

    def fit_image(
        self,
        signal: ArrayLike,
        *,
        m0: ArrayLike,
        t1_ms: ArrayLike,
        mask: ArrayLike | str | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Voxel-wise MTsat calculation.

        Parameters
        ----------
        signal : array-like
            Input array with last dim 1 as ``[Smt]``.
        m0 : array-like
            M0 map matching the spatial shape.
        t1_ms : array-like
            T1 map in milliseconds matching the spatial shape.
        mask : array-like, "otsu", or None
            Spatial mask. If "otsu", Otsu thresholding is applied.
        n_jobs : int, default=1
            Number of parallel jobs. -1 uses all CPUs.
        verbose : bool, default=False
            If True, show progress bar.
        """
        import numpy as np

        from qmrpy._mask import resolve_mask
        from qmrpy._parallel import parallel_fit

        arr = np.asarray(signal, dtype=np.float64)
        if arr.ndim == 1:
            if mask is not None:
                raise ValueError("mask must be None for 1D data")
            return self.fit(arr, m0=float(m0), t1_ms=float(t1_ms))
        if arr.shape[-1] != 1:
            raise ValueError("data must have last dim=1 as [Smt]")

        spatial_shape = arr.shape[:-1]
        m0_arr = np.asarray(m0, dtype=np.float64)
        t1_arr = np.asarray(t1_ms, dtype=np.float64)
        if m0_arr.shape != spatial_shape:
            raise ValueError(f"m0 shape {m0_arr.shape} must match spatial shape {spatial_shape}")
        if t1_arr.shape != spatial_shape:
            raise ValueError(f"t1_ms shape {t1_arr.shape} must match spatial shape {spatial_shape}")

        flat = arr.reshape((-1, 1))
        m0_flat = m0_arr.reshape((-1,))
        t1_flat = t1_arr.reshape((-1,))

        resolved_mask = resolve_mask(mask, arr)
        if resolved_mask is None:
            mask_flat = np.ones((flat.shape[0],), dtype=bool)
        else:
            if resolved_mask.shape != spatial_shape:
                raise ValueError(
                    f"mask shape {resolved_mask.shape} must match spatial shape {spatial_shape}"
                )
            mask_flat = resolved_mask.reshape((-1,))

        def fit_func(signal_1d: NDArray[Any], index: int) -> dict[str, float]:
            return self.fit(signal_1d, m0=float(m0_flat[index]), t1_ms=float(t1_flat[index]))

        indexed_flat = np.concatenate(
            [flat, np.arange(flat.shape[0], dtype=np.float64)[:, None]], axis=1
        )

        def fit_indexed(row: NDArray[Any]) -> dict[str, float]:
            index = int(row[-1])
            return fit_func(row[:1], index)

        return parallel_fit(
            fit_indexed,
            indexed_flat,
            mask_flat,
            ["mtsat", "mtsat_percent"],
            spatial_shape,
            n_jobs=n_jobs,
            verbose=verbose,
            desc="MTsat",
        )


def _spoiled_gre_signal(*, m0: float, t1_ms: float, flip_angle_deg: float, tr_ms: float) -> float:
    import numpy as np

    if not np.isfinite(m0) or m0 <= 0:
        raise ValueError("m0 must be finite and > 0")
    if not np.isfinite(t1_ms) or t1_ms <= 0:
        raise ValueError("t1_ms must be finite and > 0")
    alpha = float(np.deg2rad(flip_angle_deg))
    e1 = float(np.exp(-float(tr_ms) / float(t1_ms)))
    denom = 1.0 - e1 * float(np.cos(alpha))
    if denom <= 0:
        raise ValueError("invalid spoiled-GRE reference denominator")
    return float(m0) * float(np.sin(alpha)) * (1.0 - e1) / denom
