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
class B1DAM:
    """Double-Angle Method (DAM) for B1+ mapping (qMRLab: b1_dam).

    qMRLab formula:
        B1 = abs( acos(S(2α) / (2 S(α))) / α )

    where α is the nominal flip angle in radians and B1 is a multiplicative factor:
        FA_actual = B1 * FA_nominal

    Notes:
    - This implementation is a voxel-level core. qMRLab also supports smoothing/filtering
      and spurious-value handling; those are intentionally kept out of the core model.
    """

    alpha_deg: float

    def __post_init__(self) -> None:
        if self.alpha_deg <= 0:
            raise ValueError("alpha_deg must be > 0")

    def fit(self, signal: ArrayLike) -> dict[str, float]:
        """Fit B1 from DAM signals (alias of ``fit_raw``)."""
        return self.fit_raw(signal)

    def forward(self, *, m0: float, b1: float) -> NDArray[np.float64]:
        """Forward model returning [S(alpha), S(2*alpha)] under a simple sine model.

        This is a simplified signal model (TR→∞ assumption) used for synthetic runs/tests.
        """
        import numpy as np

        a1 = np.deg2rad(self.alpha_deg) * float(b1)
        a2 = np.deg2rad(self.alpha_deg * 2.0) * float(b1)
        return np.array([m0 * np.sin(a1), m0 * np.sin(a2)], dtype=np.float64)

    def fit_raw(self, signal: ArrayLike) -> dict[str, float]:
        """Fit B1 from [S(alpha), S(2*alpha)].

        Parameters
        ----------
        signal : array-like
            Signal array ``[S(alpha), S(2*alpha)]``.

        Returns
        -------
        dict
            ``b1_raw`` (raw estimate) and ``spurious`` flag.
        """
        import numpy as np

        y = _as_1d_float_array(signal, name="signal")
        if y.shape != (2,):
            raise ValueError("signal must be shape (2,) as [S(alpha), S(2*alpha)]")

        s1, s2 = float(y[0]), float(y[1])
        if abs(s1) < 1e-12:
            return {"b1_raw": float("nan"), "spurious": 1.0}

        ratio = s2 / (2.0 * s1)
        ratio = float(max(min(ratio, 1.0), -1.0))

        alpha_nom = float(np.deg2rad(self.alpha_deg))
        alpha_act = float(np.arccos(ratio))
        b1_raw = abs(alpha_act / alpha_nom)

        spurious = 1.0 if (not np.isfinite(b1_raw) or b1_raw < 0.5) else 0.0
        return {"b1_raw": float(b1_raw), "spurious": float(spurious)}

    def fit_image(
        self,
        signal: ArrayLike,
        *,
        mask: ArrayLike | str | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Vectorized DAM B1 estimation on an image/volume.

        Parameters
        ----------
        signal : array-like
            Input array with last dim 2 as ``[S(alpha), S(2*alpha)]``.
        mask : array-like, optional
            Spatial mask. If "otsu", Otsu thresholding is applied.
        n_jobs : int, default=1
            Number of parallel jobs. -1 uses all CPUs.
            Note: This model is fully vectorized and does not use n_jobs.
        verbose : bool, default=False
            If True, log info.

        Returns
        -------
        dict
            Maps for ``b1_raw`` and ``spurious``.
        """
        import logging

        import numpy as np

        from qmrpy._mask import resolve_mask

        logger = logging.getLogger("qmrpy")

        arr = np.asarray(signal, dtype=np.float64)
        if arr.ndim == 1:
            if mask is not None:
                raise ValueError("mask must be None for 1D data")
            return self.fit(arr)
        if arr.shape[-1] != 2:
            raise ValueError("data must have last dim=2 as [S(alpha), S(2*alpha)]")

        spatial_shape = arr.shape[:-1]
        s1 = arr[..., 0]
        s2 = arr[..., 1]

        resolved_mask = resolve_mask(mask, arr)
        if resolved_mask is None:
            m = np.ones(spatial_shape, dtype=bool)
        else:
            if resolved_mask.shape != spatial_shape:
                raise ValueError(f"mask shape {resolved_mask.shape} must match spatial shape {spatial_shape}")
            m = resolved_mask

        b1_raw = np.full(spatial_shape, np.nan, dtype=np.float64)
        spurious = np.ones(spatial_shape, dtype=np.float64)

        valid = m & np.isfinite(s1) & np.isfinite(s2) & (np.abs(s1) >= 1e-12)
        n_voxels = int(np.sum(valid))

        if verbose:
            logger.info("B1DAM: %d voxels (vectorized), shape=%s", n_voxels, spatial_shape)

        ratio = np.empty_like(s1, dtype=np.float64)
        ratio[valid] = s2[valid] / (2.0 * s1[valid])
        ratio = np.clip(ratio, -1.0, 1.0, out=ratio, where=valid)

        alpha_nom = float(np.deg2rad(self.alpha_deg))
        alpha_act = np.arccos(ratio, where=valid, out=np.full_like(ratio, np.nan))
        b1 = np.abs(alpha_act / alpha_nom)

        b1_raw[valid] = b1[valid]
        spurious[valid] = ((~np.isfinite(b1[valid])) | (b1[valid] < 0.5)).astype(np.float64)

        if verbose:
            logger.info("B1DAM complete: %d voxels processed", n_voxels)

        return {"b1_raw": b1_raw, "spurious": spurious}
