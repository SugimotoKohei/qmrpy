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
class B1Afi:
    """Actual Flip Angle Imaging (AFI) for B1+ mapping (qMRLab: b1_afi).

    Protocol:
        - nom_fa_deg: nominal excitation flip angle [deg]
        - tr1_s, tr2_s: repetition times (any consistent units; only ratio matters)

    Input signal ordering:
        - signal = [AFIData1, AFIData2]
          where AFIData1 corresponds to TR1 and AFIData2 corresponds to TR2.
    """

    nom_fa_deg: float
    tr1_s: float
    tr2_s: float

    def __post_init__(self) -> None:
        if self.nom_fa_deg <= 0:
            raise ValueError("nom_fa_deg must be > 0")
        if self.tr1_s <= 0 or self.tr2_s <= 0:
            raise ValueError("tr1_s and tr2_s must be > 0")

    def fit_raw(self, signal: ArrayLike) -> dict[str, float]:
        """Fit B1 from AFI signals [S(TR1), S(TR2)].

        Returns:
            - b1_raw: AFI B1 estimate
            - spurious: 1.0 if b1_raw < 0.5 else 0.0 (qMRLab convention)
        """
        import numpy as np

        y = _as_1d_float_array(signal, name="signal")
        if y.shape != (2,):
            raise ValueError("signal must be shape (2,) as [AFIData1, AFIData2]")

        s1 = float(y[0])
        s2 = float(y[1])
        if abs(s1) < 1e-12:
            return {"b1_raw": float("nan"), "spurious": 1.0}

        n = float(self.tr2_s) / float(self.tr1_s)
        r = abs(s2 / s1)

        # qMRLab behavior: r>1 treated as noise
        cos_arg = (r * n - 1.0) / (n - r) if r != n else 1.0
        if r > 1.0:
            cos_arg = 1.0

        cos_arg = float(np.clip(cos_arg, -1.0, 1.0))
        afi_rad = float(np.arccos(cos_arg))
        afi_deg = afi_rad * 180.0 / float(np.pi)
        b1_raw = afi_deg / float(self.nom_fa_deg)

        spurious = 1.0 if (not np.isfinite(b1_raw) or b1_raw < 0.5) else 0.0
        return {"b1_raw": float(b1_raw), "spurious": float(spurious)}

    def fit_image(self, data: ArrayLike, *, mask: ArrayLike | None = None) -> dict[str, Any]:
        """Vectorized AFI B1 estimation on an image/volume.

        Expects `data` shape (..., 2) as [AFIData1, AFIData2].
        """
        import numpy as np

        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim == 1:
            return self.fit_raw(arr)
        if arr.shape[-1] != 2:
            raise ValueError("data must have last dim=2 as [AFIData1, AFIData2]")

        spatial_shape = arr.shape[:-1]
        s1 = arr[..., 0]
        s2 = arr[..., 1]

        if mask is None:
            m = np.ones(spatial_shape, dtype=bool)
        else:
            m = np.asarray(mask, dtype=bool)
            if m.shape != spatial_shape:
                raise ValueError(f"mask shape {m.shape} must match spatial shape {spatial_shape}")

        b1_raw = np.full(spatial_shape, np.nan, dtype=np.float64)
        spurious = np.ones(spatial_shape, dtype=np.float64)

        valid = m & np.isfinite(s1) & np.isfinite(s2) & (np.abs(s1) >= 1e-12)
        n = float(self.tr2_s) / float(self.tr1_s)

        r = np.empty_like(s1, dtype=np.float64)
        r[valid] = np.abs(s2[valid] / s1[valid])

        cos_arg = np.full_like(r, 1.0, dtype=np.float64)
        denom = n - r
        mask_ok = valid & (denom != 0)
        cos_arg[mask_ok] = (r[mask_ok] * n - 1.0) / denom[mask_ok]
        cos_arg = np.clip(cos_arg, -1.0, 1.0)
        cos_arg = np.where(r > 1.0, 1.0, cos_arg)

        afi_deg = np.arccos(cos_arg) * 180.0 / float(np.pi)
        b1 = afi_deg / float(self.nom_fa_deg)

        b1_raw[valid] = b1[valid]
        spurious[valid] = np.where(np.isfinite(b1[valid]) & (b1[valid] >= 0.5), 0.0, 1.0)

        return {"b1_raw": b1_raw, "spurious": spurious}
