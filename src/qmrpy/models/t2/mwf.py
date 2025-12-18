from __future__ import annotations

from dataclasses import dataclass, field
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
class MultiComponentT2:
    """Multi-component T2 relaxometry (MWF) using NNLS.

    Signal model:
        S(TE) = sum( w_i * exp(-TE / T2_i) ) for i in 1..N_basis

    Protocol:
        te_ms: Echo times in milliseconds.

    Basis:
        t2_basis_ms: Fixed basis of T2 values.
                     If None provided in __init__, defaults to log-spaced 10ms to 2000ms (40 points).

    Notes
    -----
    This class is a minimal v0 translation aligned with qMRLab's `mwf` intent:
    we estimate a non-negative T2 spectrum via NNLS and derive summary metrics
    (MWF, T2MW, T2IEW) by integrating the spectrum over cutoff ranges.
    """

    te_ms: ArrayLike
    t2_basis_ms: ArrayLike | None = field(default=None)

    def __post_init__(self) -> None:
        import numpy as np

        te = _as_1d_float_array(self.te_ms, name="te_ms")
        if np.any(te < 0):
            raise ValueError("te_ms must be non-negative")
        object.__setattr__(self, "te_ms", te)

        if self.t2_basis_ms is None:
            basis = self.default_t2_basis_ms()
        else:
            basis = _as_1d_float_array(self.t2_basis_ms, name="t2_basis_ms")
            if np.any(basis <= 0):
                raise ValueError("t2_basis_ms must be > 0")

        object.__setattr__(self, "t2_basis_ms", basis)

    @staticmethod
    def default_t2_basis_ms(
        *, t2_min_ms: float = 10.0, t2_max_ms: float = 2000.0, n: int = 40
    ) -> NDArray[np.float64]:
        import numpy as np

        if t2_min_ms <= 0:
            raise ValueError("t2_min_ms must be > 0")
        if t2_max_ms <= t2_min_ms:
            raise ValueError("t2_max_ms must be > t2_min_ms")
        if n <= 1:
            raise ValueError("n must be > 1")
        return np.logspace(np.log10(float(t2_min_ms)), np.log10(float(t2_max_ms)), int(n))

    def _design_matrix(self) -> NDArray[np.float64]:
        import numpy as np

        # A shape: (n_te, n_basis)
        # A[i, j] = exp(-te[i] / t2_basis[j])
        return np.exp(-self.te_ms[:, None] / self.t2_basis_ms[None, :])

    def forward(self, *, weights: ArrayLike) -> NDArray[np.float64]:
        """Simulate signal from weights.

        weights: array of size (n_basis,) corresponding to t2_basis_ms.
        """
        import numpy as np

        w = _as_1d_float_array(weights, name="weights")
        if w.shape != (len(self.t2_basis_ms),):
            raise ValueError(f"weights must be size {len(self.t2_basis_ms)}, got {w.shape}")

        A = self._design_matrix()
        return A @ w

    def fit(
        self,
        signal: ArrayLike,
        *,
        regularization_alpha: float = 0.0,
        lower_cutoff_mw_ms: float | None = None,
        cutoff_ms: float = 40.0,
        upper_cutoff_iew_ms: float = 200.0,
        use_weighted_geometric_mean: bool = False,
    ) -> dict[str, Any]:
        """Fit T2 distribution using NNLS and compute MWF/T2MW/T2IEW from cutoffs.

        Args:
            signal: Observed signal array (n_te,).
            regularization_alpha: Tikhonov regularization parameter (default: 0.0).
                                  If > 0, solves: argmin ||Aw - y||^2 + alpha^2 ||w||^2
            lower_cutoff_mw_ms: Lower cutoff for MW integration. If None, uses 1.5 * first echo time,
                                following qMRLab's default (`lower_cutoff_MW = 1.5*FirstEcho`).
            cutoff_ms: MW/IEW cutoff time [ms] (qMRLab default: 40ms).
            upper_cutoff_iew_ms: Upper cutoff for IEW integration [ms] (qMRLab default bound: 200ms).
            use_weighted_geometric_mean: If True, compute T2MW/T2IEW as weighted geometric means.

        Returns:
            dict with:
                - weights: estimated weights (n_basis,)
                - t2_basis_ms: the basis used
                - mwf: Myelin Water Fraction (0..1) computed as MW / (MW + IEW) within cutoffs
                - t2mw_ms: mean T2 for MW compartment [ms]
                - t2iew_ms: mean T2 for IEW compartment [ms]
                - gmt2_ms: global geometric mean T2 over all weights [ms]
                - resid_l2: ||Aw - y||_2
        """
        import numpy as np
        from scipy.optimize import nnls

        y = _as_1d_float_array(signal, name="signal")
        if y.shape != self.te_ms.shape:
            raise ValueError(f"signal shape {y.shape} mismatch with te_ms {self.te_ms.shape}")

        if lower_cutoff_mw_ms is None:
            lower_cutoff_mw_ms = 1.5 * float(self.te_ms[0]) if self.te_ms.size else 0.0
        lower_cutoff_mw_ms = float(lower_cutoff_mw_ms)
        cutoff_ms = float(cutoff_ms)
        upper_cutoff_iew_ms = float(upper_cutoff_iew_ms)
        if lower_cutoff_mw_ms < 0:
            raise ValueError("lower_cutoff_mw_ms must be >= 0")
        if cutoff_ms <= lower_cutoff_mw_ms:
            raise ValueError("cutoff_ms must be > lower_cutoff_mw_ms")
        if upper_cutoff_iew_ms <= cutoff_ms:
            raise ValueError("upper_cutoff_iew_ms must be > cutoff_ms")

        A = self._design_matrix()

        if regularization_alpha > 0:
            # Tikhonov regularization: augment A and y
            # [     A      ] w = [ y ]
            # [ alpha * I  ]     [ 0 ]
            n_basis = len(self.t2_basis_ms)
            A_aug = np.vstack([A, regularization_alpha * np.eye(n_basis)])
            y_aug = np.concatenate([y, np.zeros(n_basis)])
            w_hat, _ = nnls(A_aug, y_aug)
        else:
            w_hat, _ = nnls(A, y)

        resid_l2 = float(np.linalg.norm(A @ w_hat - y))

        t2 = np.asarray(self.t2_basis_ms, dtype=np.float64)

        # qMRLab definition:
        #   MWF = sum(weights for T2 <= cutoff) / sum(all weights)
        mwf_mask = t2 <= cutoff_ms
        total_w = float(np.sum(w_hat))
        mw_w = float(np.sum(w_hat[mwf_mask]))
        mwf = (mw_w / total_w) if total_w > 0 else 0.0

        # Sub-ranges for compartment summaries (qMRLab uses lower_cutoff for T2MW only)
        mw_range_mask = (t2 >= lower_cutoff_mw_ms) & (t2 <= cutoff_ms)
        iew_mask = (t2 >= cutoff_ms) & (t2 <= upper_cutoff_iew_ms)
        iew_w = float(np.sum(w_hat[iew_mask]))

        def _weighted_mean(values: Any, weights: Any) -> float:
            wsum = float(np.sum(weights))
            if wsum <= 0:
                return float("nan")
            if use_weighted_geometric_mean:
                return float(np.exp(np.sum(weights * np.log(values)) / wsum))
            return float(np.sum(weights * values) / wsum)

        def _weighted_geometric_mean(values: Any, weights: Any) -> float:
            wsum = float(np.sum(weights))
            if wsum <= 0:
                return float("nan")
            return float(np.exp(np.sum(weights * np.log(values)) / wsum))

        t2mw_ms = _weighted_mean(t2[mw_range_mask], w_hat[mw_range_mask])
        t2iew_ms = _weighted_mean(t2[iew_mask], w_hat[iew_mask])
        gmt2_ms = _weighted_geometric_mean(t2, w_hat)

        return {
            "weights": w_hat,
            "t2_basis_ms": self.t2_basis_ms,
            "mwf": float(mwf),
            "t2mw_ms": float(t2mw_ms),
            "t2iew_ms": float(t2iew_ms),
            "gmt2_ms": float(gmt2_ms),
            "mw_weight": float(mw_w),
            "iew_weight": float(iew_w),
            "total_weight": float(total_w),
            "resid_l2": float(resid_l2),
        }

    def fit_image(
        self,
        data: ArrayLike,
        *,
        mask: ArrayLike | None = None,
        return_weights: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Voxel-wise NNLS fit on an image/volume.

        Expects `data` shape (..., n_te) where n_te == len(self.te_ms).
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

        out: dict[str, Any] = {
            "t2_basis_ms": np.asarray(self.t2_basis_ms, dtype=np.float64),
            "mwf": np.full(spatial_shape, np.nan, dtype=np.float64),
            "t2mw_ms": np.full(spatial_shape, np.nan, dtype=np.float64),
            "t2iew_ms": np.full(spatial_shape, np.nan, dtype=np.float64),
            "gmt2_ms": np.full(spatial_shape, np.nan, dtype=np.float64),
            "mw_weight": np.full(spatial_shape, np.nan, dtype=np.float64),
            "iew_weight": np.full(spatial_shape, np.nan, dtype=np.float64),
            "total_weight": np.full(spatial_shape, np.nan, dtype=np.float64),
            "resid_l2": np.full(spatial_shape, np.nan, dtype=np.float64),
        }
        if return_weights:
            out["weights"] = np.full(
                spatial_shape + (int(out["t2_basis_ms"].shape[0]),),
                np.nan,
                dtype=np.float64,
            )

        for idx in np.flatnonzero(mask_flat):
            res = self.fit(flat[idx], **kwargs)
            out["mwf"].flat[idx] = float(res["mwf"])
            out["t2mw_ms"].flat[idx] = float(res["t2mw_ms"])
            out["t2iew_ms"].flat[idx] = float(res["t2iew_ms"])
            out["gmt2_ms"].flat[idx] = float(res["gmt2_ms"])
            out["mw_weight"].flat[idx] = float(res["mw_weight"])
            out["iew_weight"].flat[idx] = float(res["iew_weight"])
            out["total_weight"].flat[idx] = float(res["total_weight"])
            out["resid_l2"].flat[idx] = float(res["resid_l2"])
            if return_weights:
                out["weights"].reshape((-1, out["t2_basis_ms"].shape[0]))[idx, :] = np.asarray(
                    res["weights"], dtype=np.float64
                )

        return out
