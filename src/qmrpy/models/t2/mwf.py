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
                     If None provided in __init__, defaults to a qMRLab-like log-spaced basis
                     from 1.5*TE1 to 400ms (120 points).

    Notes
    -----
    This class is a minimal v0 translation aligned with qMRLab's `mwf` intent:
    we estimate a non-negative T2 spectrum via NNLS and derive summary metrics
    (MWF, T2MW, T2IEW) by integrating the spectrum over cutoff ranges.

    References
    ----------
    .. [1] MacKay A, et al. (1994). In vivo visualization of myelin water in
           brain by magnetic resonance. Magn Reson Med, 31(6):673-677.
    .. [2] Whittall KP, MacKay AL (1989). Quantitative interpretation of NMR
           relaxation data. J Magn Reson, 84(1):134-152.
    .. [3] Laule C, et al. (2007). Myelin water imaging in multiple sclerosis:
           quantitative correlations with histopathology. Mult Scler,
           13(3):279-287.
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
            if te.size > 0 and float(te[0]) > 0:
                t2_min_ms = 1.5 * float(te[0])
            else:
                t2_min_ms = 10.0
            basis = self.default_t2_basis_ms(t2_min_ms=t2_min_ms, t2_max_ms=400.0, n=120)
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

        Parameters
        ----------
        weights : array-like
            Weights of size ``(n_basis,)`` corresponding to ``t2_basis_ms``.

        Returns
        -------
        ndarray
            Simulated signal array.
        """

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
        regularization_mode: str | None = None,
        qmrlab_sigma: float = 20.0,
        qmrlab_mu: float = 0.25,
        qmrlab_chi2_min: float = 2.0,
        qmrlab_chi2_max: float = 2.5,
        qmrlab_max_iter: int = 50,
        lower_cutoff_mw_ms: float | None = None,
        cutoff_ms: float = 40.0,
        upper_cutoff_iew_ms: float = 200.0,
        use_weighted_geometric_mean: bool = True,
    ) -> dict[str, Any]:
        """Fit T2 distribution using NNLS and compute MWF/T2MW/T2IEW from cutoffs.

        Parameters
        ----------
        signal : array-like
            Observed signal array ``(n_te,)``.
        regularization_alpha : float, optional
            Tikhonov regularization parameter. If > 0, solves
            ``argmin ||Aw - y||^2 + alpha^2 ||w||^2``.
        regularization_mode : {"none", "tikhonov", "qmrlab_regnnls"}, optional
            Select the regularization strategy. If None, uses "tikhonov" when
            ``regularization_alpha > 0`` otherwise "none".
        qmrlab_sigma : float, optional
            Sigma used by qMRLab-style regNNLS (default: 20.0).
        qmrlab_mu : float, optional
            Initial mu for qMRLab-style regNNLS (default: 0.25).
        qmrlab_chi2_min : float, optional
            Minimum chi2 diff percentage for qMRLab regNNLS (default: 2.0).
        qmrlab_chi2_max : float, optional
            Maximum chi2 diff percentage for qMRLab regNNLS (default: 2.5).
        qmrlab_max_iter : int, optional
            Maximum iterations for qMRLab regNNLS (default: 50).
        lower_cutoff_mw_ms : float, optional
            Lower cutoff for MW integration. If None, uses
            ``1.5 * first echo`` (qMRLab default).
        cutoff_ms : float, optional
            MW/IEW cutoff time in milliseconds (default: 40 ms).
        upper_cutoff_iew_ms : float, optional
            Upper cutoff for IEW integration in milliseconds (default: 200 ms).
        use_weighted_geometric_mean : bool, optional
            If True, compute T2MW/T2IEW as weighted geometric means (qMRLab default).

        Returns
        -------
        dict
            Keys include ``weights``, ``t2_basis_ms``, ``mwf``, ``t2mw_ms``,
            ``t2iew_ms``, ``gmt2_ms``, and ``resid_l2``.
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

        if regularization_mode is None:
            regularization_mode = "tikhonov" if regularization_alpha > 0 else "none"
        regularization_mode = str(regularization_mode).lower()
        if regularization_mode not in {"none", "tikhonov", "qmrlab_regnnls"}:
            raise ValueError(f"regularization_mode must be one of none/tikhonov/qmrlab_regnnls, got {regularization_mode}")

        def _do_nnls(matrix: Any, data: Any) -> NDArray[np.float64]:
            w, _ = nnls(matrix, data)
            return w

        def _chi2(matrix: Any, data: Any, weights: Any, sigma: float) -> float:
            if sigma <= 0:
                return float("inf")
            resid = matrix @ weights - data
            return float(np.sum(resid * resid) / (sigma**2))

        reg_info: dict[str, float] = {}
        if regularization_mode == "tikhonov" and regularization_alpha > 0:
            # Tikhonov regularization: augment A and y
            # [     A      ] w = [ y ]
            # [ alpha * I  ]     [ 0 ]
            n_basis = len(self.t2_basis_ms)
            A_aug = np.vstack([A, regularization_alpha * np.eye(n_basis)])
            y_aug = np.concatenate([y, np.zeros(n_basis)])
            w_hat = _do_nnls(A_aug, y_aug)
        elif regularization_mode == "qmrlab_regnnls":
            sigma = float(qmrlab_sigma)
            mu = float(qmrlab_mu)
            chi2_min = float(qmrlab_chi2_min)
            chi2_max = float(qmrlab_chi2_max)
            max_iter = int(qmrlab_max_iter)

            w_nnls = _do_nnls(A, y)
            chi2_nnls = _chi2(A, y, w_nnls, sigma)

            n_basis = len(self.t2_basis_ms)
            A_aug = np.vstack([A, mu * np.eye(n_basis)])
            y_aug = np.concatenate([y, np.zeros(n_basis)])
            w_reg = _do_nnls(A_aug, y_aug)
            chi2_reg = _chi2(A_aug, y_aug, w_reg, sigma)

            if np.isfinite(chi2_nnls):
                diff = 100.0 * (chi2_reg - chi2_nnls) / chi2_nnls
            else:
                diff = float("nan")

            n_iter = 0
            while n_iter < max_iter and np.isfinite(diff) and (diff > chi2_max or diff < chi2_min):
                if diff > chi2_max:
                    mu = mu / 2.0
                else:
                    mu = 1.9 * mu
                A_aug = np.vstack([A, mu * np.eye(n_basis)])
                y_aug = np.concatenate([y, np.zeros(n_basis)])
                w_reg = _do_nnls(A_aug, y_aug)
                chi2_reg = _chi2(A_aug, y_aug, w_reg, sigma)
                diff = 100.0 * (chi2_reg - chi2_nnls) / chi2_nnls if np.isfinite(chi2_nnls) else float("nan")
                n_iter += 1

            w_hat = w_reg
            reg_info = {
                "reg_nnls_mu": float(mu),
                "chi2_nnls": float(chi2_nnls),
                "chi2_reg": float(chi2_reg),
                "chi2_diff_percent": float(diff),
            }
        else:
            w_hat = _do_nnls(A, y)

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

        out = {
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
        out.update(reg_info)
        return out

    def fit_image(
        self,
        signal: ArrayLike,
        *,
        mask: ArrayLike | str | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
        return_weights: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Voxel-wise NNLS fit on an image/volume.

        Parameters
        ----------
        signal : array-like
            Input array with last dim as echoes.
        mask : array-like, optional
            Spatial mask. If "otsu", Otsu thresholding is applied.
        n_jobs : int, default=1
            Number of parallel jobs. -1 uses all CPUs.
        verbose : bool, default=False
            If True, show progress bar and log info.
        return_weights : bool, optional
            If True, include voxel-wise weights.
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

        logger = logging.getLogger("qmrpy")

        arr = np.asarray(signal, dtype=np.float64)
        if arr.ndim == 1:
            return self.fit(arr, **kwargs)
        if arr.shape[-1] != self.te_ms.shape[0]:
            raise ValueError(
                f"data last dim {arr.shape[-1]} must match te_ms length {self.te_ms.shape[0]}"
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

        indices = np.flatnonzero(mask_flat)
        n_voxels = len(indices)

        if verbose:
            logger.info("MWF: %d voxels, n_jobs=%s, shape=%s", n_voxels, n_jobs, spatial_shape)

        # Parallel execution for MWF
        if n_jobs == 1:
            iterator = indices
            if verbose:
                from tqdm import tqdm
                iterator = tqdm(indices, desc="MWF", unit="voxel")

            for idx in iterator:
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
        else:
            from joblib import Parallel, delayed

            def fit_single(idx: int) -> tuple[int, dict[str, Any]]:
                return idx, self.fit(flat[idx], **kwargs)

            if verbose:
                from tqdm import tqdm
                results = Parallel(n_jobs=n_jobs)(
                    delayed(fit_single)(idx)
                    for idx in tqdm(indices, desc="MWF", unit="voxel")
                )
            else:
                results = Parallel(n_jobs=n_jobs)(
                    delayed(fit_single)(idx) for idx in indices
                )

            for idx, res in results:
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

        if verbose:
            logger.info("MWF complete: %d voxels processed", n_voxels)

        return out
