from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import ArrayLike, NDArray
else:
    ArrayLike = Any  # type: ignore[misc,assignment]
    NDArray = Any  # type: ignore[misc,assignment]


def _as_1d_array(values: ArrayLike, *, name: str, dtype: Any | None = None) -> NDArray[Any]:
    import numpy as np

    array = np.asarray(values, dtype=dtype) if dtype is not None else np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={array.shape}")
    return array


def _as_1d_float_array(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    import numpy as np

    return _as_1d_array(values, name=name, dtype=np.float64)


def _zoom_t1_vec(t1_vec: NDArray[np.float64], ind: int, t1_len_z: int) -> NDArray[np.float64]:
    import numpy as np

    if ind > 0 and ind < t1_vec.size - 1:
        return np.linspace(t1_vec[ind - 1], t1_vec[ind + 1], t1_len_z, dtype=np.float64)
    if ind == 0:
        hi = min(t1_vec.size - 1, ind + 2)
        return np.linspace(t1_vec[ind], t1_vec[hi], t1_len_z, dtype=np.float64)
    lo = max(0, ind - 2)
    return np.linspace(t1_vec[lo], t1_vec[ind], t1_len_z, dtype=np.float64)


def _build_nls_struct(
    ti_ms: NDArray[np.float64],
    *,
    t1_grid_ms: ArrayLike | None = None,
    nls_zoom: int | None = None,
) -> dict[str, Any]:
    import numpy as np

    t_vec = _as_1d_float_array(ti_ms, name="ti_ms")
    if t1_grid_ms is None:
        t1_vec = np.arange(1.0, 5000.0 + 1.0, dtype=np.float64)
    else:
        t1_vec = _as_1d_float_array(t1_grid_ms, name="t1_grid_ms")
    if np.any(t1_vec <= 0):
        raise ValueError("t1_grid_ms must be positive")

    zoom = 2 if nls_zoom is None else int(nls_zoom)
    if zoom < 1:
        raise ValueError("nls_zoom must be >= 1")

    nlsS: dict[str, Any] = {
        "tVec": t_vec,
        "N": int(t_vec.size),
        "T1Vec": t1_vec,
        "T1Start": float(t1_vec[0]),
        "T1Stop": float(t1_vec[-1]),
        "T1Len": int(t1_vec.size),
        "nlsAlg": "grid",
        "nbrOfZoom": zoom,
    }
    if zoom > 1:
        nlsS["T1LenZ"] = 21

    alpha_vec = 1.0 / t1_vec
    the_exp = np.exp(-t_vec[:, None] * alpha_vec[None, :])
    rho_norm_vec = np.sum(the_exp**2, axis=0) - (1.0 / nlsS["N"]) * (np.sum(the_exp, axis=0) ** 2)
    nlsS["theExp"] = the_exp
    nlsS["rhoNormVec"] = rho_norm_vec
    return nlsS


def _rdnls_grid(data: NDArray[Any], nlsS: dict[str, Any]) -> tuple[float, float, float, float]:
    import numpy as np

    if data.size != nlsS["N"]:
        raise ValueError("nlsS.N and data must be of equal length")

    y = np.asarray(data).reshape(-1)
    y_sum = np.sum(y)
    n = nlsS["N"]
    t_vec = nlsS["tVec"]
    t1_vec = nlsS["T1Vec"]
    the_exp = nlsS["theExp"]
    rho_norm_vec = nlsS["rhoNormVec"]

    rho_ty_vec = (y.conj() @ the_exp) - (1.0 / n) * np.sum(the_exp, axis=0) * y_sum
    crit = (np.abs(rho_ty_vec) ** 2) / rho_norm_vec
    ind = int(np.argmax(crit))

    if nlsS["nbrOfZoom"] > 1:
        t1_len_z = int(nlsS.get("T1LenZ", 21))
        for _ in range(2, int(nlsS["nbrOfZoom"]) + 1):
            t1_vec = _zoom_t1_vec(t1_vec, ind, t1_len_z)
            alpha_vec = 1.0 / t1_vec
            the_exp = np.exp(-t_vec[:, None] * alpha_vec[None, :])
            y_exp_sum = y.conj() @ the_exp
            rho_norm_vec = np.sum(the_exp**2, axis=0) - (1.0 / n) * (np.sum(the_exp, axis=0) ** 2)
            rho_ty_vec = y_exp_sum - (1.0 / n) * np.sum(the_exp, axis=0) * y_sum
            crit = (np.abs(rho_ty_vec) ** 2) / rho_norm_vec
            ind = int(np.argmax(crit))

    t1_hat = float(t1_vec[ind])
    b_hat = rho_ty_vec[ind] / rho_norm_vec[ind]
    a_hat = (1.0 / n) * (y_sum - b_hat * np.sum(the_exp[:, ind]))
    with np.errstate(divide="ignore", invalid="ignore"):
        model_value = a_hat + b_hat * np.exp(-t_vec / t1_hat)
        residual = (1.0 / np.sqrt(n)) * np.linalg.norm(1.0 - model_value / y)

    return float(t1_hat), float(np.real(b_hat)), float(np.real(a_hat)), float(residual)


def _rdnls_pr_grid(
    data: NDArray[np.float64], nlsS: dict[str, Any]
) -> tuple[float, float, float, float, int]:
    import numpy as np

    if data.size != nlsS["N"]:
        raise ValueError("nlsS.N and data must be of equal length")

    t_vec = nlsS["tVec"]
    order = np.argsort(t_vec)
    t_vec_sorted = t_vec[order]
    data_sorted = np.asarray(data, dtype=np.float64).reshape(-1)[order]
    n = nlsS["N"]

    min_ind = int(np.argmin(data_sorted))

    t1_est_tmp = np.zeros((2,), dtype=np.float64)
    a_est_tmp = np.zeros((2,), dtype=np.float64)
    b_est_tmp = np.zeros((2,), dtype=np.float64)
    res_tmp = np.zeros((2,), dtype=np.float64)

    for ii in range(2):
        the_exp = nlsS["theExp"][order, :]
        t1_vec = nlsS["T1Vec"]

        signs = np.ones((n,), dtype=np.float64)
        if ii == 0:
            signs[: min_ind + 1] = -1.0
        else:
            if min_ind > 0:
                signs[:min_ind] = -1.0

        data_tmp = data_sorted * signs
        y_sum = np.sum(data_tmp)
        rho_norm_vec = nlsS["rhoNormVec"]
        rho_ty_vec = (data_tmp @ the_exp) - (1.0 / n) * np.sum(the_exp, axis=0) * y_sum
        crit = (np.abs(rho_ty_vec) ** 2) / rho_norm_vec
        ind = int(np.argmax(crit))

        if nlsS["nbrOfZoom"] > 1:
            t1_len_z = int(nlsS.get("T1LenZ", 21))
            for _ in range(2, int(nlsS["nbrOfZoom"]) + 1):
                t1_vec = _zoom_t1_vec(t1_vec, ind, t1_len_z)
                alpha_vec = 1.0 / t1_vec
                the_exp = np.exp(-t_vec_sorted[:, None] * alpha_vec[None, :])
                y_exp_sum = data_tmp @ the_exp
                rho_norm_vec = np.sum(the_exp**2, axis=0) - (1.0 / n) * (np.sum(the_exp, axis=0) ** 2)
                rho_ty_vec = y_exp_sum - (1.0 / n) * np.sum(the_exp, axis=0) * y_sum
                crit = (np.abs(rho_ty_vec) ** 2) / rho_norm_vec
                ind = int(np.argmax(crit))

        t1_est_tmp[ii] = float(t1_vec[ind])
        b_est_tmp[ii] = float(rho_ty_vec[ind] / rho_norm_vec[ind])
        a_est_tmp[ii] = float((1.0 / n) * (y_sum - b_est_tmp[ii] * np.sum(the_exp[:, ind])))
        with np.errstate(divide="ignore", invalid="ignore"):
            model_value = a_est_tmp[ii] + b_est_tmp[ii] * np.exp(-t_vec_sorted / t1_est_tmp[ii])
            res_tmp[ii] = float((1.0 / np.sqrt(n)) * np.linalg.norm(1.0 - model_value / data_tmp))

    best = int(np.argmin(res_tmp))
    t1_est = float(t1_est_tmp[best])
    b_est = float(b_est_tmp[best])
    a_est = float(a_est_tmp[best])
    res = float(res_tmp[best])
    idx = int(min_ind + 1 if best == 0 else min_ind)
    return t1_est, b_est, a_est, res, idx
@dataclass(frozen=True, slots=True)
class InversionRecovery:
    """Inversion Recovery T1 model (qMRLab: inversion_recovery, Barral).

    Signal model (Barral):
        S(TI) = ra + rb * exp(-TI / T1)

    Units
    -----
    ti_ms : milliseconds
    t1_ms : milliseconds

    method:
        - "complex": fit raw model
        - "magnitude": assume |S| observed; perform polarity restoration by searching idx
    """

    ti_ms: ArrayLike

    def __post_init__(self) -> None:
        import numpy as np

        ti = _as_1d_float_array(self.ti_ms, name="ti_ms")
        if np.any(ti < 0):
            raise ValueError("ti_ms must be non-negative")
        if np.any(np.diff(ti) < 0):
            # qMRLab sorts in UpdateFields; enforce sorted input to avoid silent surprises
            raise ValueError("ti_ms must be sorted ascending")
        object.__setattr__(self, "ti_ms", ti)

    def forward(
        self, *, t1_ms: float, ra: float, rb: float, magnitude: bool = False
    ) -> NDArray[np.float64]:
        """Simulate inversion recovery signal.

        Parameters
        ----------
        t1_ms : float
            T1 in milliseconds.
        ra : float
            Offset term.
        rb : float
            Amplitude term.
        magnitude : bool, optional
            If True, return magnitude signal.

        Returns
        -------
        ndarray
            Simulated signal array.
        """
        import numpy as np

        if t1_ms <= 0:
            raise ValueError("t1_ms must be > 0")
        s = ra + rb * np.exp(-self.ti_ms / float(t1_ms))
        return np.abs(s) if magnitude else s

    def fit(
        self,
        signal: ArrayLike,
        *,
        method: str = "magnitude",
        solver: str = "least_squares",
        t1_init_ms: float | None = None,
        ra_init: float | None = None,
        rb_init: float | None = None,
        bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
        max_nfev: int | None = None,
        t1_grid_ms: ArrayLike | None = None,
        nls_zoom: int | None = None,
    ) -> dict[str, float]:
        """Fit Barral model parameters.

        Parameters
        ----------
        signal : array-like
            Observed signal array.
        method : {"magnitude", "complex"}, optional
            Fitting mode.
        solver : {"least_squares", "rdnls"}, optional
            Optimization backend. ``rdnls`` replicates qMRLab's grid-search
            reduced-dimension NLS (rdNls/rdNlsPr).
        t1_init_ms : float, optional
            Initial guess for T1 in milliseconds.
        ra_init : float, optional
            Initial guess for ra.
        rb_init : float, optional
            Initial guess for rb.
        bounds : tuple of tuple, optional
            Bounds for parameters as ``((t1, rb, ra) min, (t1, rb, ra) max)``.
        max_nfev : int, optional
            Max number of function evaluations.
        t1_grid_ms : array-like, optional
            Explicit T1 grid for ``solver="rdnls"`` (ms). Defaults to 1..5000.
        nls_zoom : int, optional
            Number of zoom iterations for ``solver="rdnls"``. Defaults to 2.

        Returns
        -------
        dict
            Fit results with keys ``t1_ms``, ``ra``, ``rb``, ``res_rmse``,
            and ``idx`` (only for ``method="magnitude"``).
            For ``solver="rdnls"``, ``res_rmse`` follows qMRLab's relative
            residual definition.
        """
        import numpy as np

        method_norm = method.lower().strip()
        if method_norm not in {"magnitude", "complex"}:
            raise ValueError("method must be 'magnitude' or 'complex'")

        solver_norm = solver.lower().strip()
        if solver_norm not in {"least_squares", "rdnls"}:
            raise ValueError("solver must be 'least_squares' or 'rdnls'")

        if solver_norm == "rdnls":
            y = _as_1d_array(
                signal,
                name="signal",
                dtype=None if method_norm == "complex" else np.float64,
            )
        else:
            y = _as_1d_float_array(signal, name="signal")

        if y.shape != self.ti_ms.shape:
            raise ValueError(f"signal shape {y.shape} must match ti_ms shape {self.ti_ms.shape}")

        if solver_norm == "rdnls":
            nlsS = _build_nls_struct(self.ti_ms, t1_grid_ms=t1_grid_ms, nls_zoom=nls_zoom)
            if method_norm == "complex":
                t1_hat, rb_hat, ra_hat, res = _rdnls_grid(y, nlsS)
                return {
                    "t1_ms": float(t1_hat),
                    "rb": float(rb_hat),
                    "ra": float(ra_hat),
                    "res_rmse": float(res),
                }
            y_mag = np.abs(y)
            t1_hat, rb_hat, ra_hat, res, idx = _rdnls_pr_grid(y_mag, nlsS)
            return {
                "t1_ms": float(t1_hat),
                "rb": float(rb_hat),
                "ra": float(ra_hat),
                "idx": int(idx),
                "res_rmse": float(res),
            }

        from scipy.optimize import least_squares

        if bounds is None:
            # qMRLab defaults: [T1, rb, ra]
            lower = (1e-4, -10000.0, 1e-4)
            upper = (5000.0, 0.0, 10000.0)
        else:
            lower, upper = bounds

        if t1_init_ms is None:
            t1_init_ms = 600.0
        if rb_init is None:
            rb_init = -1000.0
        if ra_init is None:
            ra_init = 500.0

        def residuals(params: Any, *, y_target: Any) -> Any:
            t1_ms, rb, ra = float(params[0]), float(params[1]), float(params[2])
            pred = self.forward(t1_ms=t1_ms, ra=ra, rb=rb, magnitude=False)
            return pred - y_target

        def solve_for(y_target: Any) -> tuple[np.ndarray, float]:
            x0 = np.array([t1_init_ms, rb_init, ra_init], dtype=np.float64)
            result = least_squares(
                lambda p: residuals(p, y_target=y_target),
                x0=x0,
                bounds=(np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)),
                max_nfev=max_nfev,
            )
            r = result.fun
            rmse = float(np.sqrt(np.mean(np.asarray(r, dtype=np.float64) ** 2)))
            return result.x, rmse

        if method_norm == "complex":
            x_hat, rmse = solve_for(y)
            return {
                "t1_ms": float(x_hat[0]),
                "rb": float(x_hat[1]),
                "ra": float(x_hat[2]),
                "res_rmse": float(rmse),
            }

        # magnitude: polarity restoration (choose idx that minimizes residual)
        y_mag = np.abs(y)
        best_idx = 0
        best_x = None
        best_rmse = float("inf")
        for idx in range(0, y_mag.size + 1):
            y_rest = y_mag.copy()
            if idx > 0:
                y_rest[:idx] *= -1.0
            x_hat, rmse = solve_for(y_rest)
            if rmse < best_rmse:
                best_rmse = rmse
                best_x = x_hat
                best_idx = idx

        assert best_x is not None
        return {
            "t1_ms": float(best_x[0]),
            "rb": float(best_x[1]),
            "ra": float(best_x[2]),
            "idx": int(best_idx),
            "res_rmse": float(best_rmse),
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
        """Voxel-wise fit on an image/volume.

        Parameters
        ----------
        signal : array-like
            Input array with last dim as inversion times.
        mask : array-like, optional
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
        if arr.shape[-1] != self.ti_ms.shape[0]:
            raise ValueError(
                f"data last dim {arr.shape[-1]} must match ti_ms length {self.ti_ms.shape[0]}"
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

        method_norm = str(kwargs.get("method", "magnitude")).lower().strip()
        output_keys = ["t1_ms", "ra", "rb", "res_rmse"]
        if method_norm == "magnitude":
            output_keys.append("idx")

        def fit_func(signal: NDArray[Any]) -> dict[str, float]:
            return self.fit(signal, **kwargs)

        result = parallel_fit(
            fit_func, flat, mask_flat, output_keys, spatial_shape,
            n_jobs=n_jobs, verbose=verbose, desc="InversionRecovery"
        )

        # Convert idx to int64
        if "idx" in result:
            idx_arr = result["idx"]
            idx_arr[np.isnan(idx_arr)] = -1
            result["idx"] = idx_arr.astype(np.int64)

        return result
