from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .inversion_recovery import T1InversionRecovery
from .vfa_t1 import T1VFA

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
class T1DESPOT1HIFI:
    """Simplified DESPOT1-HIFI style T1/B1 estimation.

    Primary input is VFA data. Optional IR data can be provided as a weak prior
    to stabilize T1 when estimating B1 jointly.
    """

    flip_angle_deg: ArrayLike
    tr_ms: float
    b1_bounds: tuple[float, float] = (0.5, 1.5)
    b1_grid_size: int = 41

    def __post_init__(self) -> None:
        import numpy as np

        fa = _as_1d_float_array(self.flip_angle_deg, name="flip_angle_deg")
        if fa.size < 2:
            raise ValueError("flip_angle_deg must have at least 2 values")
        if np.any(fa <= 0):
            raise ValueError("flip_angle_deg must be > 0")
        if float(self.tr_ms) <= 0:
            raise ValueError("tr_ms must be > 0")
        b1_lo, b1_hi = float(self.b1_bounds[0]), float(self.b1_bounds[1])
        if b1_lo <= 0 or b1_hi <= b1_lo:
            raise ValueError("b1_bounds must satisfy 0 < lo < hi")
        if int(self.b1_grid_size) < 3:
            raise ValueError("b1_grid_size must be >= 3")
        object.__setattr__(self, "flip_angle_deg", fa)

    def forward(self, *, m0: float, t1_ms: float, b1: float = 1.0) -> NDArray[np.float64]:
        model = T1VFA(flip_angle_deg=self.flip_angle_deg, tr_ms=self.tr_ms, b1=b1)
        return model.forward(m0=m0, t1_ms=t1_ms)

    def fit(
        self,
        signal: ArrayLike,
        *,
        b1: float | None = None,
        estimate_b1: bool = True,
        b1_init: float | None = None,
        ir_signal: ArrayLike | None = None,
        ti_ms: ArrayLike | None = None,
        ir_weight: float = 1.0,
        b0_hz: float | None = None,
    ) -> dict[str, float]:
        import numpy as np
        from scipy.optimize import least_squares

        y = _as_1d_float_array(signal, name="signal")
        if y.shape != self.flip_angle_deg.shape:
            raise ValueError(
                f"signal shape {y.shape} must match flip_angle_deg shape {self.flip_angle_deg.shape}"
            )

        _ = b0_hz

        if b1 is not None and estimate_b1:
            raise ValueError("use either fixed b1 or estimate_b1=True, not both")
        if ir_weight <= 0:
            raise ValueError("ir_weight must be > 0")

        y_ir: NDArray[np.float64] | None = None
        ti_arr: NDArray[np.float64] | None = None
        ir_init: dict[str, float] | None = None
        if ir_signal is not None:
            if ti_ms is None:
                raise ValueError("ti_ms is required when ir_signal is provided")
            ti_arr = _as_1d_float_array(ti_ms, name="ti_ms")
            y_ir = _as_1d_float_array(ir_signal, name="ir_signal")
            if y_ir.shape != ti_arr.shape:
                raise ValueError("ir_signal shape must match ti_ms")
            ir_model = T1InversionRecovery(ti_ms=ti_arr)
            ir_init = ir_model.fit(y_ir, method="magnitude", solver="rdnls")

        if not estimate_b1:
            b1_eff = 1.0 if b1 is None else float(b1)
            if b1_eff <= 0:
                raise ValueError("b1 must be > 0")
            fit_vfa = T1VFA(
                flip_angle_deg=self.flip_angle_deg,
                tr_ms=self.tr_ms,
                b1=b1_eff,
            ).fit(y)

            if y_ir is None or ti_arr is None:
                pred = self.forward(m0=float(fit_vfa["m0"]), t1_ms=float(fit_vfa["t1_ms"]), b1=b1_eff)
                rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
                return {
                    "m0": float(fit_vfa["m0"]),
                    "t1_ms": float(fit_vfa["t1_ms"]),
                    "b1": b1_eff,
                    "res_rmse": rmse,
                    "n_points": float(y.size),
                }

            assert ir_init is not None

            ir_model = T1InversionRecovery(ti_ms=ti_arr)

            def residuals_fixed(params: NDArray[np.float64]) -> NDArray[np.float64]:
                m0_val = float(params[0])
                t1_val = float(params[1])
                ra_val = float(params[2])
                rb_val = float(params[3])
                vfa_res = self.forward(m0=m0_val, t1_ms=t1_val, b1=b1_eff) - y
                ir_res = ir_model.forward(t1_ms=t1_val, ra=ra_val, rb=rb_val, magnitude=True) - y_ir
                scale = float(np.sqrt(max(y.size, 1) / max(y_ir.size, 1)))
                return np.concatenate([vfa_res, ir_weight * scale * ir_res])

            x0 = np.array(
                [
                    float(fit_vfa["m0"]),
                    float(fit_vfa["t1_ms"]),
                    float(ir_init["ra"]),
                    float(ir_init["rb"]),
                ],
                dtype=np.float64,
            )
            lower = np.array([0.0, 1e-6, -np.inf, -np.inf], dtype=np.float64)
            upper = np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float64)
            result = least_squares(residuals_fixed, x0=x0, bounds=(lower, upper))
            m0_hat, t1_hat, _, _ = [float(v) for v in result.x]
            pred_hat = self.forward(m0=m0_hat, t1_ms=t1_hat, b1=b1_eff)
            rmse_hat = float(np.sqrt(np.mean((pred_hat - y) ** 2)))
            return {
                "m0": m0_hat,
                "t1_ms": t1_hat,
                "b1": b1_eff,
                "res_rmse": rmse_hat,
                "n_points": float(y.size),
            }

        b1_lo, b1_hi = float(self.b1_bounds[0]), float(self.b1_bounds[1])
        b1_grid = np.linspace(b1_lo, b1_hi, int(self.b1_grid_size), dtype=np.float64)
        if b1_init is not None and b1_lo <= float(b1_init) <= b1_hi:
            b1_grid = np.unique(np.concatenate([b1_grid, np.array([float(b1_init)], dtype=np.float64)]))

        best: dict[str, float] | None = None
        best_cost = np.inf

        for b1_try in b1_grid:
            fit_try = T1VFA(
                flip_angle_deg=self.flip_angle_deg,
                tr_ms=self.tr_ms,
                b1=float(b1_try),
            ).fit(y)
            pred_try = self.forward(
                m0=float(fit_try["m0"]),
                t1_ms=float(fit_try["t1_ms"]),
                b1=float(b1_try),
            )
            rmse = float(np.sqrt(np.mean((pred_try - y) ** 2)))
            cost = rmse
            if ir_init is not None:
                t1_prior = float(ir_init["t1_ms"])
                if np.isfinite(t1_prior) and t1_prior > 0:
                    cost += 0.2 * abs(np.log(float(fit_try["t1_ms"]) / t1_prior))
            if cost < best_cost:
                best_cost = cost
                best = {
                    "m0": float(fit_try["m0"]),
                    "t1_ms": float(fit_try["t1_ms"]),
                    "b1": float(b1_try),
                }

        if best is None:
            raise RuntimeError("T1DESPOT1HIFI grid search failed")

        if y_ir is None or ti_arr is None:
            def residuals(params: NDArray[np.float64]) -> NDArray[np.float64]:
                m0_val = float(params[0])
                t1_val = float(params[1])
                b1_val = float(params[2])
                pred = self.forward(m0=m0_val, t1_ms=t1_val, b1=b1_val)
                return pred - y

            x0 = np.array([best["m0"], best["t1_ms"], best["b1"]], dtype=np.float64)
            lower = np.array([0.0, 1e-6, b1_lo], dtype=np.float64)
            upper = np.array([np.inf, np.inf, b1_hi], dtype=np.float64)
            result = least_squares(residuals, x0=x0, bounds=(lower, upper))
            m0_hat, t1_hat, b1_hat = [float(v) for v in result.x]
            pred_hat = self.forward(m0=m0_hat, t1_ms=t1_hat, b1=b1_hat)
            rmse_hat = float(np.sqrt(np.mean((pred_hat - y) ** 2)))
            return {
                "m0": m0_hat,
                "t1_ms": t1_hat,
                "b1": b1_hat,
                "res_rmse": rmse_hat,
                "n_points": float(y.size),
            }

        assert ir_init is not None
        ir_model = T1InversionRecovery(ti_ms=ti_arr)

        def residuals_joint(params: NDArray[np.float64]) -> NDArray[np.float64]:
            m0_val = float(params[0])
            t1_val = float(params[1])
            b1_val = float(params[2])
            ra_val = float(params[3])
            rb_val = float(params[4])
            vfa_res = self.forward(m0=m0_val, t1_ms=t1_val, b1=b1_val) - y
            ir_res = ir_model.forward(t1_ms=t1_val, ra=ra_val, rb=rb_val, magnitude=True) - y_ir
            scale = float(np.sqrt(max(y.size, 1) / max(y_ir.size, 1)))
            return np.concatenate([vfa_res, ir_weight * scale * ir_res])

        x0 = np.array(
            [
                best["m0"],
                best["t1_ms"],
                best["b1"],
                float(ir_init["ra"]),
                float(ir_init["rb"]),
            ],
            dtype=np.float64,
        )
        lower = np.array([0.0, 1e-6, b1_lo, -np.inf, -np.inf], dtype=np.float64)
        upper = np.array([np.inf, np.inf, b1_hi, np.inf, np.inf], dtype=np.float64)
        result = least_squares(residuals_joint, x0=x0, bounds=(lower, upper))
        m0_hat, t1_hat, b1_hat, _, _ = [float(v) for v in result.x]

        pred_hat = self.forward(m0=m0_hat, t1_ms=t1_hat, b1=b1_hat)
        rmse_hat = float(np.sqrt(np.mean((pred_hat - y) ** 2)))
        return {
            "m0": m0_hat,
            "t1_ms": t1_hat,
            "b1": b1_hat,
            "res_rmse": rmse_hat,
            "n_points": float(y.size),
        }

    def fit_image(
        self,
        signal: ArrayLike,
        *,
        mask: ArrayLike | str | None = None,
        ir_signal: ArrayLike | None = None,
        ti_ms: ArrayLike | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        import logging

        import numpy as np

        from qmrpy._mask import resolve_mask
        from qmrpy._parallel import parallel_fit

        logger = logging.getLogger("qmrpy")

        arr = np.asarray(signal, dtype=np.float64)
        if arr.ndim == 1:
            if mask is not None:
                raise ValueError("mask must be None for 1D data")
            return self.fit(arr, ir_signal=ir_signal, ti_ms=ti_ms, **kwargs)

        if arr.shape[-1] != self.flip_angle_deg.shape[0]:
            raise ValueError(
                f"data last dim {arr.shape[-1]} must match flip_angle_deg length {self.flip_angle_deg.shape[0]}"
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

        ir_flat = None
        if ir_signal is not None:
            if ti_ms is None:
                raise ValueError("ti_ms is required when ir_signal is provided")
            ir_arr = np.asarray(ir_signal, dtype=np.float64)
            if ir_arr.shape[:-1] != spatial_shape:
                raise ValueError("ir_signal spatial shape must match VFA signal")
            ir_flat = ir_arr.reshape((-1, ir_arr.shape[-1]))

        output_keys = ["m0", "t1_ms", "b1", "res_rmse", "n_points"]

        if ir_flat is None:
            def fit_func(signal_1d: NDArray[Any]) -> dict[str, float]:
                return self.fit(signal_1d, **kwargs)

            return parallel_fit(
                fit_func,
                flat,
                mask_flat,
                output_keys,
                spatial_shape,
                n_jobs=n_jobs,
                verbose=verbose,
                desc="T1DESPOT1HIFI",
            )

        from joblib import Parallel, delayed

        out: dict[str, Any] = {
            key: np.full(spatial_shape, np.nan, dtype=np.float64)
            for key in output_keys
        }

        indices = np.flatnonzero(mask_flat)
        if verbose:
            logger.info("T1DESPOT1HIFI: %d voxels, n_jobs=%s", len(indices), n_jobs)

        def _fit_idx(idx: int) -> tuple[int, dict[str, float]]:
            return idx, self.fit(flat[idx], ir_signal=ir_flat[idx], ti_ms=ti_ms, **kwargs)

        if n_jobs == 1:
            iterator = indices
            if verbose:
                from tqdm import tqdm

                iterator = tqdm(indices, desc="T1DESPOT1HIFI", unit="voxel")
            for idx in iterator:
                res = self.fit(flat[idx], ir_signal=ir_flat[idx], ti_ms=ti_ms, **kwargs)
                for key in output_keys:
                    if key in res:
                        out[key].flat[idx] = float(res[key])
            return out

        if verbose:
            from tqdm import tqdm

            results = Parallel(n_jobs=n_jobs)(
                delayed(_fit_idx)(idx) for idx in tqdm(indices, desc="T1DESPOT1HIFI", unit="voxel")
            )
        else:
            results = Parallel(n_jobs=n_jobs)(delayed(_fit_idx)(idx) for idx in indices)

        for idx, res in results:
            for key in output_keys:
                if key in res:
                    out[key].flat[idx] = float(res[key])
        return out
