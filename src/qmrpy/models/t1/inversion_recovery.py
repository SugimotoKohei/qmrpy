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
class InversionRecovery:
    """Inversion Recovery T1 model (qMRLab: inversion_recovery, Barral).

    Signal model (Barral):
        S(TI) = ra + rb * exp(-TI / T1)

    Units (aligned to qMRLab protocol):
        - ti_ms: milliseconds
        - t1_ms: milliseconds

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
        t1_init_ms: float | None = None,
        ra_init: float | None = None,
        rb_init: float | None = None,
        bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
        max_nfev: int | None = None,
    ) -> dict[str, float]:
        """Fit Barral model parameters.

        Returns dict with keys:
            - t1_ms, ra, rb
            - res_rmse
            - idx (only for method="magnitude")
        """
        import numpy as np
        from scipy.optimize import least_squares

        y = _as_1d_float_array(signal, name="signal")
        if y.shape != self.ti_ms.shape:
            raise ValueError(f"signal shape {y.shape} must match ti_ms shape {self.ti_ms.shape}")

        method_norm = method.lower().strip()
        if method_norm not in {"magnitude", "complex"}:
            raise ValueError("method must be 'magnitude' or 'complex'")

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
            "idx": float(best_idx),
            "res_rmse": float(best_rmse),
        }

    def fit_image(
        self,
        data: ArrayLike,
        *,
        mask: ArrayLike | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Voxel-wise fit on an image/volume.

        Expects `data` shape (..., n_ti) where n_ti == len(self.ti_ms).
        """
        import numpy as np

        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim == 1:
            return self.fit(arr, **kwargs)
        if arr.shape[-1] != self.ti_ms.shape[0]:
            raise ValueError(
                f"data last dim {arr.shape[-1]} must match ti_ms length {self.ti_ms.shape[0]}"
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

        method_norm = str(kwargs.get("method", "magnitude")).lower().strip()
        out: dict[str, Any] = {
            "t1_ms": np.full(spatial_shape, np.nan, dtype=np.float64),
            "ra": np.full(spatial_shape, np.nan, dtype=np.float64),
            "rb": np.full(spatial_shape, np.nan, dtype=np.float64),
            "res_rmse": np.full(spatial_shape, np.nan, dtype=np.float64),
        }
        if method_norm == "magnitude":
            out["idx"] = np.full(spatial_shape, np.nan, dtype=np.float64)

        for idx in np.flatnonzero(mask_flat):
            res = self.fit(flat[idx], **kwargs)
            out["t1_ms"].flat[idx] = float(res["t1_ms"])
            out["ra"].flat[idx] = float(res["ra"])
            out["rb"].flat[idx] = float(res["rb"])
            out["res_rmse"].flat[idx] = float(res["res_rmse"])
            if method_norm == "magnitude" and "idx" in res:
                out["idx"].flat[idx] = float(res["idx"])

        return out
