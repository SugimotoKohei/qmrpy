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


def _mp2rage_uni(inv1: float, inv2: float) -> float:
    den = float(inv1 * inv1 + inv2 * inv2 + 1e-12)
    return float((inv1 * inv2) / den)


def _parse_positive_bounds(
    bounds: tuple[float, float], *, name: str
) -> tuple[float, float]:
    lo = float(bounds[0])
    hi = float(bounds[1])
    if lo <= 0 or hi <= lo:
        raise ValueError(f"{name} must satisfy 0 < lo < hi")
    return lo, hi


def _rmse(pred: NDArray[np.float64], obs: NDArray[np.float64]) -> float:
    import numpy as np

    return float(np.sqrt(np.mean((pred - obs) ** 2)))


@dataclass(frozen=True, slots=True)
class _GridCandidate:
    m0: float
    t1_ms: float
    b1: float
    score: float


@dataclass(frozen=True, slots=True)
class T1MP2RAGE:
    """Simplified T1MP2RAGE T1 estimator.

    This implementation expects two inversion images [INV1, INV2] and fits
    (m0, t1, b1) using a compact surrogate signal model.
    """

    ti1_ms: float
    ti2_ms: float
    alpha1_deg: float
    alpha2_deg: float
    b1: float = 1.0

    def __post_init__(self) -> None:
        if float(self.ti1_ms) <= 0 or float(self.ti2_ms) <= 0:
            raise ValueError("ti1_ms and ti2_ms must be > 0")
        if float(self.ti2_ms) <= float(self.ti1_ms):
            raise ValueError("ti2_ms must be greater than ti1_ms")
        if float(self.alpha1_deg) <= 0 or float(self.alpha2_deg) <= 0:
            raise ValueError("alpha1_deg and alpha2_deg must be > 0")
        if float(self.b1) <= 0:
            raise ValueError("b1 must be > 0")

    def forward(
        self,
        *,
        m0: float,
        t1_ms: float,
        b1: float | None = None,
        b0_hz: float | None = None,
    ) -> NDArray[np.float64]:
        import numpy as np

        if t1_ms <= 0:
            raise ValueError("t1_ms must be > 0")
        b1_eff = float(self.b1 if b1 is None else b1)
        if b1_eff <= 0:
            raise ValueError("b1 must be > 0")

        _ = b0_hz

        e1 = np.exp(-float(self.ti1_ms) / float(t1_ms))
        e2 = np.exp(-float(self.ti2_ms) / float(t1_ms))
        a1 = np.deg2rad(float(self.alpha1_deg) * b1_eff)
        a2 = np.deg2rad(float(self.alpha2_deg) * b1_eff)

        inv1 = float(m0) * (1.0 - 2.0 * e1) * np.sin(a1)
        inv2 = float(m0) * (1.0 - 2.0 * e2) * np.sin(a2)
        return np.asarray([inv1, inv2], dtype=np.float64)

    def forward_uni(
        self,
        *,
        m0: float,
        t1_ms: float,
        b1: float | None = None,
        b0_hz: float | None = None,
    ) -> float:
        inv1, inv2 = self.forward(m0=m0, t1_ms=t1_ms, b1=b1, b0_hz=b0_hz)
        return _mp2rage_uni(float(inv1), float(inv2))

    @staticmethod
    def _prepare_t1_grid(
        *,
        t1_bounds_ms: tuple[float, float],
        t1_grid_ms: ArrayLike | None,
    ) -> tuple[float, float, NDArray[np.float64]]:
        import numpy as np

        t1_lo, t1_hi = _parse_positive_bounds(t1_bounds_ms, name="t1_bounds_ms")
        if t1_grid_ms is None:
            t1_grid = np.linspace(t1_lo, t1_hi, 200, dtype=np.float64)
        else:
            t1_grid = _as_1d_float_array(t1_grid_ms, name="t1_grid_ms")
        if t1_grid.size == 0:
            raise ValueError("t1_grid_ms must not be empty")
        if np.any(~np.isfinite(t1_grid)):
            raise ValueError("t1_grid_ms must contain only finite values")
        if np.any(t1_grid <= 0):
            raise ValueError("t1_grid_ms must be > 0")
        return t1_lo, t1_hi, t1_grid

    def _prepare_b1_grid(
        self,
        *,
        estimate_b1: bool,
        b1: float | None,
        b1_bounds: tuple[float, float],
    ) -> tuple[float, float, NDArray[np.float64]]:
        import numpy as np

        b1_lo, b1_hi = _parse_positive_bounds(b1_bounds, name="b1_bounds")
        if estimate_b1:
            return b1_lo, b1_hi, np.linspace(b1_lo, b1_hi, 61, dtype=np.float64)

        b1_fixed = float(self.b1 if b1 is None else b1)
        if b1_fixed <= 0:
            raise ValueError("b1 must be > 0")
        return b1_lo, b1_hi, np.array([b1_fixed], dtype=np.float64)

    def _grid_search_candidate(
        self,
        *,
        y: NDArray[np.float64],
        use_uni: bool,
        t1_grid: NDArray[np.float64],
        b1_grid: NDArray[np.float64],
    ) -> _GridCandidate:
        import numpy as np

        y_uni = float(y[0]) if use_uni else _mp2rage_uni(float(y[0]), float(y[1]))
        best = _GridCandidate(
            m0=float(np.max(np.abs(y))) if not use_uni else 1.0,
            t1_ms=float(t1_grid[0]),
            b1=float(b1_grid[0]),
            score=np.inf,
        )

        for b1_try in b1_grid:
            for t1_try in t1_grid:
                basis = self.forward(m0=1.0, t1_ms=float(t1_try), b1=float(b1_try))
                uni_pred = _mp2rage_uni(float(basis[0]), float(basis[1]))
                score = float(abs(uni_pred - y_uni))
                if score >= best.score:
                    continue

                if use_uni:
                    m0_try = 1.0
                else:
                    den = float(np.dot(basis, basis))
                    if den <= 0:
                        continue
                    # Constrain magnitude scale to a physically plausible range.
                    m0_try = max(float(np.dot(y, basis) / den), 0.0)

                best = _GridCandidate(
                    m0=m0_try,
                    t1_ms=float(t1_try),
                    b1=float(b1_try),
                    score=score,
                )
        return best

    def _fit_lut_result(
        self,
        *,
        y: NDArray[np.float64],
        use_uni: bool,
        candidate: _GridCandidate,
    ) -> dict[str, float]:
        if use_uni:
            return {
                "m0": float("nan"),
                "t1_ms": candidate.t1_ms,
                "b1": candidate.b1,
                "res_rmse": candidate.score,
            }

        pred = self.forward(m0=candidate.m0, t1_ms=candidate.t1_ms, b1=candidate.b1)
        return {
            "m0": candidate.m0,
            "t1_ms": candidate.t1_ms,
            "b1": candidate.b1,
            "res_rmse": _rmse(pred, y),
        }

    def _fit_nls_joint(
        self,
        *,
        y: NDArray[np.float64],
        t1_lo: float,
        t1_hi: float,
        b1_lo: float,
        b1_hi: float,
        candidate: _GridCandidate,
    ) -> dict[str, float]:
        import numpy as np
        from scipy.optimize import least_squares

        lower = np.array([0.0, t1_lo, b1_lo], dtype=np.float64)
        upper = np.array([np.inf, t1_hi, b1_hi], dtype=np.float64)
        x0 = np.clip(np.array([candidate.m0, candidate.t1_ms, candidate.b1], dtype=np.float64), lower, upper)

        def residuals(params: NDArray[np.float64]) -> NDArray[np.float64]:
            m0_val = float(params[0])
            t1_val = float(params[1])
            b1_val = float(params[2])
            return self.forward(m0=m0_val, t1_ms=t1_val, b1=b1_val) - y

        result = least_squares(residuals, x0=x0, bounds=(lower, upper))
        m0_hat, t1_hat, b1_hat = [float(v) for v in result.x]
        pred = self.forward(m0=m0_hat, t1_ms=t1_hat, b1=b1_hat)
        return {
            "m0": m0_hat,
            "t1_ms": t1_hat,
            "b1": b1_hat,
            "res_rmse": _rmse(pred, y),
        }

    def _fit_nls_fixed(
        self,
        *,
        y: NDArray[np.float64],
        t1_lo: float,
        t1_hi: float,
        candidate: _GridCandidate,
    ) -> dict[str, float]:
        import numpy as np
        from scipy.optimize import least_squares

        lower = np.array([0.0, t1_lo], dtype=np.float64)
        upper = np.array([np.inf, t1_hi], dtype=np.float64)
        x0 = np.clip(np.array([candidate.m0, candidate.t1_ms], dtype=np.float64), lower, upper)

        def residuals(params: NDArray[np.float64]) -> NDArray[np.float64]:
            m0_val = float(params[0])
            t1_val = float(params[1])
            return self.forward(m0=m0_val, t1_ms=t1_val, b1=candidate.b1) - y

        result = least_squares(residuals, x0=x0, bounds=(lower, upper))
        m0_hat, t1_hat = [float(v) for v in result.x]
        pred = self.forward(m0=m0_hat, t1_ms=t1_hat, b1=candidate.b1)
        return {
            "m0": m0_hat,
            "t1_ms": t1_hat,
            "b1": candidate.b1,
            "res_rmse": _rmse(pred, y),
        }

    def fit(
        self,
        signal: ArrayLike,
        *,
        method: str = "nls",
        estimate_b1: bool = False,
        b1: float | None = None,
        b1_bounds: tuple[float, float] = (0.5, 1.5),
        b0_hz: float | None = None,
        t1_bounds_ms: tuple[float, float] = (200.0, 6000.0),
        t1_grid_ms: ArrayLike | None = None,
    ) -> dict[str, float]:
        y = _as_1d_float_array(signal, name="signal")
        if y.size not in {1, 2}:
            raise ValueError("signal must be shape (2,) as [INV1, INV2] or shape (1,) for UNI")

        _ = b0_hz

        if estimate_b1 and b1 is not None:
            raise ValueError("use either estimate_b1=True or fixed b1")

        if method not in {"nls", "lut"}:
            raise ValueError("method must be 'nls' or 'lut'")

        t1_lo, t1_hi, t1_grid = self._prepare_t1_grid(
            t1_bounds_ms=t1_bounds_ms,
            t1_grid_ms=t1_grid_ms,
        )
        b1_lo, b1_hi, b1_grid = self._prepare_b1_grid(
            estimate_b1=estimate_b1,
            b1=b1,
            b1_bounds=b1_bounds,
        )

        use_uni = bool(y.size == 1)
        candidate = self._grid_search_candidate(
            y=y,
            use_uni=use_uni,
            t1_grid=t1_grid,
            b1_grid=b1_grid,
        )

        if method == "lut":
            return self._fit_lut_result(y=y, use_uni=use_uni, candidate=candidate)

        if use_uni:
            raise ValueError("method='nls' requires [INV1, INV2] input")

        if estimate_b1:
            return self._fit_nls_joint(
                y=y,
                t1_lo=t1_lo,
                t1_hi=t1_hi,
                b1_lo=b1_lo,
                b1_hi=b1_hi,
                candidate=candidate,
            )

        return self._fit_nls_fixed(
            y=y,
            t1_lo=t1_lo,
            t1_hi=t1_hi,
            candidate=candidate,
        )

    def fit_image(
        self,
        signal: ArrayLike,
        *,
        mask: ArrayLike | str | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        import numpy as np

        from qmrpy._mask import resolve_mask
        from qmrpy._parallel import parallel_fit

        arr = np.asarray(signal, dtype=np.float64)
        if arr.ndim == 1:
            if mask is not None:
                raise ValueError("mask must be None for 1D data")
            return self.fit(arr, **kwargs)

        if arr.shape[-1] != 2:
            raise ValueError("data last dim must be 2 as [INV1, INV2]")

        spatial_shape = arr.shape[:-1]
        flat = arr.reshape((-1, arr.shape[-1]))

        resolved_mask = resolve_mask(mask, arr)
        if resolved_mask is None:
            mask_flat = np.ones((flat.shape[0],), dtype=bool)
        else:
            if resolved_mask.shape != spatial_shape:
                raise ValueError(f"mask shape {resolved_mask.shape} must match spatial shape {spatial_shape}")
            mask_flat = resolved_mask.reshape((-1,))

        def fit_func(signal_1d: NDArray[Any]) -> dict[str, float]:
            return self.fit(signal_1d, **kwargs)

        return parallel_fit(
            fit_func,
            flat,
            mask_flat,
            ["m0", "t1_ms", "b1", "res_rmse"],
            spatial_shape,
            n_jobs=n_jobs,
            verbose=verbose,
            desc="T1MP2RAGE",
        )
