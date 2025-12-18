from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray
else:
    ArrayLike = Any  # type: ignore[misc,assignment]
    NDArray = Any  # type: ignore[misc,assignment]


def _as_1d_float_array(values: ArrayLike, *, name: str):
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={array.shape}")
    return array


@dataclass(frozen=True, slots=True)
class MonoT2:
    """Mono-exponential T2 relaxometry model.

    Signal model:
        S(TE) = m0 * exp(-TE / T2)
    """

    te: Any

    def __post_init__(self) -> None:
        import numpy as np

        te_array = _as_1d_float_array(self.te, name="te")
        if np.any(te_array < 0):
            raise ValueError("te must be non-negative")
        object.__setattr__(self, "te", te_array)

    def forward(self, *, m0: float, t2: float):
        import numpy as np

        if t2 <= 0:
            raise ValueError("t2 must be > 0")
        return m0 * np.exp(-self.te / t2)

    def fit(
        self,
        signal: ArrayLike,
        *,
        m0_init: float | None = None,
        t2_init: float | None = None,
        bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    ) -> dict[str, float]:
        """Fit m0 and T2 using non-linear least squares.

        Parameters
        ----------
        signal:
            1D signal samples at the model's `te`.
        m0_init, t2_init:
            Optional initial guesses. If omitted, they are estimated heuristically.
        bounds:
            Optional bounds as ((m0_min, t2_min), (m0_max, t2_max)).

        Returns
        -------
        dict with keys: "m0", "t2".
        """
        import numpy as np
        from scipy.optimize import least_squares

        y = _as_1d_float_array(signal, name="signal")
        if y.shape != self.te.shape:
            raise ValueError(f"signal shape {y.shape} must match te shape {self.te.shape}")

        if m0_init is None:
            m0_init = float(y[0])
        if t2_init is None:
            positive_te = self.te[self.te > 0]
            t2_init = float(np.median(positive_te)) if positive_te.size else 50.0
        if bounds is None:
            lower = (0.0, 1e-6)
            upper = (np.inf, np.inf)
        else:
            lower, upper = bounds

        def residuals(params: NDArray[np.float64]) -> NDArray[np.float64]:
            m0_value = float(params[0])
            t2_value = float(params[1])
            return self.forward(m0=m0_value, t2=t2_value) - y

        result = least_squares(
            residuals,
            x0=np.array([m0_init, t2_init], dtype=np.float64),
            bounds=(np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)),
        )
        m0_hat, t2_hat = result.x
        return {"m0": float(m0_hat), "t2": float(t2_hat)}
