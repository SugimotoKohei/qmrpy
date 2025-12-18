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


def _logspace_range(lo: float, hi: float, n: int) -> NDArray[np.float64]:
    import numpy as np

    lo = float(lo)
    hi = float(hi)
    n = int(n)
    if lo <= 0 or hi <= 0 or hi <= lo:
        raise ValueError("invalid logspace bounds")
    if n < 2:
        raise ValueError("n must be >= 2")
    return np.logspace(np.log10(lo), np.log10(hi), n)


def _sigmoid_weights(*, t2_times_s: NDArray[np.float64], spwin_hi_s: float, sigmoid_s: float) -> NDArray[np.float64]:
    """Port of DECAES.jl `sigmoid_weights` (T2partSEcorr.jl).

    Uses a complementary normal CDF (survival function) to smooth the SPWin upper cutoff.
    """

    import numpy as np
    from scipy.special import erfc, erfinv

    k = 0.1
    t2_kperc = float(sigmoid_s)
    t2_50perc = float(spwin_hi_s)

    sigma = abs(t2_kperc / (np.sqrt(2.0) * float(erfinv(2 * k - 1))))
    z = (np.asarray(t2_times_s, dtype=np.float64) - t2_50perc) / sigma

    # normccdf(z) = 0.5 * erfc(z/sqrt(2))
    w = 0.5 * erfc(z / np.sqrt(2.0))
    w[w <= np.finfo(float).eps] = 0.0
    return w


@dataclass(frozen=True, slots=True)
class DecaesT2Part:
    """DECAES-like T2-parts analysis (`T2partSEcorr`) on precomputed T2 distributions.

    Units are seconds.
    """

    n_t2: int
    t2_range_s: tuple[float, float]
    spwin_s: tuple[float, float]
    mpwin_s: tuple[float, float]
    sigmoid_s: float | None = None

    def __post_init__(self) -> None:
        if int(self.n_t2) < 2:
            raise ValueError("n_t2 must be >= 2")
        lo, hi = self.t2_range_s
        if lo <= 0 or hi <= 0 or hi <= lo:
            raise ValueError("t2_range_s must be (lo, hi) with 0 < lo < hi")
        if self.spwin_s[0] >= self.spwin_s[1]:
            raise ValueError("spwin_s must be (lo, hi) with lo < hi")
        if self.mpwin_s[0] >= self.mpwin_s[1]:
            raise ValueError("mpwin_s must be (lo, hi) with lo < hi")
        if self.sigmoid_s is not None and float(self.sigmoid_s) <= 0:
            raise ValueError("sigmoid_s must be > 0")

    def t2_times_s(self) -> NDArray[np.float64]:
        lo, hi = self.t2_range_s
        return _logspace_range(lo, hi, self.n_t2)

    def fit(self, distribution: ArrayLike) -> dict[str, float]:
        import numpy as np

        dist = _as_1d_float_array(distribution, name="distribution")
        if dist.shape != (self.n_t2,):
            raise ValueError(f"distribution must be shape ({self.n_t2},), got {dist.shape}")
        if np.any(np.isnan(dist)):
            return {"sfr": float("nan"), "sgm": float("nan"), "mfr": float("nan"), "mgm": float("nan")}

        t2 = self.t2_times_s()
        logt2 = np.log(t2)

        sp = np.where((t2 >= self.spwin_s[0]) & (t2 <= self.spwin_s[1]))[0]
        mp = np.where((t2 >= self.mpwin_s[0]) & (t2 <= self.mpwin_s[1]))[0]

        sum_all = float(np.sum(dist))
        sum_sp = float(np.sum(dist[sp])) if sp.size else 0.0
        sum_mp = float(np.sum(dist[mp])) if mp.size else 0.0

        sfr = float("nan")
        mfr = float("nan")
        sgm = float("nan")
        mgm = float("nan")

        if sum_all > 0:
            if self.sigmoid_s is None:
                sfr = float(sum_sp / sum_all)
            else:
                w = _sigmoid_weights(t2_times_s=t2, spwin_hi_s=self.spwin_s[1], sigmoid_s=float(self.sigmoid_s))
                sfr = float(np.dot(dist, w) / sum_all)
            mfr = float(sum_mp / sum_all)

        if sum_sp > 0:
            sgm = float(np.exp(float(np.dot(dist[sp], logt2[sp]) / sum_sp)))
        if sum_mp > 0:
            mgm = float(np.exp(float(np.dot(dist[mp], logt2[mp]) / sum_mp)))

        return {"sfr": float(sfr), "sgm": float(sgm), "mfr": float(mfr), "mgm": float(mgm)}

    def fit_image(self, distributions: ArrayLike) -> dict[str, NDArray[np.float64]]:
        import numpy as np

        dist4 = np.asarray(distributions, dtype=np.float64)
        if dist4.ndim != 4 or dist4.shape[-1] != self.n_t2:
            raise ValueError(f"distributions must be 4D with last dim n_t2={self.n_t2}")

        sfr = np.full(dist4.shape[:3], np.nan, dtype=np.float64)
        sgm = np.full(dist4.shape[:3], np.nan, dtype=np.float64)
        mfr = np.full(dist4.shape[:3], np.nan, dtype=np.float64)
        mgm = np.full(dist4.shape[:3], np.nan, dtype=np.float64)

        for idx in np.ndindex(dist4.shape[:3]):
            out = self.fit(dist4[idx])
            sfr[idx] = out["sfr"]
            sgm[idx] = out["sgm"]
            mfr[idx] = out["mfr"]
            mgm[idx] = out["mgm"]

        return {"sfr": sfr, "sgm": sgm, "mfr": mfr, "mgm": mgm}
