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
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


@dataclass(frozen=True, slots=True)
class MRFDictionary:
    """Dictionary-based simultaneous T1-T2 MR fingerprinting model.

    This minimal implementation simulates a spoiled FISP-like transient using
    the qmrpy EPG state engine. At each frame, an RF pulse is applied, signal is
    sampled after ``te_ms``, transverse states are spoiled, and longitudinal
    recovery continues for the remainder of TR.

    Units
    -----
    flip_angle_deg : degrees
    tr_ms, te_ms : milliseconds
    t1_ms, t2_ms : milliseconds

    References
    ----------
    .. [1] Ma D, Gulani V, Seiberlich N, et al. (2013). Magnetic resonance
           fingerprinting. Nature, 495:187-192.
    .. [2] Jiang Y, Ma D, Seiberlich N, Gulani V, Griswold MA (2015). MR
           fingerprinting using fast imaging with steady state precession
           (FISP) with spiral readout. Magnetic Resonance in Medicine,
           74(6):1621-1631.
    .. [3] Weigel M (2015). Extended phase graphs: dephasing, RF pulses, and
           echoes - pure and simple. Journal of Magnetic Resonance Imaging,
           41(2):266-295.
    """

    flip_angle_deg: ArrayLike
    tr_ms: ArrayLike
    te_ms: ArrayLike | float | None = None

    def __post_init__(self) -> None:
        import numpy as np

        fa = _as_1d_float_array(self.flip_angle_deg, name="flip_angle_deg")
        tr = _as_1d_float_array(self.tr_ms, name="tr_ms")
        if fa.shape != tr.shape:
            raise ValueError(f"flip_angle_deg shape {fa.shape} must match tr_ms shape {tr.shape}")
        if np.any(fa <= 0):
            raise ValueError("flip_angle_deg must be > 0")
        if np.any(tr <= 0):
            raise ValueError("tr_ms must be > 0")
        if self.te_ms is None:
            te = tr / 2.0
        else:
            te_raw = np.asarray(self.te_ms, dtype=np.float64)
            if te_raw.ndim == 0:
                te = np.full_like(tr, float(te_raw))
            else:
                te = _as_1d_float_array(te_raw, name="te_ms")
        if te.shape != tr.shape:
            raise ValueError(f"te_ms shape {te.shape} must match tr_ms shape {tr.shape}")
        if np.any(te < 0):
            raise ValueError("te_ms must be non-negative")
        if np.any(te > tr):
            raise ValueError("te_ms must be <= tr_ms for every frame")
        object.__setattr__(self, "flip_angle_deg", fa)
        object.__setattr__(self, "tr_ms", tr)
        object.__setattr__(self, "te_ms", te)

    def forward(
        self, *, m0: float = 1.0, t1_ms: float, t2_ms: float, b1: float = 1.0
    ) -> NDArray[np.float64]:
        """Simulate one MRF fingerprint."""
        import numpy as np

        from qmrpy.epg import EPGSimulator

        if not np.isfinite(m0):
            raise ValueError("m0 must be finite")
        if not np.isfinite(t1_ms) or t1_ms <= 0:
            raise ValueError("t1_ms must be finite and > 0")
        if not np.isfinite(t2_ms) or t2_ms <= 0:
            raise ValueError("t2_ms must be finite and > 0")
        if not np.isfinite(b1) or b1 <= 0:
            raise ValueError("b1 must be finite and > 0")

        sim = EPGSimulator(n_states=2, t1_ms=float(t1_ms), t2_ms=float(t2_ms))
        sim.reset(m0=1.0)
        out = np.zeros(self.flip_angle_deg.shape[0], dtype=np.float64)

        for i, (fa, tr, te) in enumerate(
            zip(self.flip_angle_deg, self.tr_ms, self.te_ms, strict=True)
        ):
            sim.apply_rf(float(fa) * float(b1))
            if te > 0:
                sim.apply_relaxation(float(te), recovery=False)
            out[i] = sim.get_signal_magnitude()
            sim.apply_spoiler()
            remaining = float(tr - te)
            if remaining > 0:
                sim.apply_relaxation(remaining, recovery=True)

        return float(m0) * out

    def generate_dictionary(
        self,
        *,
        t1_grid_ms: ArrayLike,
        t2_grid_ms: ArrayLike,
        b1: float = 1.0,
    ) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]]:
        """Generate normalized MRF dictionary fingerprints."""
        import numpy as np

        t1_grid = _as_positive_grid(t1_grid_ms, name="t1_grid_ms")
        t2_grid = _as_positive_grid(t2_grid_ms, name="t2_grid_ms")
        fingerprints = []
        norms = []
        t1_values = []
        t2_values = []
        for t1 in t1_grid:
            for t2 in t2_grid:
                if t2 > t1:
                    continue
                fp = self.forward(m0=1.0, t1_ms=float(t1), t2_ms=float(t2), b1=b1)
                fp_norm = _norm(fp)
                fingerprints.append(fp / fp_norm)
                norms.append(fp_norm)
                t1_values.append(float(t1))
                t2_values.append(float(t2))
        if not fingerprints:
            raise ValueError("dictionary grid produced no valid entries")
        dictionary = np.stack(fingerprints, axis=0)
        params = {
            "t1_ms": np.asarray(t1_values, dtype=np.float64),
            "t2_ms": np.asarray(t2_values, dtype=np.float64),
            "norm": np.asarray(norms, dtype=np.float64),
        }
        return dictionary, params

    def fit(
        self,
        signal: ArrayLike,
        *,
        t1_grid_ms: ArrayLike,
        t2_grid_ms: ArrayLike,
        b1: float = 1.0,
        dictionary: NDArray[np.float64] | None = None,
        dictionary_params: dict[str, NDArray[np.float64]] | None = None,
    ) -> dict[str, float]:
        """Match one signal to the T1/T2 dictionary by normalized inner product."""
        import numpy as np

        y = _as_1d_float_array(signal, name="signal")
        if y.shape != self.flip_angle_deg.shape:
            raise ValueError(
                f"signal shape {y.shape} must match flip_angle_deg shape {self.flip_angle_deg.shape}"
            )
        if dictionary is None or dictionary_params is None:
            dictionary, dictionary_params = self.generate_dictionary(
                t1_grid_ms=t1_grid_ms,
                t2_grid_ms=t2_grid_ms,
                b1=b1,
            )
        dictionary = np.asarray(dictionary, dtype=np.float64)
        if dictionary.ndim != 2 or dictionary.shape[1] != y.shape[0]:
            raise ValueError("dictionary must be 2D with second dim matching signal length")
        y_norm = _normalize(y)
        scores = dictionary @ y_norm
        idx = int(np.argmax(scores))
        best = dictionary[idx]
        denom = float(np.dot(best, best))
        scale = float(np.dot(y, best) / denom) if denom > 0 else float("nan")
        fp_norm = float(dictionary_params.get("norm", np.ones(dictionary.shape[0]))[idx])
        m0 = scale / fp_norm
        return {
            "m0": m0,
            "t1_ms": float(dictionary_params["t1_ms"][idx]),
            "t2_ms": float(dictionary_params["t2_ms"][idx]),
            "dictionary_index": float(idx),
            "correlation": float(scores[idx]),
        }

    def fit_image(
        self,
        signal: ArrayLike,
        *,
        t1_grid_ms: ArrayLike,
        t2_grid_ms: ArrayLike,
        b1: float = 1.0,
        mask: ArrayLike | str | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Voxel-wise MRF dictionary matching."""
        import numpy as np

        from qmrpy._mask import resolve_mask
        from qmrpy._parallel import parallel_fit

        arr = np.asarray(signal, dtype=np.float64)
        if arr.ndim == 1:
            if mask is not None:
                raise ValueError("mask must be None for 1D data")
            return self.fit(arr, t1_grid_ms=t1_grid_ms, t2_grid_ms=t2_grid_ms, b1=b1)
        if arr.shape[-1] != self.flip_angle_deg.shape[0]:
            raise ValueError(
                f"data last dim {arr.shape[-1]} must match sequence length {self.flip_angle_deg.shape[0]}"
            )

        dictionary, dictionary_params = self.generate_dictionary(
            t1_grid_ms=t1_grid_ms,
            t2_grid_ms=t2_grid_ms,
            b1=b1,
        )
        spatial_shape = arr.shape[:-1]
        flat = arr.reshape((-1, arr.shape[-1]))

        resolved_mask = resolve_mask(mask, arr)
        if resolved_mask is None:
            mask_flat = np.ones((flat.shape[0],), dtype=bool)
        else:
            if resolved_mask.shape != spatial_shape:
                raise ValueError(
                    f"mask shape {resolved_mask.shape} must match spatial shape {spatial_shape}"
                )
            mask_flat = resolved_mask.reshape((-1,))

        def fit_func(signal_1d: NDArray[Any]) -> dict[str, float]:
            return self.fit(
                signal_1d,
                t1_grid_ms=t1_grid_ms,
                t2_grid_ms=t2_grid_ms,
                b1=b1,
                dictionary=dictionary,
                dictionary_params=dictionary_params,
            )

        return parallel_fit(
            fit_func,
            flat,
            mask_flat,
            ["m0", "t1_ms", "t2_ms", "dictionary_index", "correlation"],
            spatial_shape,
            n_jobs=n_jobs,
            verbose=verbose,
            desc="MRFDictionary",
        )


def _as_positive_grid(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    import numpy as np

    grid = _as_1d_float_array(values, name=name)
    if np.any(grid <= 0):
        raise ValueError(f"{name} must be > 0")
    return grid


def _normalize(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return values / _norm(values)


def _norm(values: NDArray[np.float64]) -> float:
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm <= 0:
        raise ValueError("signal norm must be > 0")
    return norm
