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


def _element_flipmat(alpha_deg: float) -> NDArray[np.complex128]:
    """DECAES element flip matrix (Hennig 1988), ported from DECAES.jl."""

    import numpy as np

    a2 = float(alpha_deg) / 2.0
    cos2 = np.cos(np.deg2rad(a2))
    sin2 = np.sin(np.deg2rad(a2))
    sin = np.sin(np.deg2rad(float(alpha_deg)))

    return np.array(
        [
            [cos2**2, sin2**2, -1j * sin],
            [sin2**2, cos2**2, 1j * sin],
            [-1j * sin / 2.0, 1j * sin / 2.0, np.cos(np.deg2rad(float(alpha_deg)))],
        ],
        dtype=np.complex128,
    )


def _epg_decay_curve_decaes(
    *,
    etl: int,
    alpha_deg: float,
    te_s: float,
    t2_s: float,
    t1_s: float,
    beta_deg: float,
) -> NDArray[np.float64]:
    """Compute normalized MSE echo decay curve using EPG w/ stimulated echo correction.

    Port of `DECAES.jl/src/EPGdecaycurve.jl:epg_decay_curve!(::EPGWork_Basic_Cplx, ::EPGOptions)`.
    """

    import numpy as np

    etl = int(etl)
    if etl < 1:
        raise ValueError("etl must be >= 1")
    te_s = float(te_s)
    if te_s <= 0:
        raise ValueError("te_s must be > 0")
    t2_s = float(t2_s)
    t1_s = float(t1_s)
    if t2_s <= 0:
        raise ValueError("t2_s must be > 0")
    if t1_s <= 0:
        raise ValueError("t1_s must be > 0")

    A = float(alpha_deg) / 180.0
    alpha_ex = A * 90.0
    alpha1 = A * 180.0
    alphai = A * float(beta_deg)

    # Relaxation for TE/2
    E1 = float(np.exp(-((te_s / 2.0) / t1_s)))
    E2 = float(np.exp(-((te_s / 2.0) / t2_s)))
    E = np.array([E2, E2, E1], dtype=np.complex128)

    R1 = _element_flipmat(alpha1)
    Ri = _element_flipmat(alphai)

    # Magnetization phase state vector (ETL x 3)
    MPSV = np.zeros((etl, 3), dtype=np.complex128)
    MPSV[0, 0] = np.sin(np.deg2rad(alpha_ex))

    dc = np.zeros(etl, dtype=np.float64)

    for i in range(etl):
        R = R1 if i == 0 else Ri

        # Relaxation for TE/2 then flip
        MPSV = (R @ (E * MPSV).T).T

        # Transition between phase states (Jones 1997 correction)
        if etl >= 2:
            Mi = MPSV[0].copy()
            Mip1 = MPSV[1].copy()
            MPSV[0] = np.array([Mi[1], Mip1[1], Mi[2]], dtype=np.complex128)

            Mim1 = Mi
            Mi = Mip1
            for j in range(1, etl - 1):
                Mip1 = MPSV[j + 1].copy()
                MPSV[j] = np.array([Mim1[0], Mip1[1], Mi[2]], dtype=np.complex128)
                Mim1, Mi = Mi, Mip1

            MPSV[etl - 1] = np.array([Mim1[0], 0.0 + 0.0j, Mi[2]], dtype=np.complex128)

        # Relaxation for TE/2
        MPSV = E * MPSV

        dc[i] = float(np.abs(MPSV[0, 0]))

    return dc


def _epg_decay_curve_epgpy(
    *,
    etl: int,
    alpha_deg: float,
    te_s: float,
    t2_s: float,
    t1_s: float,
    beta_deg: float,
) -> NDArray[np.float64]:
    """Compute normalized MSE echo decay curve using vendored epgpy."""
    import numpy as np

    from epgpy import functions, operators

    etl = int(etl)
    if etl < 1:
        raise ValueError("etl must be >= 1")
    te_s = float(te_s)
    if te_s <= 0:
        raise ValueError("te_s must be > 0")
    t2_s = float(t2_s)
    t1_s = float(t1_s)
    if t2_s <= 0:
        raise ValueError("t2_s must be > 0")
    if t1_s <= 0:
        raise ValueError("t1_s must be > 0")

    A = float(alpha_deg) / 180.0
    alpha_ex = A * 90.0
    alpha1 = A * 180.0
    alphai = A * float(beta_deg)

    tau_ms = float(te_s) * 1000.0 / 2.0
    t1_ms = float(t1_s) * 1000.0
    t2_ms = float(t2_s) * 1000.0

    seq = [operators.T(alpha_ex, 0.0)]
    for echo in range(etl):
        ref = alpha1 if echo == 0 else alphai
        seq.extend(
            [
                operators.E(tau_ms, t1_ms, t2_ms),
                operators.S(1),
                operators.T(ref, 0.0),
                operators.S(1),
                operators.E(tau_ms, t1_ms, t2_ms),
                operators.ADC,
            ]
        )

    values = functions.simulate(seq, asarray=True)
    y = np.asarray(values, dtype=np.complex128).reshape(-1)
    return np.abs(y).astype(np.float64)


def epg_decay_curve(
    *,
    etl: int,
    alpha_deg: float,
    te_s: float,
    t2_s: float,
    t1_s: float,
    beta_deg: float,
    backend: str = "epgpy",
) -> NDArray[np.float64]:
    """Compute normalized MSE echo decay curve using EPG backend.

    backend:
      - "epgpy": use vendored `epgpy` (default)
      - "decaes": use DECAES-style reduced EPG implementation
    """
    backend_norm = str(backend).lower().strip()
    if backend_norm == "epgpy":
        return _epg_decay_curve_epgpy(
            etl=etl,
            alpha_deg=alpha_deg,
            te_s=te_s,
            t2_s=t2_s,
            t1_s=t1_s,
            beta_deg=beta_deg,
        )
    if backend_norm == "decaes":
        return _epg_decay_curve_decaes(
            etl=etl,
            alpha_deg=alpha_deg,
            te_s=te_s,
            t2_s=t2_s,
            t1_s=t1_s,
            beta_deg=beta_deg,
        )
    raise ValueError("backend must be 'epgpy' or 'decaes'")


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


def _basis_matrix(
    *,
    n_te: int,
    te_s: float,
    t2_times_s,
    t1_s: float,
    alpha_deg: float,
    refcon_angle_deg: float,
    epg_backend: str,
) -> NDArray[np.float64]:
    import numpy as np

    A = np.zeros((n_te, len(t2_times_s)), dtype=np.float64)
    for j, t2 in enumerate(t2_times_s):
        A[:, j] = epg_decay_curve(
            etl=n_te,
            alpha_deg=float(alpha_deg),
            te_s=float(te_s),
            t2_s=float(t2),
            t1_s=float(t1_s),
            beta_deg=float(refcon_angle_deg),
            backend=str(epg_backend),
        )
    return A


def _basis_matrix_dalpha_fd(
    *,
    n_te: int,
    te_s: float,
    t2_times_s,
    t1_s: float,
    alpha_deg: float,
    refcon_angle_deg: float,
    epg_backend: str,
    alpha_min_deg: float,
    h_deg: float = 1e-3,
):
    """Finite-difference derivative dA/dalpha matching DECAES' ∇A intent."""

    import numpy as np

    a = float(alpha_deg)
    h = float(h_deg)
    if a - h < float(alpha_min_deg):
        A0 = _basis_matrix(
            n_te=n_te,
            te_s=te_s,
            t2_times_s=t2_times_s,
            t1_s=t1_s,
            alpha_deg=a,
            refcon_angle_deg=refcon_angle_deg,
            epg_backend=epg_backend,
        )
        A1 = _basis_matrix(
            n_te=n_te,
            te_s=te_s,
            t2_times_s=t2_times_s,
            t1_s=t1_s,
            alpha_deg=a + h,
            refcon_angle_deg=refcon_angle_deg,
            epg_backend=epg_backend,
        )
        return (A1 - A0) / h
    if a + h > 180.0:
        A0 = _basis_matrix(
            n_te=n_te,
            te_s=te_s,
            t2_times_s=t2_times_s,
            t1_s=t1_s,
            alpha_deg=a - h,
            refcon_angle_deg=refcon_angle_deg,
            epg_backend=epg_backend,
        )
        A1 = _basis_matrix(
            n_te=n_te,
            te_s=te_s,
            t2_times_s=t2_times_s,
            t1_s=t1_s,
            alpha_deg=a,
            refcon_angle_deg=refcon_angle_deg,
            epg_backend=epg_backend,
        )
        return (A1 - A0) / h

    A_plus = _basis_matrix(
        n_te=n_te,
        te_s=te_s,
        t2_times_s=t2_times_s,
        t1_s=t1_s,
        alpha_deg=a + h,
        refcon_angle_deg=refcon_angle_deg,
        epg_backend=epg_backend,
    )
    A_minus = _basis_matrix(
        n_te=n_te,
        te_s=te_s,
        t2_times_s=t2_times_s,
        t1_s=t1_s,
        alpha_deg=a - h,
        refcon_angle_deg=refcon_angle_deg,
        epg_backend=epg_backend,
    )
    return (A_plus - A_minus) / (2.0 * h)


def _gcv_dof(m: int, s: Any, mu: float) -> float:
    import numpy as np

    mu2 = float(mu) * float(mu)
    s2 = np.asarray(s, dtype=np.float64) ** 2
    return float(m - np.sum(s2 / (s2 + mu2)))


def _gcv_objective(A: Any, b: Any, s: Any, mu: float) -> float:
    import numpy as np

    from qmrpy._decaes.nnls import nnls_tikhonov

    mu = float(mu)
    res = nnls_tikhonov(A, b, mu)
    r = A @ res.x - b
    rss = float(r @ r)
    t = _gcv_dof(A.shape[0], s, mu)
    return float(rss / max(t * t, float(np.finfo(float).eps)))


def _choose_mu(
    A: Any,
    b: Any,
    *,
    reg: str,
    chi2_factor: float | None,
    noise_level: float | None,
):
    import numpy as np
    from scipy.optimize import brentq, minimize_scalar

    reg = reg.lower().strip()

    if reg == "none":
        return 0.0

    # unregularized residual baseline
    from qmrpy._decaes.nnls import nnls_tikhonov

    r0 = nnls_tikhonov(A, b, 0.0)
    rvec0 = A @ r0.x - b
    res0_sq = float(rvec0 @ rvec0)

    # bounds in DECAES (logmu)
    lo, hi = -8.0, 2.0

    if reg == "chi2":
        if chi2_factor is None or chi2_factor <= 1.0:
            raise ValueError("chi2_factor must be > 1.0 when reg='chi2'")
        target = float(chi2_factor) * res0_sq

        def f(logmu: float) -> float:
            mu = float(np.exp(logmu))
            r = nnls_tikhonov(A, b, mu)
            rv = A @ r.x - b
            return float(rv @ rv - target)

        # bracket
        a, b_ = lo, hi
        fa, fb = f(a), f(b_)
        if fa * fb > 0:
            # fallback: choose closest
            return float(np.exp(a if abs(fa) < abs(fb) else b_))
        return float(np.exp(brentq(f, a, b_, xtol=0.0, rtol=0.0, maxiter=100)))

    if reg == "mdp":
        if noise_level is None or noise_level <= 0.0:
            raise ValueError("noise_level must be > 0 when reg='mdp'")
        delta = float(np.sqrt(A.shape[0]) * noise_level)
        target = delta * delta

        def f(logmu: float) -> float:
            mu = float(np.exp(logmu))
            r = nnls_tikhonov(A, b, mu)
            rv = A @ r.x - b
            return float(rv @ rv - target)

        a, b_ = lo, hi
        fa, fb = f(a), f(b_)
        if fa * fb > 0:
            return float(np.exp(a if abs(fa) < abs(fb) else b_))
        return float(np.exp(brentq(f, a, b_, xtol=0.0, rtol=0.0, maxiter=100)))

    if reg == "gcv":
        # SVD singular values
        s = np.linalg.svd(A, compute_uv=False)

        def obj(logmu: float) -> float:
            mu = float(np.exp(logmu))
            return float(np.log(max(_gcv_objective(A, b, s, mu), np.finfo(float).tiny)))

        res = minimize_scalar(obj, bounds=(lo, hi), method="bounded", options={"xatol": 1e-4})
        return float(np.exp(float(res.x)))

    if reg == "lcurve":
        # Minimal port of DECAES lcurve_corner: maximize Menger curvature on (log||Ax-b||^2, log||x||^2)
        from qmrpy._decaes.nnls import nnls_tikhonov

        def point(logmu: float):
            mu = float(np.exp(logmu))
            r = nnls_tikhonov(A, b, mu)
            rv = A @ r.x - b
            xi = float(np.log(max(float(rv @ rv), np.finfo(float).tiny)))
            eta = float(np.log(max(float(r.x @ r.x), np.finfo(float).tiny)))
            return np.array([xi, eta], dtype=np.float64)

        phi = (1 + 5**0.5) / 2

        # caches
        P_cache: dict[float, np.ndarray] = {}
        C_cache: dict[float, float] = {}

        def getP(x: float) -> np.ndarray:
            if x not in P_cache:
                P_cache[x] = point(x)
            return P_cache[x]

        def menger(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
            a = np.linalg.norm(p2 - p1)
            b_ = np.linalg.norm(p3 - p2)
            c_ = np.linalg.norm(p1 - p3)
            s_ = (a + b_ + c_) / 2.0
            area2 = max(s_ * (s_ - a) * (s_ - b_) * (s_ - c_), 0.0)
            if area2 == 0.0 or a * b_ * c_ == 0.0:
                return float("-inf")
            area = np.sqrt(area2)
            return float(4.0 * area / (a * b_ * c_))

        def curvature(x: float) -> float:
            if x in C_cache:
                return C_cache[x]
            xs = sorted(P_cache.keys())
            # ensure we have at least neighbors
            if x not in xs:
                xs.append(x)
                xs.sort()
            i = xs.index(x)
            if i == 0 or i == len(xs) - 1:
                C_cache[x] = float("-inf")
                return C_cache[x]
            p_prev = getP(xs[i - 1])
            p = getP(x)
            p_next = getP(xs[i + 1])
            C_cache[x] = menger(p_prev, p, p_next)
            return C_cache[x]

        # initial state
        x1, x4 = float(lo), float(hi)
        x2 = (phi * x1 + x4) / (phi + 1)
        x3 = x1 + (x4 - x2)
        for x in (x1, x2, x3, x4):
            getP(x)

        xtol = 1e-4
        for _ in range(200):
            # update curvatures (needs neighbors)
            xs = sorted(P_cache.keys())
            for x in xs[1:-1]:
                curvature(x)

            # choose direction based on C(x2) vs C(x3)
            c2 = curvature(x2)
            c3 = curvature(x3)
            if c2 > c3:
                # move left
                x4 = x3
                x3 = x2
                x2 = (phi * x1 + x3) / (phi + 1)
                getP(x2)
            else:
                # move right
                x1 = x2
                x2 = x3
                x3 = x2 + (x4 - x2) / (phi + 1)
                getP(x3)

            if abs(x4 - x1) < xtol:
                break

        # final: pick max curvature among cached points
        xs = sorted(P_cache.keys())
        best_x = xs[0]
        best_c = float("-inf")
        for x in xs[1:-1]:
            c = curvature(x)
            if c > best_c:
                best_c, best_x = c, x
        return float(np.exp(best_x))

    raise ValueError(f"Unknown reg: {reg}")


@dataclass(frozen=True, slots=True)
class DecaesT2Map:
    """DECAES-like multi-component T2 mapping (T2mapSEcorr + core outputs)."""

    n_te: int
    te_s: float
    n_t2: int
    t2_range_s: tuple[float, float]

    t1_s: float = 1.0
    refcon_angle_deg: float = 180.0
    epg_backend: str = "epgpy"  # epgpy|decaes

    threshold: float = 0.0

    reg: str = "gcv"  # none|lcurve|gcv|chi2|mdp
    chi2_factor: float | None = None
    noise_level: float | None = None

    min_ref_angle_deg: float = 50.0
    n_ref_angles: int = 64
    n_ref_angles_min: int | None = None
    set_flip_angle_deg: float | None = None

    save_residual_norm: bool = False
    save_decay_curve: bool = False
    save_reg_param: bool = False
    save_nnls_basis: bool = False

    def __post_init__(self) -> None:
        if int(self.n_te) < 4:
            raise ValueError("n_te must be >= 4")
        if float(self.te_s) <= 0:
            raise ValueError("te_s must be > 0")
        if int(self.n_t2) < 2:
            raise ValueError("n_t2 must be >= 2")
        lo, hi = self.t2_range_s
        if lo <= 0 or hi <= 0 or hi <= lo:
            raise ValueError("t2_range_s must be (lo, hi) with 0 < lo < hi")

        reg = self.reg.lower().strip()
        if reg not in {"none", "lcurve", "gcv", "chi2", "mdp"}:
            raise ValueError("reg must be one of: none, lcurve, gcv, chi2, mdp")

        backend = str(self.epg_backend).lower().strip()
        if backend not in {"epgpy", "decaes"}:
            raise ValueError("epg_backend must be 'epgpy' or 'decaes'")

    def echotimes_s(self) -> NDArray[np.float64]:
        import numpy as np

        return float(self.te_s) * np.arange(1, int(self.n_te) + 1, dtype=np.float64)

    def t2_times_s(self) -> NDArray[np.float64]:
        lo, hi = self.t2_range_s
        return _logspace_range(lo, hi, self.n_t2)

    def _flip_angles(self) -> NDArray[np.float64]:
        import numpy as np

        if self.set_flip_angle_deg is not None:
            return np.array([float(self.set_flip_angle_deg)], dtype=np.float64)
        return np.linspace(
            float(self.min_ref_angle_deg), 180.0, int(self.n_ref_angles), dtype=np.float64
        )

    def _optimize_alpha(self, b_norm: NDArray[np.float64]) -> tuple[float, NDArray[np.float64], Any, Any]:
        import numpy as np

        if self.set_flip_angle_deg is not None:
            alpha = float(self.set_flip_angle_deg)
            t2s = self.t2_times_s()
            A = _basis_matrix(
                n_te=self.n_te,
                te_s=self.te_s,
                t2_times_s=t2s,
                t1_s=self.t1_s,
                alpha_deg=alpha,
                refcon_angle_deg=self.refcon_angle_deg,
                epg_backend=self.epg_backend,
            )
            return alpha, A, float(alpha), A

        from qmrpy._decaes.surrogate_1d import NNLSDiscreteSurrogateSearch1D, surrogate_optimize_1d

        t2s = self.t2_times_s()
        grid = self._flip_angles()
        P = grid.size

        As = np.zeros((self.n_te, self.n_t2, P), dtype=np.float64)
        dAs = np.zeros((self.n_te, self.n_t2, P), dtype=np.float64)

        for k, a in enumerate(grid):
            As[:, :, k] = _basis_matrix(
                n_te=self.n_te,
                te_s=self.te_s,
                t2_times_s=t2s,
                t1_s=self.t1_s,
                alpha_deg=float(a),
                refcon_angle_deg=self.refcon_angle_deg,
                epg_backend=self.epg_backend,
            )
            dAs[:, :, k] = _basis_matrix_dalpha_fd(
                n_te=self.n_te,
                te_s=self.te_s,
                t2_times_s=t2s,
                t1_s=self.t1_s,
                alpha_deg=float(a),
                refcon_angle_deg=self.refcon_angle_deg,
                epg_backend=self.epg_backend,
                alpha_min_deg=float(self.min_ref_angle_deg),
            )

        mineval = int(self.n_ref_angles_min) if self.n_ref_angles_min is not None else min(5, int(self.n_ref_angles))
        mineval = max(2, min(mineval, int(self.n_ref_angles)))

        prob = NNLSDiscreteSurrogateSearch1D(As=As, dAs=dAs, grid=grid, b=np.asarray(b_norm, dtype=np.float64))
        alpha_opt, _ = surrogate_optimize_1d(prob, mineval=mineval, maxeval=int(self.n_ref_angles))

        A_opt = _basis_matrix(
            n_te=self.n_te,
            te_s=self.te_s,
            t2_times_s=t2s,
            t1_s=self.t1_s,
            alpha_deg=float(alpha_opt),
            refcon_angle_deg=self.refcon_angle_deg,
            epg_backend=self.epg_backend,
        )

        return float(alpha_opt), A_opt, grid, As

    def fit(self, signal: ArrayLike) -> dict[str, Any]:
        import numpy as np

        y = _as_1d_float_array(signal, name="signal")
        if y.shape != (self.n_te,):
            raise ValueError(f"signal must be shape ({self.n_te},), got {y.shape}")

        max_signal = float(np.max(y))
        b_norm = (y / max_signal).astype(np.float64) if max_signal > 0 else y.astype(np.float64)

        alpha_deg, A, refangleset, decaybasisset = self._optimize_alpha(b_norm)

        # Choose mu and solve
        mu = _choose_mu(
            A,
            b_norm,
            reg=self.reg,
            chi2_factor=self.chi2_factor,
            noise_level=self.noise_level,
        )

        from qmrpy._decaes.nnls import nnls_tikhonov

        sol = nnls_tikhonov(A, b_norm, float(mu))
        x_hat = sol.x * max_signal

        # Unregularized for chi2factor output
        sol0 = nnls_tikhonov(A, b_norm, 0.0)
        r0 = A @ sol0.x - b_norm
        r = A @ sol.x - b_norm
        chi2factor = float((r @ r) / max((r0 @ r0), np.finfo(float).eps))

        decay_curvefit = A @ x_hat
        residuals = decay_curvefit - y

        t2s = self.t2_times_s()
        logt2 = np.log(t2s)
        sumx = float(np.sum(x_hat))

        if sumx > 0:
            log_ggm = float(np.dot(x_hat, logt2) / sumx)
            log1p_gva = float(np.dot(x_hat, (logt2 - log_ggm) ** 2) / sumx)
            ggm = float(np.exp(log_ggm))
            gva = float(np.expm1(log1p_gva))
        else:
            ggm = float("nan")
            gva = float("nan")

        res2 = float(np.dot(residuals, residuals))
        sigma_res = float(np.std(residuals))
        fnr = float(sumx / np.sqrt(res2 / max(self.n_te - 1, 1))) if res2 > 0 else float("inf")
        snr = float(max_signal / sigma_res) if sigma_res > 0 else float("inf")

        out: dict[str, Any] = {
            "echotimes_s": self.echotimes_s(),
            "t2times_s": t2s,
            "refangleset": refangleset,
            "decaybasisset": decaybasisset,
            "alpha_deg": float(alpha_deg),
            "distribution": x_hat,
            "gdn": float(sumx),
            "ggm": float(ggm),
            "gva": float(gva),
            "fnr": float(fnr),
            "snr": float(snr),
        }

        if self.save_reg_param:
            out["mu"] = float(mu)
            out["chi2factor"] = float(chi2factor)
        if self.save_residual_norm:
            out["resnorm"] = float(np.linalg.norm(residuals))
        if self.save_decay_curve:
            out["decaycurve"] = decay_curvefit
        if self.save_nnls_basis:
            out["decaybasis"] = A

        return out

    def fit_image(
        self,
        image: ArrayLike,
        *,
        mask: ArrayLike | None = None,
        alpha_map_deg: ArrayLike | None = None,
    ) -> tuple[dict[str, Any], NDArray[np.float64]]:
        import numpy as np

        img = np.asarray(image, dtype=np.float64)
        if img.ndim != 4 or img.shape[-1] != self.n_te:
            raise ValueError(f"image must be 4D with last dim n_te={self.n_te}, got {img.shape}")

        if mask is None:
            m = np.ones(img.shape[:3], dtype=bool)
        else:
            m = np.asarray(mask, dtype=bool)
            if m.shape != img.shape[:3]:
                raise ValueError(f"mask shape {m.shape} must match image spatial shape {img.shape[:3]}")

        if alpha_map_deg is not None:
            alpha_map = np.asarray(alpha_map_deg, dtype=np.float64)
            if alpha_map.shape != img.shape[:3]:
                raise ValueError("alpha_map_deg must match image spatial shape")
        else:
            alpha_map = None

        t2s = self.t2_times_s()
        echotimes = self.echotimes_s()

        shape3 = img.shape[:3]
        gdn = np.full(shape3, np.nan, dtype=np.float64)
        ggm = np.full(shape3, np.nan, dtype=np.float64)
        gva = np.full(shape3, np.nan, dtype=np.float64)
        fnr = np.full(shape3, np.nan, dtype=np.float64)
        snr = np.full(shape3, np.nan, dtype=np.float64)
        alpha = np.full(shape3, np.nan, dtype=np.float64)

        resnorm = np.full(shape3, np.nan, dtype=np.float64) if self.save_residual_norm else None
        mu_map = np.full(shape3, np.nan, dtype=np.float64) if self.save_reg_param else None
        chi2_map = np.full(shape3, np.nan, dtype=np.float64) if self.save_reg_param else None
        decaycurve = (
            np.full((*shape3, self.n_te), np.nan, dtype=np.float64) if self.save_decay_curve else None
        )
        decaybasis = (
            np.full((*shape3, self.n_te, self.n_t2), np.nan, dtype=np.float64)
            if (self.save_nnls_basis and self.set_flip_angle_deg is None)
            else None
        )

        dist = np.full((*shape3, self.n_t2), np.nan, dtype=np.float64)

        refangleset = self._flip_angles() if self.set_flip_angle_deg is None else float(self.set_flip_angle_deg)
        if self.set_flip_angle_deg is None:
            decaybasisset = np.zeros((self.n_te, self.n_t2, len(refangleset)), dtype=np.float64)
            for k, a in enumerate(refangleset):
                decaybasisset[:, :, k] = _basis_matrix(
                    n_te=self.n_te,
                    te_s=self.te_s,
                    t2_times_s=t2s,
                    t1_s=self.t1_s,
                    alpha_deg=float(a),
                    refcon_angle_deg=self.refcon_angle_deg,
                    epg_backend=self.epg_backend,
                )
        else:
            decaybasisset = _basis_matrix(
                n_te=self.n_te,
                te_s=self.te_s,
                t2_times_s=t2s,
                t1_s=self.t1_s,
                alpha_deg=float(refangleset),
                refcon_angle_deg=self.refcon_angle_deg,
                epg_backend=self.epg_backend,
            )

        from qmrpy._decaes.nnls import nnls_tikhonov

        for idx in np.ndindex(shape3):
            if not bool(m[idx]):
                continue
            if float(img[idx + (0,)]) <= float(self.threshold):
                continue

            y = img[idx]
            max_signal = float(np.max(y))
            b_norm = (y / max_signal).astype(np.float64) if max_signal > 0 else y.astype(np.float64)

            if alpha_map is not None:
                alpha_deg = float(alpha_map[idx])
                A = _basis_matrix(
                    n_te=self.n_te,
                    te_s=self.te_s,
                    t2_times_s=t2s,
                    t1_s=self.t1_s,
                    alpha_deg=float(alpha_deg),
                    refcon_angle_deg=self.refcon_angle_deg,
                    epg_backend=self.epg_backend,
                )
            else:
                alpha_deg, A, _refangleset_unused, _basisset_unused = self._optimize_alpha(b_norm)

            mu_i = _choose_mu(
                A,
                b_norm,
                reg=self.reg,
                chi2_factor=self.chi2_factor,
                noise_level=self.noise_level,
            )

            sol = nnls_tikhonov(A, b_norm, float(mu_i))
            x_hat = sol.x * max_signal

            sol0 = nnls_tikhonov(A, b_norm, 0.0)
            r0 = A @ sol0.x - b_norm
            r = A @ sol.x - b_norm
            chi2_i = float((r @ r) / max((r0 @ r0), np.finfo(float).eps))

            decay_fit = A @ x_hat
            resid = decay_fit - y

            sumx = float(np.sum(x_hat))
            logt2 = np.log(t2s)
            if sumx > 0:
                log_ggm = float(np.dot(x_hat, logt2) / sumx)
                log1p_gva = float(np.dot(x_hat, (logt2 - log_ggm) ** 2) / sumx)
                ggm[idx] = float(np.exp(log_ggm))
                gva[idx] = float(np.expm1(log1p_gva))
                gdn[idx] = float(sumx)

            res2 = float(np.dot(resid, resid))
            sigma_res = float(np.std(resid))
            fnr[idx] = float(sumx / np.sqrt(res2 / max(self.n_te - 1, 1))) if res2 > 0 else float("inf")
            snr[idx] = float(max_signal / sigma_res) if sigma_res > 0 else float("inf")
            alpha[idx] = float(alpha_deg)
            dist[idx] = x_hat

            if resnorm is not None:
                resnorm[idx] = float(np.linalg.norm(resid))
            if mu_map is not None and chi2_map is not None:
                mu_map[idx] = float(mu_i)
                chi2_map[idx] = float(chi2_i)
            if decaycurve is not None:
                decaycurve[idx] = decay_fit
            if decaybasis is not None:
                decaybasis[idx] = A

        maps: dict[str, Any] = {
            "echotimes": echotimes,
            "t2times": t2s,
            "refangleset": refangleset,
            "decaybasisset": decaybasisset,
            "gdn": gdn,
            "ggm": ggm,
            "gva": gva,
            "fnr": fnr,
            "snr": snr,
            "alpha": alpha,
        }

        if resnorm is not None:
            maps["resnorm"] = resnorm
        if decaycurve is not None:
            maps["decaycurve"] = decaycurve
        if mu_map is not None and chi2_map is not None:
            maps["mu"] = mu_map
            maps["chi2factor"] = chi2_map
        if decaybasis is not None:
            maps["decaybasis"] = decaybasis

        return maps, dist
