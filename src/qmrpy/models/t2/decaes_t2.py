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
    te_ms: float,
    t2_ms: float,
    t1_ms: float,
    beta_deg: float,
) -> NDArray[np.float64]:
    """Compute normalized MSE echo decay curve using EPG w/ stimulated echo correction.

    Port of `DECAES.jl/src/EPGdecaycurve.jl:epg_decay_curve!(::EPGWork_Basic_Cplx, ::EPGOptions)`.
    """

    import numpy as np

    etl = int(etl)
    if etl < 1:
        raise ValueError("etl must be >= 1")
    te_ms = float(te_ms)
    if te_ms <= 0:
        raise ValueError("te_ms must be > 0")
    t2_ms = float(t2_ms)
    t1_ms = float(t1_ms)
    if t2_ms <= 0:
        raise ValueError("t2_ms must be > 0")
    if t1_ms <= 0:
        raise ValueError("t1_ms must be > 0")

    A = float(alpha_deg) / 180.0
    alpha_ex = A * 90.0
    alpha1 = A * 180.0
    alphai = A * float(beta_deg)

    # Relaxation for TE/2
    E1 = float(np.exp(-((te_ms / 2.0) / t1_ms)))
    E2 = float(np.exp(-((te_ms / 2.0) / t2_ms)))
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


def epg_decay_curve(
    *,
    etl: int,
    alpha_deg: float,
    te_ms: float,
    t2_ms: float,
    t1_ms: float,
    beta_deg: float,
    backend: str = "decaes",
) -> NDArray[np.float64]:
    """Compute normalized MSE echo decay curve using DECAES backend.

    Parameters
    ----------
    etl : int
        Echo train length.
    alpha_deg : float
        Refocusing flip angle in degrees.
    te_ms : float
        Echo spacing in milliseconds.
    t2_ms : float
        T2 in milliseconds.
    t1_ms : float
        T1 in milliseconds.
    beta_deg : float
        Refocusing phase angle in degrees.
    backend : {"decaes"}, optional
        Backend implementation.

    Returns
    -------
    ndarray
        Normalized decay curve of length ``etl``.
    """
    backend_norm = str(backend).lower().strip()
    if backend_norm == "decaes":
        return _epg_decay_curve_decaes(
            etl=etl,
            alpha_deg=alpha_deg,
            te_ms=te_ms,
            t2_ms=t2_ms,
            t1_ms=t1_ms,
            beta_deg=beta_deg,
        )
    raise ValueError("backend must be 'decaes'")


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
    te_ms: float,
    t2_times_ms,
    t1_ms: float,
    alpha_deg: float,
    refcon_angle_deg: float,
    epg_backend: str,
) -> NDArray[np.float64]:
    import numpy as np

    A = np.zeros((n_te, len(t2_times_ms)), dtype=np.float64)
    for j, t2 in enumerate(t2_times_ms):
        A[:, j] = epg_decay_curve(
            etl=n_te,
            alpha_deg=float(alpha_deg),
            te_ms=float(te_ms),
            t2_ms=float(t2),
            t1_ms=float(t1_ms),
            beta_deg=float(refcon_angle_deg),
            backend=str(epg_backend),
        )
    return A


def _basis_matrix_dalpha_fd(
    *,
    n_te: int,
    te_ms: float,
    t2_times_ms,
    t1_ms: float,
    alpha_deg: float,
    refcon_angle_deg: float,
    epg_backend: str,
    alpha_min_deg: float,
    h_deg: float = 1e-3,
):
    """Finite-difference derivative dA/dalpha matching DECAES' ∇A intent."""


    a = float(alpha_deg)
    h = float(h_deg)
    if a - h < float(alpha_min_deg):
        A0 = _basis_matrix(
            n_te=n_te,
            te_ms=te_ms,
            t2_times_ms=t2_times_ms,
            t1_ms=t1_ms,
            alpha_deg=a,
            refcon_angle_deg=refcon_angle_deg,
            epg_backend=epg_backend,
        )
        A1 = _basis_matrix(
            n_te=n_te,
            te_ms=te_ms,
            t2_times_ms=t2_times_ms,
            t1_ms=t1_ms,
            alpha_deg=a + h,
            refcon_angle_deg=refcon_angle_deg,
            epg_backend=epg_backend,
        )
        return (A1 - A0) / h
    if a + h > 180.0:
        A0 = _basis_matrix(
            n_te=n_te,
            te_ms=te_ms,
            t2_times_ms=t2_times_ms,
            t1_ms=t1_ms,
            alpha_deg=a - h,
            refcon_angle_deg=refcon_angle_deg,
            epg_backend=epg_backend,
        )
        A1 = _basis_matrix(
            n_te=n_te,
            te_ms=te_ms,
            t2_times_ms=t2_times_ms,
            t1_ms=t1_ms,
            alpha_deg=a,
            refcon_angle_deg=refcon_angle_deg,
            epg_backend=epg_backend,
        )
        return (A1 - A0) / h

    A_plus = _basis_matrix(
        n_te=n_te,
        te_ms=te_ms,
        t2_times_ms=t2_times_ms,
        t1_ms=t1_ms,
        alpha_deg=a + h,
        refcon_angle_deg=refcon_angle_deg,
        epg_backend=epg_backend,
    )
    A_minus = _basis_matrix(
        n_te=n_te,
        te_ms=te_ms,
        t2_times_ms=t2_times_ms,
        t1_ms=t1_ms,
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
    from scipy.optimize import minimize_scalar

    from qmrpy._decaes.nnls import nnls_tikhonov

    reg = reg.lower().strip()

    if reg == "none":
        return 0.0

    # Unregularized baseline
    r0 = nnls_tikhonov(A, b, 0.0)
    rvec0 = A @ r0.x - b
    res0_sq = float(rvec0 @ rvec0)

    lo, hi = -8.0, 2.0  # DECAES bounds (logmu)

    def bracket_root_monotonic(
        f, a: float, delta: float, *, dilate: float = 1.0, mono: int = +1, maxiters: int = 100
    ) -> tuple[float, float, float, float]:
        # Port of DECAES.optimization.bracket_root_monotonic
        if delta <= 0:
            raise ValueError("Initial step size must be positive")
        if dilate < 1:
            raise ValueError("Dilation factor must be at least 1")
        if mono == 0:
            raise ValueError("Monotonicity must be non-zero")

        fa = float(f(a))
        if not np.isfinite(fa):
            return float(a), float(a), float("nan"), float("nan")
        if fa == 0.0:
            return float(a), float(a), float(fa), float(fa)

        sgn_delta = float(np.sign(float(mono)) * np.sign(fa))
        b_ = float(a - sgn_delta * float(delta))
        fb = float(f(b_))
        if not np.isfinite(fb):
            return float(a), float(a), float(fa), float(fa)
        if fb == 0.0:
            return float(b_), float(b_), float(fb), float(fb)

        delta = float(delta) * float(dilate)
        cnt = 0
        while fa * fb > 0 and cnt < int(maxiters):
            a, fa = b_, fb
            b_ = float(a - sgn_delta * delta)
            fb = float(f(b_))
            if not np.isfinite(fb):
                return float(a), float(a), float(fa), float(fa)
            if fb == 0.0:
                return float(b_), float(b_), float(fb), float(fb)
            delta *= float(dilate)
            cnt += 1

        return (float(a), float(b_), float(fa), float(fb)) if a < b_ else (float(b_), float(a), float(fb), float(fa))

    def brent_root(
        f,
        x0: float,
        x1: float,
        fx0: float,
        fx1: float,
        *,
        xatol: float = 0.0,
        xrtol: float = 0.0,
        ftol: float = 0.0,
        maxiters: int = 100,
    ) -> tuple[float, float]:
        # Port of DECAES.optimization.brent_root
        if fx0 == 0.0:
            return float(x0), float(fx0)
        if fx1 == 0.0:
            return float(x1), float(fx1)
        if fx0 * fx1 >= 0.0:
            raise ValueError("Root must be bracketed")

        a, b_, fa, fb = float(x0), float(x1), float(fx0), float(fx1)
        if abs(fa) < abs(fb):
            a, b_, fa, fb = b_, a, fb, fa
        c, d, fc, mflag = float(x0), float(x0), float(fx0), True

        def secant_step(a_: float, b__: float, fa_: float, fb_: float) -> float:
            den = (fb_ - fa_)
            if den == 0.0:
                return float("nan")
            return a_ - fa_ * (b__ - a_) / den

        def inverse_quadratic_step(
            a_: float, b__: float, c_: float, fa_: float, fb_: float, fc_: float
        ) -> float:
            try:
                s_ = 0.0
                s_ += a_ * fb_ * fc_ / (fa_ - fb_) / (fa_ - fc_)
                s_ += b__ * fa_ * fc_ / (fb_ - fa_) / (fb_ - fc_)
                s_ += c_ * fa_ * fb_ / (fc_ - fa_) / (fc_ - fb_)
                return s_
            except ZeroDivisionError:
                return float("nan")

        for _ in range(int(maxiters)):
            if abs(b_ - a) <= 2.0 * (float(xatol) + float(xrtol) * abs(b_)):
                return float(b_), float(fb)

            s = inverse_quadratic_step(a, b_, c, fa, fb, fc)
            if not np.isfinite(s):
                s = secant_step(a, b_, fa, fb)

            u, v = (3.0 * a + b_) / 4.0, b_
            if u > v:
                u, v = v, u

            tol = max(float(xatol), float(xrtol) * max(abs(b_), abs(c), abs(d)))
            if (
                not (u < s < v)
                or (mflag and abs(s - b_) >= abs(b_ - c) / 2.0)
                or ((not mflag) and abs(s - b_) >= abs(b_ - c) / 2.0)
                or (mflag and abs(b_ - c) <= tol)
                or ((not mflag) and abs(c - d) <= tol)
            ):
                s = (a + b_) / 2.0
                mflag = True
            else:
                mflag = False

            fs = float(f(s))
            if fs == 0.0:
                return float(s), float(fs)
            if not np.isfinite(fs):
                return float(b_), float(fb)
            if abs(fs) <= float(ftol):
                return float(s), float(fs)

            c, fc, d = b_, fb, c
            if np.sign(fa) * np.sign(fs) < 0:
                b_, fb = s, fs
            else:
                a, fa = s, fs

            if abs(fa) < abs(fb):
                a, b_, fa, fb = b_, a, fb, fa

        return float(b_), float(fb)

    if reg == "gcv":
        s = np.linalg.svd(A, compute_uv=False)

        def obj(logmu: float) -> float:
            mu = float(np.exp(logmu))
            return float(np.log(max(_gcv_objective(A, b, s, mu), np.finfo(float).tiny)))

        res = minimize_scalar(obj, bounds=(lo, hi), method="bounded", options={"xatol": 1e-4})
        return float(np.exp(float(res.x)))

    if reg == "chi2":
        if chi2_factor is None or chi2_factor <= 1.0:
            raise ValueError("chi2_factor must be > 1.0 when reg='chi2'")

        res_target = float(chi2_factor) * res0_sq

        def f(logmu: float) -> float:
            mu = float(np.exp(logmu))
            r = nnls_tikhonov(A, b, mu)
            rv = A @ r.x - b
            res_sq = float(rv @ rv)
            return float((res_sq - res_target) / res_target)

        a, b_, fa, fb = bracket_root_monotonic(f, -4.0, 1.0, dilate=1.5, mono=+1, maxiters=6)
        if fa * fb < 0:
            logmu, _ = brent_root(
                f,
                a,
                b_,
                fa,
                fb,
                xatol=0.0,
                xrtol=0.0,
                ftol=float(1e-3) * (float(chi2_factor) - 1.0),
                maxiters=100,
            )
            return float(np.exp(logmu))
        return float(np.exp(a if abs(fa) < abs(fb) else b_))

    if reg == "mdp":
        if noise_level is None or noise_level <= 0.0:
            raise ValueError("noise_level must be > 0 when reg='mdp'")

        delta = float(np.sqrt(A.shape[0]) * float(noise_level))
        if delta <= float(np.sqrt(res0_sq)):
            return 0.0

        bnorm_sq = float(b @ b)
        if delta * delta >= bnorm_sq:
            return float("inf")

        target = delta * delta

        def f(logmu: float) -> float:
            mu = float(np.exp(logmu))
            r = nnls_tikhonov(A, b, mu)
            rv = A @ r.x - b
            return float((rv @ rv) - target)

        a, b_, fa, fb = bracket_root_monotonic(f, -4.0, 1.0, dilate=1.5, mono=+1, maxiters=6)
        if fa * fb < 0:
            logmu, err = brent_root(
                f,
                a,
                b_,
                fa,
                fb,
                xatol=0.0,
                xrtol=0.0,
                ftol=float(1e-3) * target,
                maxiters=100,
            )
            if np.isfinite(err):
                return float(np.exp(logmu))
            return 0.0
        return float(np.exp(a if abs(fa) < abs(fb) else b_))

    if reg == "lcurve":
        # Port of DECAES.jl `lcurve_corner` (lsqnonneg.jl). Keep a straightforward cache keyed by
        # exact float values; this matches the deterministic evaluation points used by the method.
        phi = float((1 + 5**0.5) / 2.0)
        xtol = 1e-4
        ptol = 1e-4
        ctol = 1e-4

        point_cache: dict[float, tuple[np.ndarray, float]] = {}  # logmu -> (P, C)
        state_cache: list[
            tuple[tuple[float, float, float, float], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
        ] = []

        def f_lcurve(logmu: float) -> np.ndarray:
            mu = float(np.exp(float(logmu)))
            r = nnls_tikhonov(A, b, mu)
            rv = A @ r.x - b
            xi = float(np.log(float(rv @ rv)))
            eta = float(np.log(float(r.x @ r.x)))
            return np.array([xi, eta], dtype=np.float64)

        def menger(pj: np.ndarray, pk: np.ndarray, pl: np.ndarray) -> float:
            d_jk = pj - pk
            d_kl = pk - pl
            d_lj = pl - pj
            pjk = float(d_jk @ d_jk)
            pkl = float(d_kl @ d_kl)
            plj = float(d_lj @ d_lj)
            denom = float(np.sqrt(max(pjk * pkl * plj, 0.0)))
            if denom == 0.0:
                return float("-inf")
            cross = float(d_jk[0] * d_kl[1] - d_jk[1] * d_kl[0])
            return float(2.0 * cross / denom)

        def _get_p(x: float) -> np.ndarray:
            x = float(x)
            if x not in point_cache:
                point_cache[x] = (f_lcurve(x), float("-inf"))
            return point_cache[x][0]

        def update_curvature(
            xvec: tuple[float, float, float, float], ptopleft: np.ndarray, pbottomright: np.ndarray
        ) -> None:
            def pfilter(p: np.ndarray) -> bool:
                return (
                    min(float(np.linalg.norm(p - ptopleft)), float(np.linalg.norm(p - pbottomright))) > ctol
                )

            for x in xvec:
                _get_p(x)

            xs = sorted(point_cache.keys())
            for x in xvec:
                p, _ = point_cache[x]
                c = float("-inf")
                if pfilter(p):
                    x_m = max((t for t in xs if t < x), default=None)
                    x_p = min((t for t in xs if t > x), default=None)
                    if x_m is not None and x_p is not None:
                        pm = point_cache[float(x_m)][0]
                        pp = point_cache[float(x_p)][0]
                        c = menger(pm, p, pp)
                point_cache[x] = (p, float(c))

        def initial_state(x1: float, x4: float):
            x2 = (phi * x1 + x4) / (phi + 1.0)
            x3 = x1 + (x4 - x2)
            xvec = (float(x1), float(x2), float(x3), float(x4))
            pvec = (_get_p(xvec[0]), _get_p(xvec[1]), _get_p(xvec[2]), _get_p(xvec[3]))
            return xvec, pvec

        def move_left(xvec, pvec):
            x1, x2, x3, _x4 = xvec
            x2_new = (phi * x1 + x3) / (phi + 1.0)
            xnew = (float(x1), float(x2_new), float(x2), float(x3))
            pnew = (pvec[0], _get_p(xnew[1]), pvec[1], pvec[2])
            return xnew, pnew

        def move_right(xvec, pvec):
            _x1, x2, x3, x4 = xvec
            x3_new = x2 + (x4 - x3)
            xnew = (float(x2), float(x3), float(x3_new), float(x4))
            pnew = (pvec[1], pvec[2], _get_p(xnew[2]), pvec[3])
            return xnew, pnew

        xvec, pvec = initial_state(float(lo), float(hi))
        ptopleft, pbottomright = pvec[0], pvec[3]
        update_curvature(xvec, ptopleft, pbottomright)

        it = 0
        while abs(xvec[3] - xvec[0]) >= xtol and float(np.linalg.norm(pvec[0] - pvec[3])) >= ptol:
            it += 1

            # backtracking
            if point_cache and state_cache:
                best_x = max(point_cache.items(), key=lambda kv: kv[1][1])[0]
                best_diam = abs(xvec[3] - xvec[0])
                for sx, sp in state_cache:
                    if (sx[1] == best_x or sx[2] == best_x) and abs(sx[3] - sx[0]) <= best_diam:
                        xvec, pvec = sx, sp
                        best_diam = abs(sx[3] - sx[0])

            c2 = point_cache.get(xvec[1], (None, float("-inf")))[1]
            c3 = point_cache.get(xvec[2], (None, float("-inf")))[1]
            if c2 > c3:
                xvec, pvec = move_left(xvec, pvec)
            else:
                xvec, pvec = move_right(xvec, pvec)

            update_curvature(xvec, ptopleft, pbottomright)
            state_cache.append((xvec, pvec))
            if it > 200:
                break

        best_x = max(point_cache.items(), key=lambda kv: kv[1][1])[0]
        return float(np.exp(best_x))

    raise ValueError(f"Unknown reg: {reg}")


@dataclass(frozen=True, slots=True)
class DECAEST2Map:
    """DECAES-like multi-component T2 mapping (T2mapSEcorr + core outputs).

    Units:
        - te_ms, t2_range_ms, t1_ms: milliseconds
    """

    n_te: int
    te_ms: float
    n_t2: int
    t2_range_ms: tuple[float, float]
    reg: str  # none|lcurve|gcv|chi2|mdp

    t1_ms: float = 1000.0
    refcon_angle_deg: float = 180.0
    epg_backend: str = "decaes"

    threshold: float = 0.0

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
        if float(self.te_ms) <= 0:
            raise ValueError("te_ms must be > 0")
        if int(self.n_t2) < 2:
            raise ValueError("n_t2 must be >= 2")
        lo, hi = self.t2_range_ms
        if lo <= 0 or hi <= 0 or hi <= lo:
            raise ValueError("t2_range_ms must be (lo, hi) with 0 < lo < hi")

        reg = str(self.reg).lower().strip()
        if reg not in {"none", "lcurve", "gcv", "chi2", "mdp"}:
            raise ValueError("reg must be one of: none, lcurve, gcv, chi2, mdp")
        object.__setattr__(self, "reg", reg)

        if reg == "chi2":
            if self.chi2_factor is None or float(self.chi2_factor) <= 1.0:
                raise ValueError("chi2_factor must be > 1.0 when reg='chi2'")
        if reg == "mdp":
            if self.noise_level is None or float(self.noise_level) <= 0.0:
                raise ValueError("noise_level must be > 0 when reg='mdp'")

        backend = str(self.epg_backend).lower().strip()
        if backend != "decaes":
            raise ValueError("epg_backend must be 'decaes'")
        object.__setattr__(self, "epg_backend", backend)

        if self.n_ref_angles_min is None:
            object.__setattr__(self, "n_ref_angles_min", min(5, int(self.n_ref_angles)))

    def echotimes_ms(self) -> NDArray[np.float64]:
        import numpy as np

        return float(self.te_ms) * np.arange(1, int(self.n_te) + 1, dtype=np.float64)

    def t2_times_ms(self) -> NDArray[np.float64]:
        lo, hi = self.t2_range_ms
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
            t2s = self.t2_times_ms()
            A = _basis_matrix(
                n_te=self.n_te,
                te_ms=self.te_ms,
                t2_times_ms=t2s,
                t1_ms=self.t1_ms,
                alpha_deg=alpha,
                refcon_angle_deg=self.refcon_angle_deg,
                epg_backend=self.epg_backend,
            )
            return alpha, A, float(alpha), A

        from qmrpy._decaes.surrogate_1d import NNLSDiscreteSurrogateSearch1D, surrogate_optimize_1d

        t2s = self.t2_times_ms()
        grid = self._flip_angles()
        P = grid.size

        As = np.zeros((self.n_te, self.n_t2, P), dtype=np.float64)
        dAs = np.zeros((self.n_te, self.n_t2, P), dtype=np.float64)

        for k, a in enumerate(grid):
            As[:, :, k] = _basis_matrix(
                n_te=self.n_te,
                te_ms=self.te_ms,
                t2_times_ms=t2s,
                t1_ms=self.t1_ms,
                alpha_deg=float(a),
                refcon_angle_deg=self.refcon_angle_deg,
                epg_backend=self.epg_backend,
            )
            dAs[:, :, k] = _basis_matrix_dalpha_fd(
                n_te=self.n_te,
                te_ms=self.te_ms,
                t2_times_ms=t2s,
                t1_ms=self.t1_ms,
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
            te_ms=self.te_ms,
            t2_times_ms=t2s,
            t1_ms=self.t1_ms,
            alpha_deg=float(alpha_opt),
            refcon_angle_deg=self.refcon_angle_deg,
            epg_backend=self.epg_backend,
        )

        return float(alpha_opt), A_opt, grid, As

    def fit(self, signal: ArrayLike) -> dict[str, Any]:
        """Fit T2 distribution for a single voxel.

        Parameters
        ----------
        signal : array-like
            Observed signal array of length ``n_te``.

        Returns
        -------
        dict
            Fit results including ``distribution`` and diagnostics.
        """
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

        t2s = self.t2_times_ms()
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
            "echotimes_ms": self.echotimes_ms(),
            "t2times_ms": t2s,
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
        signal: ArrayLike,
        *,
        mask: ArrayLike | str | None = None,
        alpha_map_deg: ArrayLike | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
    ) -> tuple[dict[str, Any], NDArray[np.float64]]:
        """Fit T2 distribution voxel-wise for a 4D image.

        Parameters
        ----------
        signal : array-like
            Input image with shape ``(x, y, z, n_te)``.
        mask : array-like, optional
            Spatial mask. If "otsu", Otsu thresholding is applied.
        alpha_map_deg : array-like, optional
            Precomputed flip angle map in degrees.
        n_jobs : int, default=1
            Number of parallel jobs. -1 uses all CPUs.
        verbose : bool, default=False
            If True, show progress bar and log info.

        Returns
        -------
        tuple
            (maps, distributions) where ``maps`` is a dict of parameter maps
            and ``distributions`` is a 4D array with last dim ``n_t2``.
        """
        import logging

        import numpy as np

        from qmrpy._mask import resolve_mask

        logger = logging.getLogger("qmrpy")

        img = np.asarray(signal, dtype=np.float64)
        if img.ndim != 4 or img.shape[-1] != self.n_te:
            raise ValueError(f"image must be 4D with last dim n_te={self.n_te}, got {img.shape}")

        spatial_shape = img.shape[:3]
        resolved_mask = resolve_mask(mask, img)
        if resolved_mask is None:
            m = np.ones(spatial_shape, dtype=bool)
        else:
            if resolved_mask.shape != spatial_shape:
                raise ValueError(f"mask shape {resolved_mask.shape} must match image spatial shape {spatial_shape}")
            m = resolved_mask

        if alpha_map_deg is not None:
            alpha_map = np.asarray(alpha_map_deg, dtype=np.float64)
            if alpha_map.shape != img.shape[:3]:
                raise ValueError("alpha_map_deg must match image spatial shape")
        else:
            alpha_map = None

        t2s = self.t2_times_ms()
        echotimes = self.echotimes_ms()

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
                    te_ms=self.te_ms,
                    t2_times_ms=t2s,
                    t1_ms=self.t1_ms,
                    alpha_deg=float(a),
                    refcon_angle_deg=self.refcon_angle_deg,
                    epg_backend=self.epg_backend,
                )
        else:
            decaybasisset = _basis_matrix(
                n_te=self.n_te,
                te_ms=self.te_ms,
                t2_times_ms=t2s,
                t1_ms=self.t1_ms,
                alpha_deg=float(refangleset),
                refcon_angle_deg=self.refcon_angle_deg,
                epg_backend=self.epg_backend,
            )

        from qmrpy._decaes.nnls import nnls_tikhonov

        # Helper function for processing single voxel
        def process_voxel(idx: tuple[int, ...]) -> tuple[tuple[int, ...], dict[str, Any]]:
            if not bool(m[idx]):
                return idx, {}
            if float(img[idx + (0,)]) <= float(self.threshold):
                return idx, {}

            y = img[idx]
            max_signal = float(np.max(y))
            b_norm = (y / max_signal).astype(np.float64) if max_signal > 0 else y.astype(np.float64)

            if alpha_map is not None:
                alpha_deg = float(alpha_map[idx])
                A = _basis_matrix(
                    n_te=self.n_te,
                    te_ms=self.te_ms,
                    t2_times_ms=t2s,
                    t1_ms=self.t1_ms,
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
            ggm_val = float("nan")
            gva_val = float("nan")
            if sumx > 0:
                log_ggm = float(np.dot(x_hat, logt2) / sumx)
                log1p_gva = float(np.dot(x_hat, (logt2 - log_ggm) ** 2) / sumx)
                ggm_val = float(np.exp(log_ggm))
                gva_val = float(np.expm1(log1p_gva))

            res2 = float(np.dot(resid, resid))
            sigma_res = float(np.std(resid))
            fnr_val = float(sumx / np.sqrt(res2 / max(self.n_te - 1, 1))) if res2 > 0 else float("inf")
            snr_val = float(max_signal / sigma_res) if sigma_res > 0 else float("inf")

            result = {
                "gdn": float(sumx),
                "ggm": ggm_val,
                "gva": gva_val,
                "fnr": fnr_val,
                "snr": snr_val,
                "alpha": float(alpha_deg),
                "x_hat": x_hat,
            }
            if self.save_residual_norm:
                result["resnorm"] = float(np.linalg.norm(resid))
            if self.save_reg_param:
                result["mu"] = float(mu_i)
                result["chi2"] = float(chi2_i)
            if self.save_decay_curve:
                result["decay_fit"] = decay_fit
            if self.save_nnls_basis and self.set_flip_angle_deg is None:
                result["basis"] = A

            return idx, result

        # Collect indices to process
        indices = list(np.ndindex(shape3))
        n_voxels = int(np.sum(m))

        if verbose:
            logger.info("DECAEST2Map: %d voxels, n_jobs=%s, shape=%s", n_voxels, n_jobs, shape3)

        if n_jobs == 1:
            # Serial execution
            iterator = indices
            if verbose:
                from tqdm import tqdm
                iterator = tqdm(indices, desc="DECAEST2Map", unit="voxel")

            for idx in iterator:
                _idx, res = process_voxel(idx)
                if not res:
                    continue
                gdn[_idx] = float(res["gdn"])
                ggm[_idx] = float(res["ggm"])
                gva[_idx] = float(res["gva"])
                fnr[_idx] = float(res["fnr"])
                snr[_idx] = float(res["snr"])
                alpha[_idx] = float(res["alpha"])
                dist[_idx] = res["x_hat"]
                if resnorm is not None:
                    resnorm[_idx] = float(res["resnorm"])
                if mu_map is not None and chi2_map is not None:
                    mu_map[_idx] = float(res["mu"])
                    chi2_map[_idx] = float(res["chi2"])
                if decaycurve is not None:
                    decaycurve[_idx] = res["decay_fit"]
                if decaybasis is not None:
                    decaybasis[_idx] = res["basis"]
        else:
            # Parallel execution
            from joblib import Parallel, delayed

            if verbose:
                from tqdm import tqdm
                results = Parallel(n_jobs=n_jobs)(
                    delayed(process_voxel)(idx)
                    for idx in tqdm(indices, desc="DECAEST2Map", unit="voxel")
                )
            else:
                results = Parallel(n_jobs=n_jobs)(
                    delayed(process_voxel)(idx) for idx in indices
                )

            for _idx, res in results:
                if not res:
                    continue
                gdn[_idx] = float(res["gdn"])
                ggm[_idx] = float(res["ggm"])
                gva[_idx] = float(res["gva"])
                fnr[_idx] = float(res["fnr"])
                snr[_idx] = float(res["snr"])
                alpha[_idx] = float(res["alpha"])
                dist[_idx] = res["x_hat"]
                if resnorm is not None:
                    resnorm[_idx] = float(res["resnorm"])
                if mu_map is not None and chi2_map is not None:
                    mu_map[_idx] = float(res["mu"])
                    chi2_map[_idx] = float(res["chi2"])
                if decaycurve is not None:
                    decaycurve[_idx] = res["decay_fit"]
                if decaybasis is not None:
                    decaybasis[_idx] = res["basis"]

        if verbose:
            logger.info("DECAEST2Map complete: %d voxels processed", n_voxels)

        maps: dict[str, Any] = {
            "echotimes_ms": echotimes,
            "t2times_ms": t2s,
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
