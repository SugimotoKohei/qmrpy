from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .nnls import nnls_decaes


@dataclass(slots=True)
class CubicHermiteSegment:
    a: float
    b: float
    u0: float
    u1: float
    m0: float
    m1: float

    def minimize(self) -> tuple[float, float]:
        """Port of DECAES.jl CubicHermiteInterpolator.minimize (1D)."""

        u0, u1, m0, m1 = self.u0, self.u1, self.m0, self.m1
        a, b = self.a, self.b

        # scale slopes to canonical domain [-1,1]
        r = (b - a) / 2.0
        m0 = r * m0
        m1 = r * m1

        du = u1 - u0
        dm = m1 - m0
        su = u1 + u0
        sm = m1 + m0

        # coefficients for evalpoly(t): c0 + c1*t + c2*t^2 + c3*t^3
        c0 = su / 2.0 - dm / 4.0
        c1 = (3 * du - sm) / 4.0
        c2 = dm / 4.0
        c3 = (sm - du) / 4.0

        def eval_t(t: float) -> float:
            return ((c3 * t + c2) * t + c1) * t + c0

        # Endpoint minimum in x-space
        if u0 < u1:
            xend, uend = a, u0
        else:
            xend, uend = b, u1

        # Minimizer formula from DECAES
        delta = 3.0 * (u0 - u1)
        theta = delta / 2.0 + (m0 + m1)
        gamma = theta * theta - m0 * m1
        gamma = -np.sqrt(gamma) if gamma > 0 else 0.0
        p = -(delta + (m0 + m1))
        q = 2.0 * gamma + (m0 - m1)

        if abs(p) < abs(q) and q != 0:
            t = p / q
            y = eval_t(t)
            if y < uend:
                # to domain
                c = (a + b) / 2.0
                r = (b - a) / 2.0
                x = float(np.clip(r * t + c, a, b))
                return x, float(y)

        return xend, float(uend)


@dataclass(slots=True)
class NNLSDiscreteSurrogateSearch1D:
    As: np.ndarray  # (M,N,P)
    dAs: np.ndarray  # (M,N,P)
    grid: np.ndarray  # (P,)
    b: np.ndarray  # (M,)

    def loss(self, i: int) -> float:
        A = self.As[:, :, i]
        res = nnls_decaes(A, self.b, init_last_column=True)
        r = A @ res.x - self.b
        return float(np.dot(r, r))

    def loss_with_grad(self, i: int) -> tuple[float, float]:
        A = self.As[:, :, i]
        dA = self.dAs[:, :, i]
        res = nnls_decaes(A, self.b, init_last_column=True)
        x = res.x

        Axpb = (A @ x) - self.b
        dAxp = dA @ x
        u = float(np.dot(Axpb, Axpb))
        du = float(2.0 * np.dot(dAxp, Axpb))
        return u, du


@dataclass(slots=True)
class CubicHermiteSplineSurrogate1D:
    prob: NNLSDiscreteSurrogateSearch1D

    seen: np.ndarray
    u: np.ndarray
    du: np.ndarray
    idx: list[int]

    @classmethod
    def create(cls, prob: NNLSDiscreteSurrogateSearch1D) -> "CubicHermiteSplineSurrogate1D":
        P = prob.grid.size
        return cls(
            prob=prob,
            seen=np.zeros(P, dtype=bool),
            u=np.full(P, np.nan, dtype=np.float64),
            du=np.full(P, np.nan, dtype=np.float64),
            idx=[],
        )

    def empty(self) -> None:
        self.seen[:] = False
        self.u[:] = np.nan
        self.du[:] = np.nan
        self.idx.clear()

    def update(self, I: int) -> None:
        if self.seen[I]:
            return
        u, du = self.prob.loss_with_grad(I)
        self.seen[I] = True
        self.u[I] = u
        self.du[I] = du
        # keep idx sorted
        self.idx.append(I)
        self.idx.sort()

    def suggest_point(self) -> tuple[float, float]:
        assert len(self.idx) >= 1
        p_best = self.prob.grid[self.idx[0]]
        u_best = float(self.u[self.idx[0]])

        plast = float(self.prob.grid[self.idx[0]])
        ulast = float(self.u[self.idx[0]])
        dlast = float(self.du[self.idx[0]])

        for k in self.idx[1:]:
            pcurr = float(self.prob.grid[k])
            ucurr = float(self.u[k])
            dcurr = float(self.du[k])
            seg = CubicHermiteSegment(plast, pcurr, ulast, ucurr, dlast, dcurr)
            x, u = seg.minimize()
            if u < u_best:
                p_best, u_best = x, u
            plast, ulast, dlast = pcurr, ucurr, dcurr

        return float(p_best), float(u_best)


@dataclass(slots=True)
class DiscreteSurrogateSearcher1D:
    P: int
    seen: np.ndarray
    numeval: int

    @classmethod
    def create(cls, P: int) -> "DiscreteSurrogateSearcher1D":
        return cls(P=P, seen=np.zeros(P, dtype=bool), numeval=0)


def _evaluate_box(surr: CubicHermiteSplineSurrogate1D, state: DiscreteSurrogateSearcher1D, lo: int, hi: int, *, maxeval: int):
    # Evaluate corners; sorted by distance to suggested point is omitted here since 1D endpoints only.
    for I in (lo, hi):
        if state.numeval >= maxeval:
            break
        if state.seen[I]:
            continue
        surr.update(I)
        state.seen[I] = True
        state.numeval += 1


def surrogate_optimize_1d(
    prob: NNLSDiscreteSurrogateSearch1D,
    *,
    mineval: int,
    maxeval: int,
) -> tuple[float, float]:
    """Port of DECAES.jl surrogate_spline_opt for 1D cubic hermite surrogate."""

    P = prob.grid.size
    surr = CubicHermiteSplineSurrogate1D.create(prob)
    state = DiscreteSurrogateSearcher1D.create(P)

    # initialize by recursive bisection (breadth-first)
    def init(lo: int, hi: int, depth: int):
        if depth <= 0:
            return
        _evaluate_box(surr, state, lo, hi, maxeval=maxeval)
        if state.numeval >= mineval:
            return
        if hi - lo <= 1:
            return
        mid = (lo + hi) // 2
        init(lo, mid, depth - 1)
        init(mid, hi, depth - 1)

    for depth in range(1, mineval + 1):
        init(0, P - 1, depth)
        if state.numeval >= mineval:
            break

    x, u = surr.suggest_point()

    # bisection search loop
    while True:
        # minimal bounding box containing x: find nearest interval in index space
        # For 1D: choose smallest [lo,hi] that contains x and is not fully evaluated.
        # We'll locate enclosing indices.
        grid = prob.grid
        if x <= grid[0]:
            lo, hi = 0, 1
        elif x >= grid[-1]:
            lo, hi = P - 2, P - 1
        else:
            hi = int(np.searchsorted(grid, x, side="right"))
            lo = hi - 1

        # expand to unevaluated interval if both corners already evaluated and can split
        while (state.seen[lo] and state.seen[hi]) and (hi - lo > 1):
            mid = (lo + hi) // 2
            if x <= grid[mid]:
                hi = mid
            else:
                lo = mid

        _evaluate_box(surr, state, lo, hi, maxeval=maxeval)
        x, u = surr.suggest_point()

        if state.numeval >= maxeval or (hi - lo) <= 1:
            return x, u
