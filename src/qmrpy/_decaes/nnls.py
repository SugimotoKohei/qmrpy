from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class NnlsResult:
    x: np.ndarray
    rnorm: float


@dataclass(slots=True)
class _NnlsWorkspace:
    # Mirrors DECAES.jl NNLSWorkspace (subset)
    A: np.ndarray  # (m,n) modified in-place
    b: np.ndarray  # (m,) modified in-place
    x: np.ndarray  # (n,)
    w: np.ndarray  # (n,)
    zz: np.ndarray  # (m,) (also used as temp)
    idx: np.ndarray  # (n,) permutation (0-based)
    invidx: np.ndarray  # (n,)
    diag: np.ndarray  # (n,) only used by lambda variant

    mode: int = 0
    nsetp: int = 0
    rnorm: float = 0.0


def _largest_positive_dual(w: np.ndarray, j1: int) -> tuple[float, int]:
    # DECAES: largest_positive_dual(w, nsetp+1)
    wmax = 0.0
    jmax = -1
    for j in range(j1, w.size):
        if w[j] > wmax:
            wmax = float(w[j])
            jmax = int(j)
    return wmax, jmax


def _compute_dual(w: np.ndarray, A: np.ndarray, b: np.ndarray, j1: int, m1: int) -> None:
    # w[j] = sum_i A[i,j]*b[i] for i=j1..m1-1, j=j1..n-1
    m, n = A.shape
    for j in range(j1, n):
        sm = 0.0
        for i in range(j1, m1):
            sm += A[i, j] * b[i]
        w[j] = sm


def _orthogonal_rotmat(a: float, b: float) -> tuple[float, float, float]:
    sigma = float(np.hypot(a, b))
    c = a / sigma
    s = b / sigma
    return c, s, sigma


def _orthogonal_rotmatvec(c: float, s: float, a: float, b: float) -> tuple[float, float]:
    x = c * a + s * b
    y = -s * a + c * b
    return x, y


def _solve_upper_triangular(z: np.ndarray, A: np.ndarray, n: int) -> None:
    # In-place solve Ux=b where U = A[:n,:n] upper-triangular, b=z[:n]
    for j in range(n - 1, -1, -1):
        zi = -z[j] / A[j, j]
        for i in range(0, j):
            z[i] = z[i] + A[i, j] * zi
        z[j] = -zi


def _construct_apply_householder(A: np.ndarray, b: np.ndarray, ip: int, jp: int, m: int) -> float:
    # Port of DECAES construct_apply_householder!(A,b,ip,jp,m)
    # Uses 0-based indices; ip, jp are 0-based.
    if ip >= m:
        return 0.0

    alpha = float(A[ip, jp])
    xnorm = float(np.sqrt(np.sum(A[ip:m, jp] ** 2)))
    if xnorm == 0.0:
        return -1.0

    beta = float(np.copysign(xnorm, alpha))
    alpha2 = alpha + beta
    tau = alpha2 / beta

    sm = float(b[ip])
    if ip + 1 < m:
        for i in range(ip + 1, m):
            sm += b[i] * (A[i, jp] / alpha2)
    sm *= -tau

    A1 = -beta
    b1 = float(b[ip] + sm)

    if A1 != 0.0 and (b1 / A1) > 0.0:
        # good column, update b and perform swap/householder application
        if ip < m - 1:
            if ip != jp:
                # swap columns for rows < ip
                for i in range(0, ip):
                    A[i, ip], A[i, jp] = A[i, jp], A[i, ip]

                b[ip] = b1
                A[ip, ip], A[ip, jp] = A1, A[ip, ip]

                for i in range(ip + 1, m):
                    Aij = A[i, jp] / alpha2
                    A[i, ip], A[i, jp] = Aij, A[i, ip]
                    b[i] = b[i] + sm * Aij
            else:
                b[ip] = b1
                A[ip, ip] = A1
                for i in range(ip + 1, m):
                    Aii = A[i, ip] / alpha2
                    A[i, ip] = Aii
                    b[i] = b[i] + sm * Aii
        else:
            # ip == m-1
            tau = 0.0
            if ip != jp:
                for i in range(0, m):
                    A[i, ip], A[i, jp] = A[i, jp], A[i, ip]
        return float(tau)

    return -1.0


def _apply_householder_dual(A: np.ndarray, w: np.ndarray, b: np.ndarray, tau: float, j1: int, m1: int) -> None:
    # Simplified port of apply_householder_dual!(...) without loop unrolling.
    if j1 >= m1:
        return

    aii = float(A[j1, j1])
    A[j1, j1] = 1.0

    m, n = A.shape
    for j in range(j1 + 1, n):
        sm = 0.0
        for i in range(j1, m1):
            sm += A[i, j] * A[i, j1]
        sm *= -tau

        wj = 0.0
        A[j1, j] = A[j1, j] + sm
        for i in range(j1 + 1, m1):
            Aij = A[i, j] + sm * A[i, j1]
            wj += Aij * b[i]
            A[i, j] = Aij
        w[j] = wj

    A[j1, j1] = aii


def _unsafe_nnls(work: _NnlsWorkspace, *, init_dual: bool = True, max_iter: int | None = None) -> None:
    A = work.A
    b = work.b
    x = work.x
    w = work.w
    zz = work.zz
    idx = work.idx
    invidx = work.invidx

    m, n = A.shape
    if max_iter is None:
        max_iter = 3 * n

    if init_dual:
        w.fill(0.0)
        _compute_dual(w, A, b, 0, m)

    nsetp = 0
    it = 0
    work.mode = 0

    while True:
        if nsetp >= n or nsetp >= m:
            break

        # choose column to enter
        while True:
            wmax, jmax = _largest_positive_dual(w, nsetp)
            if wmax <= 0.0:
                jmax = -1
                break
            tau = _construct_apply_householder(A, b, nsetp, jmax, m)
            if tau >= 0.0:
                break
            w[jmax] = 0.0

        if jmax < 0:
            break

        # move column from Z to P
        nsetp += 1
        idx[nsetp - 1], idx[jmax] = idx[jmax], idx[nsetp - 1]

        if nsetp < n:
            _apply_householder_dual(A, w, b, tau, nsetp - 1, m)

        for i in range(nsetp, m):
            A[i, nsetp - 1] = 0.0
        w[nsetp - 1] = 0.0

        # solve triangular system
        for i in range(nsetp):
            zz[i] = b[i]
        _solve_upper_triangular(zz, A, nsetp)

        dual_flag = False
        while True:
            it += 1
            if it > max_iter:
                work.mode = 1
                break

            # feasibility check
            imv = nsetp
            alpha = 2.0
            for i in range(nsetp):
                if zz[i] <= 0.0:
                    xi = x[idx[i]]
                    t = -xi / (zz[i] - xi)
                    if alpha > t:
                        imv = i + 1
                        alpha = t

            if alpha == 2.0:
                break

            dual_flag = True

            for i in range(nsetp):
                ix = idx[i]
                x[ix] = x[ix] + alpha * (zz[i] - x[ix])

            while True:
                x[idx[imv - 1]] = 0.0

                if (imv - 1) != (nsetp - 1):
                    for i in range(imv, nsetp):
                        cc, ss, rr = _orthogonal_rotmat(A[i - 1, i], A[i, i])
                        A[i - 1, i] = rr
                        A[i, i] = 0.0

                        for j in range(0, i):
                            A[i - 1, j], A[i, j] = _orthogonal_rotmatvec(cc, ss, A[i - 1, j], A[i, j])
                        for j in range(i + 1, n):
                            A[i - 1, j], A[i, j] = _orthogonal_rotmatvec(cc, ss, A[i - 1, j], A[i, j])

                        b[i - 1], b[i] = _orthogonal_rotmatvec(cc, ss, b[i - 1], b[i])

                    # swap columns
                    for j in range(imv - 1, nsetp - 1):
                        for i in range(0, m):
                            A[i, j + 1], A[i, j] = A[i, j], A[i, j + 1]
                        idx[j], idx[j + 1] = idx[j + 1], idx[j]

                nsetp -= 1

                allfeasible = True
                for i in range(nsetp):
                    if x[idx[i]] <= 0.0:
                        allfeasible = False
                        imv = i + 1
                        break
                if allfeasible:
                    break

            for i in range(nsetp):
                zz[i] = b[i]
            _solve_upper_triangular(zz, A, nsetp)

        if work.mode == 1:
            break

        if dual_flag:
            _compute_dual(w, A, b, nsetp, m)

        for i in range(nsetp):
            x[idx[i]] = zz[i]

    # inverse perm
    for i in range(n):
        invidx[idx[i]] = i

    sm = 0.0
    if nsetp < m:
        for i in range(nsetp, m):
            bi = float(b[i])
            zz[i] = bi
            sm += bi * bi
    else:
        w.fill(0.0)

    work.rnorm = float(np.sqrt(sm))
    work.nsetp = int(nsetp)


def _unsafe_nnls_tikhonov(
    work: _NnlsWorkspace,
    lam: float,
    *,
    init_dual: bool = True,
    max_iter: int | None = None,
) -> int:
    # Port of DECAES.NNLS.unsafe_nnls!(work, λ)
    A = work.A
    b = work.b
    x = work.x
    w = work.w
    zz = work.zz
    idx = work.idx
    invidx = work.invidx
    diag = work.diag

    M, N = A.shape
    m = M - N  # active row count (starts at data rows)
    n = N

    if max_iter is None:
        max_iter = 3 * n

    if init_dual:
        w.fill(0.0)
        _compute_dual(w, A, b, 0, m)

    nsetp = 0
    it = 0
    work.mode = 0

    while True:
        if nsetp >= n:
            break

        # choose column to enter
        while True:
            wmax, jmax = _largest_positive_dual(w, nsetp)
            if wmax <= 0.0:
                jmax = -1
                break

            if not diag[idx[jmax]]:
                if m < M:
                    A[m, jmax] = lam

            tau = _construct_apply_householder(A, b, nsetp, jmax, min(m + 1, M))
            if tau >= 0.0:
                break

            w[jmax] = 0.0
            if m < M:
                A[m, jmax] = 0.0

        if jmax < 0:
            break

        if not diag[idx[jmax]]:
            m = min(m + 1, M)
            diag[idx[jmax]] = True

        # move column from Z to P
        nsetp += 1
        idx[nsetp - 1], idx[jmax] = idx[jmax], idx[nsetp - 1]

        if nsetp < n:
            _apply_householder_dual(A, w, b, tau, nsetp - 1, m)

        for i in range(nsetp, m):
            A[i, nsetp - 1] = 0.0
        w[nsetp - 1] = 0.0

        # solve triangular system
        for i in range(nsetp):
            zz[i] = b[i]
        _solve_upper_triangular(zz, A, nsetp)

        dual_flag = False
        while True:
            it += 1
            if it > max_iter:
                work.mode = 1
                break

            imv = nsetp
            alpha = 2.0
            for i in range(nsetp):
                if zz[i] <= 0.0:
                    xi = x[idx[i]]
                    t = -xi / (zz[i] - xi)
                    if alpha > t:
                        imv = i + 1
                        alpha = t

            if alpha == 2.0:
                break

            dual_flag = True

            for i in range(nsetp):
                ix = idx[i]
                x[ix] = x[ix] + alpha * (zz[i] - x[ix])

            while True:
                x[idx[imv - 1]] = 0.0

                if (imv - 1) != (nsetp - 1):
                    for i in range(imv, nsetp):
                        cc, ss, rr = _orthogonal_rotmat(A[i - 1, i], A[i, i])
                        A[i - 1, i] = rr
                        A[i, i] = 0.0

                        for j in range(0, i):
                            A[i - 1, j], A[i, j] = _orthogonal_rotmatvec(cc, ss, A[i - 1, j], A[i, j])
                        for j in range(i + 1, n):
                            A[i - 1, j], A[i, j] = _orthogonal_rotmatvec(cc, ss, A[i - 1, j], A[i, j])

                        b[i - 1], b[i] = _orthogonal_rotmatvec(cc, ss, b[i - 1], b[i])

                    # swap columns
                    for j in range(imv - 1, nsetp - 1):
                        for i in range(0, m):
                            A[i, j + 1], A[i, j] = A[i, j], A[i, j + 1]
                        idx[j], idx[j + 1] = idx[j + 1], idx[j]

                nsetp -= 1

                allfeasible = True
                for i in range(nsetp):
                    if x[idx[i]] <= 0.0:
                        allfeasible = False
                        imv = i + 1
                        break
                if allfeasible:
                    break

            for i in range(nsetp):
                zz[i] = b[i]
            _solve_upper_triangular(zz, A, nsetp)

        if work.mode == 1:
            break

        if dual_flag:
            _compute_dual(w, A, b, nsetp, m)

        for i in range(nsetp):
            x[idx[i]] = zz[i]

    for i in range(n):
        invidx[idx[i]] = i

    sm = 0.0
    if nsetp < M:
        for i in range(nsetp, M):
            bi = float(b[i])
            zz[i] = bi
            sm += bi * bi
    else:
        w.fill(0.0)

    work.rnorm = float(np.sqrt(sm))
    work.nsetp = int(nsetp)
    return m


def _workspace(m: int, n: int) -> _NnlsWorkspace:
    return _NnlsWorkspace(
        A=np.zeros((m, n), dtype=np.float64),
        b=np.zeros(m, dtype=np.float64),
        x=np.zeros(n, dtype=np.float64),
        w=np.zeros(n, dtype=np.float64),
        zz=np.zeros(m, dtype=np.float64),
        idx=np.arange(n, dtype=np.int64),
        invidx=np.arange(n, dtype=np.int64),
        diag=np.zeros(n, dtype=bool),
    )


def nnls_decaes(
    A: np.ndarray,
    b: np.ndarray,
    *,
    max_iter: int | None = None,
    init_last_column: bool = True,
) -> NnlsResult:
    """NNLS solver matching DECAES.jl lsqnonneg initialization + NNLS core."""

    A0 = np.asarray(A, dtype=np.float64)
    b0 = np.asarray(b, dtype=np.float64)
    if A0.ndim != 2:
        raise ValueError("A must be 2D")
    m, n = A0.shape
    if b0.shape != (m,):
        raise ValueError(f"b must be shape ({m},), got {b0.shape}")

    w = _workspace(m, n)

    C = w.A
    f = w.b
    x = w.x
    dual = w.w
    z = w.zz
    idx = w.idx

    if init_last_column:
        # DECAES initialization that biases the first active-set candidate toward the last column.
        den = float(np.dot(A0[:, n - 1], A0[:, n - 1]))
        xj = float(np.dot(A0[:, n - 1], b0) / den) if den != 0.0 else 0.0

        z[:m] = b0 - A0[:, n - 1] * xj

        for j in range(0, n - 1):
            dual[j] = float(np.dot(A0[:, j], z[:m]))
            C[:, j] = A0[:, j]

        dual[n - 1] = 0.0
        dual[n - 1] = float(np.all(dual <= 0.0))

        f[:] = b0
        C[:, n - 1] = A0[:, n - 1]

        x.fill(0.0)
        for j in range(n):
            idx[j] = j

        _unsafe_nnls(w, init_dual=False, max_iter=max_iter)
        return NnlsResult(x=w.x.copy(), rnorm=w.rnorm)

    # Generic initialization
    w.A[:, :] = A0
    w.b[:] = b0
    w.x.fill(0.0)
    w.idx[:] = np.arange(n, dtype=np.int64)
    _unsafe_nnls(w, init_dual=True, max_iter=max_iter)
    return NnlsResult(x=w.x.copy(), rnorm=w.rnorm)


def nnls_tikhonov(
    A: np.ndarray,
    b: np.ndarray,
    mu: float,
    *,
    max_iter: int | None = None,
    init_last_column: bool = True,
) -> NnlsResult:
    """Tikhonov-regularized NNLS matching DECAES.jl (no explicit augmentation)."""

    A0 = np.asarray(A, dtype=np.float64)
    b0 = np.asarray(b, dtype=np.float64)
    mu = float(mu)
    if mu < 0:
        raise ValueError("mu must be >= 0")
    if mu == 0.0:
        return nnls_decaes(A0, b0, max_iter=max_iter, init_last_column=init_last_column)

    if not init_last_column:
        # DECAES only defines the last-column-biased initialization; keep behavior consistent.
        return nnls_decaes(A0, b0, max_iter=max_iter, init_last_column=False)

    m0, n = A0.shape
    M = m0 + n

    w = _workspace(M, n)
    C = w.A
    f = w.b
    x = w.x
    dual = w.w
    z = w.zz
    idx = w.idx
    diag = w.diag

    # Initialize workspace matrix with A0 in top block and zeros elsewhere
    C[:m0, :] = A0
    C[m0:, :] = 0.0

    # Initialize padded RHS
    f[:m0] = b0
    f[m0:] = 0.0

    x.fill(0.0)
    idx[:] = np.arange(n, dtype=np.int64)
    diag[:] = False

    # x = A[:, end] \ b with Tikhonov adjustment (||A[:,end]||^2 + mu^2)
    den = float(np.dot(A0[:, n - 1], A0[:, n - 1]) + mu * mu)
    xj = float(np.dot(A0[:, n - 1], b0) / den) if den != 0.0 else 0.0

    # z = b - A[:, end] * xj (data rows only)
    z[:m0] = b0 - A0[:, n - 1] * xj

    for j in range(0, n - 1):
        dual[j] = float(np.dot(A0[:, j], z[:m0]))

    dual[n - 1] = 0.0
    dual[n - 1] = float(np.all(dual <= 0.0))

    _unsafe_nnls_tikhonov(w, mu, init_dual=False, max_iter=max_iter)
    return NnlsResult(x=w.x.copy(), rnorm=w.rnorm)
