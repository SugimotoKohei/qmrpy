"""Core EPG (Extended Phase Graph) simulation engine.

This module provides the fundamental EPG state transition matrices and
relaxation operators used by both spin echo and gradient echo sequences.

References
----------
.. [1] Hennig J. (1988). Multiecho imaging sequences with low refocusing flip angles.
       J Magn Reson, 78(3):397-407.
.. [2] Weigel M. (2015). Extended phase graphs: dephasing, RF pulses, and echoes -
       pure and simple. J Magn Reson Imaging, 41(2):266-295.
.. [3] DECAES.jl - https://github.com/jondeuce/DECAES.jl
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
else:
    NDArray = Any  # type: ignore[misc,assignment]


def rf_rotation_matrix(alpha_deg: float) -> NDArray[np.complex128]:
    """Compute the RF rotation matrix for EPG state transitions.

    This implements the Hennig (1988) formulation for the effect of an
    RF pulse on the EPG states (F+, F-, Z).

    Parameters
    ----------
    alpha_deg : float
        Flip angle in degrees.

    Returns
    -------
    ndarray
        3x3 complex rotation matrix.

    References
    ----------
    .. [1] Hennig J. (1988). J Magn Reson, 78(3):397-407.
    """
    import numpy as np

    alpha_rad = np.deg2rad(float(alpha_deg))
    a2 = alpha_rad / 2.0

    cos2 = np.cos(a2) ** 2
    sin2 = np.sin(a2) ** 2
    sin_alpha = np.sin(alpha_rad)
    cos_alpha = np.cos(alpha_rad)

    return np.array(
        [
            [cos2, sin2, -1j * sin_alpha],
            [sin2, cos2, 1j * sin_alpha],
            [-1j * sin_alpha / 2.0, 1j * sin_alpha / 2.0, cos_alpha],
        ],
        dtype=np.complex128,
    )


def relaxation_operator(
    t_ms: float,
    t1_ms: float,
    t2_ms: float,
) -> NDArray[np.float64]:
    """Compute relaxation operator for EPG states.

    Parameters
    ----------
    t_ms : float
        Time interval in milliseconds.
    t1_ms : float
        T1 relaxation time in milliseconds.
    t2_ms : float
        T2 relaxation time in milliseconds.

    Returns
    -------
    ndarray
        Relaxation factors [E2, E2, E1] for F+, F-, Z states.
    """
    import numpy as np

    if t_ms < 0:
        raise ValueError("t_ms must be >= 0")
    if t1_ms <= 0:
        raise ValueError("t1_ms must be > 0")
    if t2_ms <= 0:
        raise ValueError("t2_ms must be > 0")

    e1 = float(np.exp(-t_ms / t1_ms))
    e2 = float(np.exp(-t_ms / t2_ms))

    return np.array([e2, e2, e1], dtype=np.float64)


def epg_weigel(
    *,
    n_echoes: int,
    alpha_deg: float | NDArray[np.float64] | list[float],
    te_ms: float,
    t2_ms: float,
    t1_ms: float,
    b1: float = 1.0,
    cpmg: bool = True,
) -> NDArray[np.float64]:
    """Compute spin echo decay curve using Weigel's EPG algorithm.

    This is a faithful port of Weigel's MATLAB EPG implementation from
    https://github.com/matthias-weigel/EPG, extended to include B1 effects
    on the excitation pulse and variable flip angle support.

    Parameters
    ----------
    n_echoes : int
        Number of echoes (echo train length).
    alpha_deg : float or array-like
        Refocusing flip angle(s) in degrees (nominal, before B1 scaling).
        If a scalar, the same angle is used for all refocusing pulses.
        If an array of length ``n_echoes``, each echo uses its own angle.
    te_ms : float
        Echo spacing in milliseconds.
    t2_ms : float
        T2 relaxation time in milliseconds.
    t1_ms : float
        T1 relaxation time in milliseconds.
    b1 : float, optional
        B1 inhomogeneity factor (default 1.0). Applied to both excitation
        and refocusing pulses. Actual flip angles become ``b1 * nominal``.
    cpmg : bool, optional
        If True (default), use CPMG phase cycling (refocusing 90° from excitation).
        If False, use CP phase cycling (same phase as excitation).

    Returns
    -------
    ndarray
        Echo magnitudes of length ``n_echoes``.

    Notes
    -----
    **B1 inhomogeneity effects:**

    - Excitation pulse: Initial transverse magnetization = sin(b1 × 90°)
    - Refocusing pulses: Actual flip angle = b1 × alpha_deg

    **CP vs CPMG phase cycling:**

    - **CPMG**: Initial state F+(0) = F-(0) = M0 (real).
      Even echoes self-compensate for B1 errors.

    - **CP**: Initial state F+(0) = -i×M0, F-(0) = +i×M0 (imaginary).
      B1 errors accumulate with each refocusing pulse.

    References
    ----------
    .. [1] Weigel M (2015). Extended phase graphs: dephasing, RF pulses, and echoes -
           pure and simple. J Magn Reson Imaging, 41(2):266-295.
    """
    import numpy as np

    N = int(n_echoes)
    if N < 1:
        raise ValueError("n_echoes must be >= 1")
    if te_ms <= 0:
        raise ValueError("te_ms must be > 0")
    if t2_ms <= 0:
        raise ValueError("t2_ms must be > 0")
    if t1_ms <= 0:
        raise ValueError("t1_ms must be > 0")

    # Handle variable flip angles
    alpha_arr = np.atleast_1d(np.asarray(alpha_deg, dtype=np.float64))
    if alpha_arr.size == 1:
        # Single angle: broadcast to all echoes
        alpha_arr = np.full(N, alpha_arr[0], dtype=np.float64)
    elif alpha_arr.size != N:
        raise ValueError(
            f"alpha_deg must be a scalar or array of length n_echoes ({N}), "
            f"got length {alpha_arr.size}"
        )

    # Apply B1 scaling to refocusing angles
    alpha_rad_arr = np.deg2rad(alpha_arr * b1)

    # Initial transverse magnetization after B1-scaled 90° excitation
    # M_xy = sin(B1 × 90°)
    m0 = np.sin(np.deg2rad(90.0 * b1))

    # Relaxation for TE/2
    E1 = np.exp(-te_ms / t1_ms / 2.0)
    E2 = np.exp(-te_ms / t2_ms / 2.0)

    # State vectors: Omega[0]=F+, Omega[1]=F-, Omega[2]=Z
    # Index k corresponds to dephasing order
    Nt2p1 = 2 * N + 1
    Omega_preRF = np.zeros((3, Nt2p1), dtype=np.complex128)
    Omega_postRF = np.zeros((3, Nt2p1), dtype=np.complex128)

    # Initial state after B1-scaled 90° excitation
    if cpmg:
        # CPMG: F+(0) = F-(0) = M0 (real)
        Omega_postRF[0, 0] = m0
        Omega_postRF[1, 0] = m0
    else:
        # CP: F+(0) = -i×M0, F-(0) = +i×M0 (imaginary)
        Omega_postRF[0, 0] = -1j * m0
        Omega_postRF[1, 0] = +1j * m0

    # Output: echo signal at each echo time
    echoes = np.zeros(N, dtype=np.float64)

    for pn in range(1, N + 1):
        # RF rotation matrix for current refocusing pulse (0-indexed: pn-1)
        fa = alpha_rad_arr[pn - 1]
        cos2 = np.cos(fa / 2) ** 2
        sin2 = np.sin(fa / 2) ** 2
        sin_fa = np.sin(fa)
        cos_fa = np.cos(fa)

        T = np.array([
            [cos2, sin2, -1j * sin_fa],
            [sin2, cos2, +1j * sin_fa],
            [-0.5j * sin_fa, +0.5j * sin_fa, cos_fa],
        ], dtype=np.complex128)

        pn2 = 2 * pn
        # k indices (0-based in Python)
        k_max = pn2 - 1  # up to index pn2-2 in 0-based

        # === First TE/2: Relaxation then dephasing ===
        # Relaxation
        Omega_preRF[0:2, :k_max] = E2 * Omega_postRF[0:2, :k_max]
        Omega_preRF[2, 1:k_max] = E1 * Omega_postRF[2, 1:k_max]
        Omega_preRF[2, 0] = E1 * Omega_postRF[2, 0] + (1 - E1)  # T1 recovery

        # Gradient dephasing: F+ shifts up, F- shifts down
        # F+(k) -> F+(k+1), F-(k+1) -> F-(k), F-(0) -> F+(0)*
        Omega_preRF[0, 1:k_max + 1] = Omega_preRF[0, 0:k_max]
        Omega_preRF[1, 0:k_max] = Omega_preRF[1, 1:k_max + 1]
        Omega_preRF[0, 0] = np.conj(Omega_preRF[1, 0])

        # === RF pulse ===
        kp1_max = pn2  # up to index pn2-1 in 0-based
        Omega_postRF[:, :kp1_max] = T @ Omega_preRF[:, :kp1_max]

        # === Second TE/2: Relaxation then dephasing ===
        # Relaxation
        Omega_postRF[0:2, :kp1_max] = E2 * Omega_postRF[0:2, :kp1_max]
        Omega_postRF[2, 1:kp1_max] = E1 * Omega_postRF[2, 1:kp1_max]
        Omega_postRF[2, 0] = E1 * Omega_postRF[2, 0] + (1 - E1)

        # Gradient dephasing
        kp2_max = pn2 + 1
        Omega_postRF[0, 1:kp2_max] = Omega_postRF[0, 0:kp2_max - 1]
        Omega_postRF[1, 0:kp2_max - 1] = Omega_postRF[1, 1:kp2_max]
        Omega_postRF[0, 0] = np.conj(Omega_postRF[1, 0])

        # Record echo (F+(0) at echo time)
        echoes[pn - 1] = np.abs(Omega_postRF[0, 0])

    return echoes


def epg_cpmg_decaes(
    *,
    etl: int,
    alpha_deg: float,
    te_ms: float,
    t2_ms: float,
    t1_ms: float,
    beta_deg: float = 180.0,
) -> NDArray[np.float64]:
    """Compute CPMG spin echo decay curve using the DECAES EPG algorithm.

    This is a faithful port of the DECAES.jl EPG implementation, which uses
    the Hennig (1988) algorithm with Jones (1997) phase state transitions.

    This implementation assumes CPMG phase cycling (refocusing pulses 90°
    from excitation pulse).

    Parameters
    ----------
    etl : int
        Echo train length.
    alpha_deg : float
        Effective flip angle in degrees (includes B1 scaling).
    te_ms : float
        Echo spacing in milliseconds.
    t2_ms : float
        T2 relaxation time in milliseconds.
    t1_ms : float
        T1 relaxation time in milliseconds.
    beta_deg : float, optional
        Refocusing angle for echoes 2+ in degrees (default: 180).

    Returns
    -------
    ndarray
        Decay curve of length ``etl``.
    """
    import numpy as np

    etl = int(etl)
    if etl < 1:
        raise ValueError("etl must be >= 1")
    if te_ms <= 0:
        raise ValueError("te_ms must be > 0")
    if t2_ms <= 0:
        raise ValueError("t2_ms must be > 0")
    if t1_ms <= 0:
        raise ValueError("t1_ms must be > 0")

    # B1 factor (alpha_deg/180 is the effective B1)
    A = float(alpha_deg) / 180.0
    alpha_ex = A * 90.0      # Excitation pulse
    alpha1 = A * 180.0       # First refocusing pulse
    alphai = A * float(beta_deg)  # Subsequent refocusing pulses

    # Relaxation for TE/2
    E1 = float(np.exp(-((te_ms / 2.0) / t1_ms)))
    E2 = float(np.exp(-((te_ms / 2.0) / t2_ms)))
    E = np.array([E2, E2, E1], dtype=np.complex128)

    # RF rotation matrices
    R1 = rf_rotation_matrix(alpha1)
    Ri = rf_rotation_matrix(alphai)

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


class EPGSimulator:
    """Base EPG simulator with state tracking.

    This class provides a general EPG simulation engine that tracks
    magnetization states through RF pulses, relaxation, and gradients.

    Note: For CPMG/spin echo sequences, use :func:`epg_cpmg_decaes` instead,
    which implements the optimized DECAES algorithm.

    Parameters
    ----------
    n_states : int
        Number of phase states to track.
    t1_ms : float
        T1 relaxation time in milliseconds.
    t2_ms : float
        T2 relaxation time in milliseconds.

    Attributes
    ----------
    states : ndarray
        Current EPG state matrix of shape (n_states, 3).
        Columns are [F+, F-, Z].
    """

    def __init__(
        self,
        n_states: int,
        t1_ms: float,
        t2_ms: float,
    ) -> None:
        import numpy as np

        self.n_states = int(n_states)
        self.t1_ms = float(t1_ms)
        self.t2_ms = float(t2_ms)

        if self.n_states < 1:
            raise ValueError("n_states must be >= 1")
        if self.t1_ms <= 0:
            raise ValueError("t1_ms must be > 0")
        if self.t2_ms <= 0:
            raise ValueError("t2_ms must be > 0")

        # Initialize state matrix: (n_states, 3) for [F+, F-, Z]
        self.states: NDArray[np.complex128] = np.zeros(
            (self.n_states, 3), dtype=np.complex128
        )

    def reset(self, m0: float = 1.0) -> None:
        """Reset to thermal equilibrium.

        Parameters
        ----------
        m0 : float
            Initial longitudinal magnetization.
        """
        import numpy as np

        self.states = np.zeros((self.n_states, 3), dtype=np.complex128)
        self.states[0, 2] = float(m0)  # Z0 = M0

    def apply_rf(self, alpha_deg: float) -> None:
        """Apply an RF pulse.

        Parameters
        ----------
        alpha_deg : float
            Flip angle in degrees.
        """
        r_mat = rf_rotation_matrix(alpha_deg)
        self.states = (r_mat @ self.states.T).T

    def apply_relaxation(self, t_ms: float, recovery: bool = True) -> None:
        """Apply relaxation for a time interval.

        Parameters
        ----------
        t_ms : float
            Time interval in milliseconds.
        recovery : bool, optional
            Whether to include T1 recovery (default: True).
        """
        import numpy as np

        e_vec = relaxation_operator(t_ms, self.t1_ms, self.t2_ms)
        self.states = e_vec * self.states

        # T1 recovery toward equilibrium
        if recovery:
            e1 = float(np.exp(-t_ms / self.t1_ms))
            self.states[0, 2] += 1.0 - e1

    def apply_spoiler(self) -> None:
        """Apply ideal spoiler gradient (destroy all transverse magnetization)."""
        self.states[:, 0] = 0.0  # F+ = 0
        self.states[:, 1] = 0.0  # F- = 0

    def apply_gradient_dephasing(self) -> None:
        """Apply gradient dephasing (shift F+ states up, F- states down).

        This shifts the phase state indices:
        - F+[k] -> F+[k+1]
        - F-[k] -> F-[k-1]
        - F-[0] -> F+[0] (echo formation condition)
        """
        import numpy as np

        # Shift F+ up (higher k states)
        self.states[1:, 0] = self.states[:-1, 0]
        self.states[0, 0] = 0.0  # New F+[0] will be filled by F-[0]

        # Shift F- down (lower k states)
        # F-[0] becomes new F+[0] (echo)
        self.states[0, 0] = np.conj(self.states[0, 1])
        self.states[:-1, 1] = self.states[1:, 1]
        self.states[-1, 1] = 0.0

    def apply_gradient_rephasing(self) -> None:
        """Apply gradient rephasing (shift F+ states down, F- states up).

        This is the inverse of dephasing, used in SSFP-Echo sequences.
        """
        import numpy as np

        # Shift F+ down (lower k states)
        temp = self.states[0, 0]
        self.states[:-1, 0] = self.states[1:, 0]
        self.states[-1, 0] = 0.0

        # Shift F- up (higher k states)
        self.states[1:, 1] = self.states[:-1, 1]
        self.states[0, 1] = np.conj(temp)

    def get_signal(self) -> complex:
        """Get the current signal (F+[0] state).

        Returns
        -------
        complex
            Current transverse magnetization signal.
        """
        return complex(self.states[0, 0])

    def get_signal_magnitude(self) -> float:
        """Get the magnitude of the current signal.

        Returns
        -------
        float
            Magnitude of transverse magnetization.
        """
        import numpy as np

        return float(np.abs(self.states[0, 0]))
