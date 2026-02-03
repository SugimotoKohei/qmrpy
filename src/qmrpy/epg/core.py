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


def gradient_dephasing(
    states: NDArray[np.complex128],
    n_states: int,
) -> NDArray[np.complex128]:
    """Apply gradient dephasing to EPG state matrix.

    Shifts F+ states up, F- states down, and keeps Z states in place.
    This represents the effect of a dephasing gradient.

    Parameters
    ----------
    states : ndarray
        EPG state matrix of shape (n_states, 3) with columns [F+, F-, Z].
    n_states : int
        Number of states to track.

    Returns
    -------
    ndarray
        Updated state matrix after gradient dephasing.
    """
    import numpy as np

    n = int(n_states)
    new_states = np.zeros((n, 3), dtype=np.complex128)

    # F+ shifts up (higher order)
    new_states[1:, 0] = states[:-1, 0]
    # F- shifts down (lower order), with F-[0] coming from conjugate of F+[1]
    new_states[0, 1] = np.conj(states[1, 0])
    new_states[1:, 1] = states[:-1, 1]
    # Z states stay in place
    new_states[:, 2] = states[:, 2]

    return new_states


def gradient_rephasing(
    states: NDArray[np.complex128],
    n_states: int,
) -> NDArray[np.complex128]:
    """Apply gradient rephasing to EPG state matrix.

    Opposite of dephasing: shifts F+ down, F- up.

    Parameters
    ----------
    states : ndarray
        EPG state matrix of shape (n_states, 3).
    n_states : int
        Number of states to track.

    Returns
    -------
    ndarray
        Updated state matrix after gradient rephasing.
    """
    import numpy as np

    n = int(n_states)
    new_states = np.zeros((n, 3), dtype=np.complex128)

    # F+ shifts down
    new_states[:-1, 0] = states[1:, 0]
    new_states[0, 0] += np.conj(states[1, 1])
    # F- shifts up
    new_states[1:, 1] = states[:-1, 1]
    # Z states stay
    new_states[:, 2] = states[:, 2]

    return new_states


class EPGSimulator:
    """Base EPG simulator with state tracking.

    This class provides the core EPG simulation engine that tracks
    magnetization states through RF pulses, relaxation, and gradients.

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

    def apply_relaxation(self, t_ms: float) -> None:
        """Apply relaxation for a time interval.

        Parameters
        ----------
        t_ms : float
            Time interval in milliseconds.
        """
        e_vec = relaxation_operator(t_ms, self.t1_ms, self.t2_ms)
        self.states = e_vec * self.states

        # T1 recovery toward equilibrium
        import numpy as np

        e1 = float(np.exp(-t_ms / self.t1_ms))
        self.states[0, 2] += 1.0 - e1

    def apply_gradient_dephasing(self) -> None:
        """Apply a dephasing gradient."""
        self.states = gradient_dephasing(self.states, self.n_states)

    def apply_gradient_rephasing(self) -> None:
        """Apply a rephasing gradient."""
        self.states = gradient_rephasing(self.states, self.n_states)

    def apply_spoiler(self) -> None:
        """Apply ideal spoiler gradient (destroy all transverse magnetization)."""
        self.states[:, 0] = 0.0  # F+ = 0
        self.states[:, 1] = 0.0  # F- = 0

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
