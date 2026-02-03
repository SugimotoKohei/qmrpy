"""EPG simulations for Gradient Echo sequences.

This module provides EPG-based signal simulations for gradient echo sequences
including SPGR/FLASH (spoiled gradient echo) and bSSFP/TrueFISP (balanced SSFP).

References
----------
.. [1] Haase A, et al. (1986). FLASH imaging. Rapid NMR imaging using low
       flip-angle pulses. J Magn Reson, 67(2):258-266.
.. [2] Oppelt A, et al. (1986). FISP - a new fast MRI sequence.
       Electromedica, 54(1):15-18.
.. [3] Scheffler K, Lehnhardt S (2003). Principles and applications of
       balanced SSFP techniques. Eur Radiol, 13(11):2409-2418.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
else:
    NDArray = Any  # type: ignore[misc,assignment]


def spgr(
    *,
    t1_ms: float,
    tr_ms: float,
    fa_deg: float,
    n_pulses: int = 1,
    b1: float = 1.0,
) -> NDArray[np.float64]:
    """Simulate Spoiled Gradient Recalled Echo (SPGR/FLASH) sequence.

    SPGR uses RF and gradient spoiling to destroy residual transverse
    magnetization, creating pure T1-weighted contrast.

    Parameters
    ----------
    t1_ms : float
        T1 relaxation time in milliseconds.
    tr_ms : float
        Repetition time in milliseconds.
    fa_deg : float
        Flip angle in degrees.
    n_pulses : int, optional
        Number of TR periods to simulate (default: 1).
        Use >1 to observe approach to steady state.
    b1 : float, optional
        B1 scaling factor (default: 1.0).

    Returns
    -------
    ndarray
        Signal magnitude after each TR, shape (n_pulses,).

    Notes
    -----
    The steady-state SPGR signal follows the Ernst equation:

    .. math::

        S = M_0 \\sin(\\alpha) \\frac{1 - E_1}{1 - E_1 \\cos(\\alpha)}

    where :math:`E_1 = \\exp(-TR/T_1)` and :math:`\\alpha` is the flip angle.

    Examples
    --------
    >>> from qmrpy.epg import epg_gre
    >>> signal = epg_gre.spgr(t1_ms=1000, tr_ms=10, fa_deg=15)
    >>> signal[0]  # Steady-state signal
    """
    from .core import EPGSimulator

    import numpy as np

    n_pulses = int(n_pulses)
    if n_pulses < 1:
        raise ValueError("n_pulses must be >= 1")
    if tr_ms <= 0:
        raise ValueError("tr_ms must be > 0")
    if t1_ms <= 0:
        raise ValueError("t1_ms must be > 0")
    if b1 <= 0:
        raise ValueError("b1 must be > 0")

    # Apply B1 scaling
    alpha = float(fa_deg) * float(b1)

    # T2 doesn't matter for SPGR (spoiled), use large value
    t2_ms = 10000.0

    # Initialize simulator
    sim = EPGSimulator(n_states=2, t1_ms=t1_ms, t2_ms=t2_ms)
    sim.reset(m0=1.0)

    signals = np.zeros(n_pulses, dtype=np.float64)

    for i in range(n_pulses):
        # RF excitation
        sim.apply_rf(alpha)

        # Record signal immediately after excitation
        signals[i] = sim.get_signal_magnitude()

        # Ideal spoiling - destroy all transverse magnetization
        sim.apply_spoiler()

        # T1 relaxation during TR
        sim.apply_relaxation(tr_ms)

    return signals


def spgr_steady_state(
    *,
    t1_ms: float,
    tr_ms: float,
    fa_deg: float,
    b1: float = 1.0,
) -> float:
    """Calculate the steady-state SPGR signal using the Ernst equation.

    This is an analytical solution, faster than running the full EPG simulation.

    Parameters
    ----------
    t1_ms : float
        T1 relaxation time in milliseconds.
    tr_ms : float
        Repetition time in milliseconds.
    fa_deg : float
        Flip angle in degrees.
    b1 : float, optional
        B1 scaling factor (default: 1.0).

    Returns
    -------
    float
        Steady-state signal magnitude (normalized to M0=1).

    Notes
    -----
    Uses the Ernst equation:
    S = sin(α) * (1 - E1) / (1 - E1 * cos(α))
    """
    import numpy as np

    if tr_ms <= 0:
        raise ValueError("tr_ms must be > 0")
    if t1_ms <= 0:
        raise ValueError("t1_ms must be > 0")
    if b1 <= 0:
        raise ValueError("b1 must be > 0")

    alpha = np.deg2rad(float(fa_deg) * float(b1))
    e1 = np.exp(-tr_ms / t1_ms)

    return float(np.sin(alpha) * (1 - e1) / (1 - e1 * np.cos(alpha)))


def ernst_angle(t1_ms: float, tr_ms: float) -> float:
    """Calculate the Ernst angle for maximum SPGR signal.

    Parameters
    ----------
    t1_ms : float
        T1 relaxation time in milliseconds.
    tr_ms : float
        Repetition time in milliseconds.

    Returns
    -------
    float
        Optimal flip angle in degrees.

    Notes
    -----
    The Ernst angle is: α = arccos(exp(-TR/T1))
    """
    import numpy as np

    if tr_ms <= 0:
        raise ValueError("tr_ms must be > 0")
    if t1_ms <= 0:
        raise ValueError("t1_ms must be > 0")

    e1 = np.exp(-tr_ms / t1_ms)
    return float(np.rad2deg(np.arccos(e1)))


def bssfp(
    *,
    t1_ms: float,
    t2_ms: float,
    tr_ms: float,
    fa_deg: float,
    n_pulses: int = 1,
    off_resonance_hz: float = 0.0,
    b1: float = 1.0,
) -> NDArray[np.complex128]:
    """Simulate balanced Steady-State Free Precession (bSSFP/TrueFISP) sequence.

    bSSFP maintains coherent transverse magnetization by using balanced
    gradients, creating T2/T1-weighted contrast with high SNR efficiency.

    Parameters
    ----------
    t1_ms : float
        T1 relaxation time in milliseconds.
    t2_ms : float
        T2 relaxation time in milliseconds.
    tr_ms : float
        Repetition time in milliseconds.
    fa_deg : float
        Flip angle in degrees.
    n_pulses : int, optional
        Number of TR periods to simulate (default: 1).
    off_resonance_hz : float, optional
        Off-resonance frequency in Hz (default: 0).
    b1 : float, optional
        B1 scaling factor (default: 1.0).

    Returns
    -------
    ndarray
        Complex signal after each TR, shape (n_pulses,).

    Notes
    -----
    The bSSFP signal depends strongly on off-resonance, creating characteristic
    banding artifacts when off-resonance = ±1/(2*TR).

    Examples
    --------
    >>> from qmrpy.epg import epg_gre
    >>> signal = epg_gre.bssfp(t1_ms=1000, t2_ms=80, tr_ms=5, fa_deg=45)
    """
    from .core import EPGSimulator

    import numpy as np

    n_pulses = int(n_pulses)
    if n_pulses < 1:
        raise ValueError("n_pulses must be >= 1")
    if tr_ms <= 0:
        raise ValueError("tr_ms must be > 0")
    if t1_ms <= 0:
        raise ValueError("t1_ms must be > 0")
    if t2_ms <= 0:
        raise ValueError("t2_ms must be > 0")
    if b1 <= 0:
        raise ValueError("b1 must be > 0")

    # Apply B1 scaling
    alpha = float(fa_deg) * float(b1)

    # Phase accumulation due to off-resonance
    phi = 2.0 * np.pi * float(off_resonance_hz) * (tr_ms / 1000.0)

    # Initialize simulator
    sim = EPGSimulator(n_states=n_pulses + 1, t1_ms=t1_ms, t2_ms=t2_ms)
    sim.reset(m0=1.0)

    signals = np.zeros(n_pulses, dtype=np.complex128)

    for i in range(n_pulses):
        # Alternating RF phase for bSSFP (phase cycling)
        rf_phase = np.pi * i

        # Apply RF with phase
        # For simplicity, we apply the flip angle directly
        # (full implementation would include RF phase)
        sim.apply_rf(alpha)

        # Half TR relaxation
        sim.apply_relaxation(tr_ms / 2.0)

        # Apply off-resonance phase accumulation
        phase_factor = np.exp(1j * phi / 2.0)
        sim.states[:, 0] *= phase_factor
        sim.states[:, 1] *= np.conj(phase_factor)

        # Record signal at echo time (middle of TR)
        signals[i] = sim.get_signal()

        # Apply remaining off-resonance phase
        sim.states[:, 0] *= phase_factor
        sim.states[:, 1] *= np.conj(phase_factor)

        # Second half TR relaxation
        sim.apply_relaxation(tr_ms / 2.0)

    return signals


def bssfp_steady_state(
    *,
    t1_ms: float,
    t2_ms: float,
    tr_ms: float,
    fa_deg: float,
    off_resonance_hz: float = 0.0,
    b1: float = 1.0,
) -> float:
    """Calculate the steady-state bSSFP signal magnitude.

    Parameters
    ----------
    t1_ms : float
        T1 relaxation time in milliseconds.
    t2_ms : float
        T2 relaxation time in milliseconds.
    tr_ms : float
        Repetition time in milliseconds.
    fa_deg : float
        Flip angle in degrees.
    off_resonance_hz : float, optional
        Off-resonance frequency in Hz (default: 0).
    b1 : float, optional
        B1 scaling factor (default: 1.0).

    Returns
    -------
    float
        Steady-state signal magnitude.

    Notes
    -----
    On-resonance bSSFP signal (simplified):
    S ≈ M0 * sin(α) / (1 + cos(α) + (1 - cos(α)) * T1/T2)
    """
    import numpy as np

    if tr_ms <= 0:
        raise ValueError("tr_ms must be > 0")
    if t1_ms <= 0:
        raise ValueError("t1_ms must be > 0")
    if t2_ms <= 0:
        raise ValueError("t2_ms must be > 0")
    if b1 <= 0:
        raise ValueError("b1 must be > 0")

    alpha = np.deg2rad(float(fa_deg) * float(b1))
    e1 = np.exp(-tr_ms / t1_ms)
    e2 = np.exp(-tr_ms / t2_ms)

    # Off-resonance phase
    phi = 2.0 * np.pi * float(off_resonance_hz) * (tr_ms / 1000.0)

    # Freeman-Hill formula for bSSFP
    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)

    # Simplified on-resonance approximation
    if abs(off_resonance_hz) < 1e-6:
        # On-resonance steady state
        denom = 1.0 - e1 * e2 - (e1 - e2) * cos_a
        if abs(denom) < 1e-10:
            return 0.0
        return float(np.sqrt(e2) * (1.0 - e1) * sin_a / denom)
    else:
        # Off-resonance - use numerical approach
        signal = bssfp(
            t1_ms=t1_ms,
            t2_ms=t2_ms,
            tr_ms=tr_ms,
            fa_deg=fa_deg,
            n_pulses=200,  # Enough to reach steady state
            off_resonance_hz=off_resonance_hz,
            b1=b1,
        )
        return float(np.abs(signal[-1]))


def ssfp_fid(
    *,
    t1_ms: float,
    t2_ms: float,
    tr_ms: float,
    fa_deg: float,
    n_pulses: int = 1,
    b1: float = 1.0,
) -> NDArray[np.float64]:
    """Simulate SSFP-FID (Gradient Echo) sequence.

    SSFP-FID acquires signal immediately after the RF pulse,
    sampling the FID portion of the steady-state signal.

    Parameters
    ----------
    t1_ms : float
        T1 relaxation time in milliseconds.
    t2_ms : float
        T2 relaxation time in milliseconds.
    tr_ms : float
        Repetition time in milliseconds.
    fa_deg : float
        Flip angle in degrees.
    n_pulses : int, optional
        Number of TR periods to simulate (default: 1).
    b1 : float, optional
        B1 scaling factor (default: 1.0).

    Returns
    -------
    ndarray
        Signal magnitude after each TR, shape (n_pulses,).
    """
    from .core import EPGSimulator

    import numpy as np

    n_pulses = int(n_pulses)
    if n_pulses < 1:
        raise ValueError("n_pulses must be >= 1")
    if tr_ms <= 0:
        raise ValueError("tr_ms must be > 0")
    if t1_ms <= 0:
        raise ValueError("t1_ms must be > 0")
    if t2_ms <= 0:
        raise ValueError("t2_ms must be > 0")
    if b1 <= 0:
        raise ValueError("b1 must be > 0")

    alpha = float(fa_deg) * float(b1)

    sim = EPGSimulator(n_states=n_pulses + 1, t1_ms=t1_ms, t2_ms=t2_ms)
    sim.reset(m0=1.0)

    signals = np.zeros(n_pulses, dtype=np.float64)

    for i in range(n_pulses):
        # RF excitation
        sim.apply_rf(alpha)

        # Record FID signal immediately
        signals[i] = sim.get_signal_magnitude()

        # Dephasing gradient
        sim.apply_gradient_dephasing()

        # Relaxation during TR
        sim.apply_relaxation(tr_ms)

    return signals


def ssfp_echo(
    *,
    t1_ms: float,
    t2_ms: float,
    tr_ms: float,
    fa_deg: float,
    n_pulses: int = 1,
    b1: float = 1.0,
) -> NDArray[np.float64]:
    """Simulate SSFP-Echo sequence.

    SSFP-Echo acquires the refocused signal just before the next RF pulse,
    sampling the echo portion of the steady-state signal.

    Parameters
    ----------
    t1_ms : float
        T1 relaxation time in milliseconds.
    t2_ms : float
        T2 relaxation time in milliseconds.
    tr_ms : float
        Repetition time in milliseconds.
    fa_deg : float
        Flip angle in degrees.
    n_pulses : int, optional
        Number of TR periods to simulate (default: 1).
    b1 : float, optional
        B1 scaling factor (default: 1.0).

    Returns
    -------
    ndarray
        Signal magnitude before each RF pulse, shape (n_pulses,).
    """
    from .core import EPGSimulator

    import numpy as np

    n_pulses = int(n_pulses)
    if n_pulses < 1:
        raise ValueError("n_pulses must be >= 1")
    if tr_ms <= 0:
        raise ValueError("tr_ms must be > 0")
    if t1_ms <= 0:
        raise ValueError("t1_ms must be > 0")
    if t2_ms <= 0:
        raise ValueError("t2_ms must be > 0")
    if b1 <= 0:
        raise ValueError("b1 must be > 0")

    alpha = float(fa_deg) * float(b1)

    sim = EPGSimulator(n_states=n_pulses + 1, t1_ms=t1_ms, t2_ms=t2_ms)
    sim.reset(m0=1.0)

    signals = np.zeros(n_pulses, dtype=np.float64)

    # First RF pulse
    sim.apply_rf(alpha)

    for i in range(n_pulses):
        # Dephasing gradient
        sim.apply_gradient_dephasing()

        # Relaxation during TR
        sim.apply_relaxation(tr_ms)

        # Rephasing gradient
        sim.apply_gradient_rephasing()

        # Record echo signal just before next RF
        signals[i] = sim.get_signal_magnitude()

        # Next RF pulse
        sim.apply_rf(alpha)

    return signals
