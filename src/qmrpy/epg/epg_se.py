"""EPG simulations for Spin Echo sequences.

This module provides EPG-based signal simulations for spin echo sequences
including CPMG (Carr-Purcell-Meiboom-Gill) and TSE/FSE (Turbo/Fast Spin Echo).

References
----------
.. [1] Carr HY, Purcell EM (1954). Effects of diffusion on free precession in
       nuclear magnetic resonance experiments. Phys Rev, 94(3):630-638.
.. [2] Meiboom S, Gill D (1958). Modified spin-echo method for measuring nuclear
       relaxation times. Rev Sci Instrum, 29(8):688-691.
.. [3] Hennig J (1988). Multiecho imaging sequences with low refocusing flip angles.
       J Magn Reson, 78(3):397-407.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
else:
    NDArray = Any  # type: ignore[misc,assignment]


def cpmg(
    *,
    t2_ms: float,
    t1_ms: float,
    te_ms: float,
    n_echoes: int,
    excitation_deg: float = 90.0,
    refocus_deg: float = 180.0,
    b1: float = 1.0,
) -> NDArray[np.float64]:
    """Simulate CPMG (Carr-Purcell-Meiboom-Gill) spin echo train.

    This simulates a standard CPMG sequence with a 90° excitation pulse
    followed by a train of 180° refocusing pulses.

    Parameters
    ----------
    t2_ms : float
        T2 relaxation time in milliseconds.
    t1_ms : float
        T1 relaxation time in milliseconds.
    te_ms : float
        Echo spacing in milliseconds.
    n_echoes : int
        Number of echoes to simulate.
    excitation_deg : float, optional
        Excitation flip angle in degrees (default: 90).
    refocus_deg : float, optional
        Refocusing flip angle in degrees (default: 180).
    b1 : float, optional
        B1 scaling factor (default: 1.0). Values < 1 simulate B1 inhomogeneity.

    Returns
    -------
    ndarray
        Signal magnitude at each echo time, shape (n_echoes,).

    Examples
    --------
    >>> from qmrpy.epg import epg_se
    >>> signal = epg_se.cpmg(t2_ms=80, t1_ms=1000, te_ms=10, n_echoes=32)
    >>> signal.shape
    (32,)
    """
    from .core import EPGSimulator

    n_echoes = int(n_echoes)
    if n_echoes < 1:
        raise ValueError("n_echoes must be >= 1")
    if te_ms <= 0:
        raise ValueError("te_ms must be > 0")
    if t1_ms <= 0:
        raise ValueError("t1_ms must be > 0")
    if t2_ms <= 0:
        raise ValueError("t2_ms must be > 0")
    if b1 <= 0:
        raise ValueError("b1 must be > 0")

    import numpy as np

    # Apply B1 scaling
    alpha_ex = float(excitation_deg) * float(b1)
    alpha_ref = float(refocus_deg) * float(b1)

    # Initialize simulator
    sim = EPGSimulator(n_states=n_echoes + 1, t1_ms=t1_ms, t2_ms=t2_ms)
    sim.reset(m0=1.0)

    # Excitation pulse
    sim.apply_rf(alpha_ex)

    signals = np.zeros(n_echoes, dtype=np.float64)

    for i in range(n_echoes):
        # TE/2 relaxation + dephasing
        sim.apply_relaxation(te_ms / 2.0)
        sim.apply_gradient_dephasing()

        # Refocusing pulse
        sim.apply_rf(alpha_ref)

        # TE/2 relaxation + rephasing
        sim.apply_gradient_rephasing()
        sim.apply_relaxation(te_ms / 2.0)

        # Record echo
        signals[i] = sim.get_signal_magnitude()

    return signals


def mese(
    *,
    t2_ms: float,
    t1_ms: float,
    te_ms: float,
    n_echoes: int,
    b1: float = 1.0,
) -> NDArray[np.float64]:
    """Simulate Multi-Echo Spin Echo (MESE) sequence.

    Alias for CPMG with standard parameters.

    Parameters
    ----------
    t2_ms : float
        T2 relaxation time in milliseconds.
    t1_ms : float
        T1 relaxation time in milliseconds.
    te_ms : float
        Echo spacing in milliseconds.
    n_echoes : int
        Number of echoes.
    b1 : float, optional
        B1 scaling factor (default: 1.0).

    Returns
    -------
    ndarray
        Signal magnitude at each echo, shape (n_echoes,).
    """
    return cpmg(
        t2_ms=t2_ms,
        t1_ms=t1_ms,
        te_ms=te_ms,
        n_echoes=n_echoes,
        b1=b1,
    )


def tse(
    *,
    t2_ms: float,
    t1_ms: float,
    te_ms: float,
    etl: int,
    refocus_angles_deg: NDArray[np.float64] | list[float] | None = None,
    b1: float = 1.0,
) -> NDArray[np.float64]:
    """Simulate Turbo Spin Echo (TSE) / Fast Spin Echo (FSE) sequence.

    TSE/FSE uses variable refocusing flip angles to optimize image contrast
    and reduce SAR (Specific Absorption Rate).

    Parameters
    ----------
    t2_ms : float
        T2 relaxation time in milliseconds.
    t1_ms : float
        T1 relaxation time in milliseconds.
    te_ms : float
        Echo spacing in milliseconds.
    etl : int
        Echo train length.
    refocus_angles_deg : array-like, optional
        Refocusing flip angles for each echo in degrees.
        If None, uses constant 180° pulses (equivalent to CPMG).
    b1 : float, optional
        B1 scaling factor (default: 1.0).

    Returns
    -------
    ndarray
        Signal magnitude at each echo, shape (etl,).

    Examples
    --------
    >>> from qmrpy.epg import epg_se
    >>> # Variable flip angle TSE
    >>> angles = [180, 160, 140, 120, 100, 80, 60, 40]
    >>> signal = epg_se.tse(t2_ms=80, t1_ms=1000, te_ms=10, etl=8,
    ...                     refocus_angles_deg=angles)
    """
    from .core import EPGSimulator

    import numpy as np

    etl = int(etl)
    if etl < 1:
        raise ValueError("etl must be >= 1")
    if te_ms <= 0:
        raise ValueError("te_ms must be > 0")
    if t1_ms <= 0:
        raise ValueError("t1_ms must be > 0")
    if t2_ms <= 0:
        raise ValueError("t2_ms must be > 0")
    if b1 <= 0:
        raise ValueError("b1 must be > 0")

    # Set up refocusing angles
    if refocus_angles_deg is None:
        refocus_angles = np.full(etl, 180.0, dtype=np.float64)
    else:
        refocus_angles = np.asarray(refocus_angles_deg, dtype=np.float64)
        if refocus_angles.shape != (etl,):
            raise ValueError(f"refocus_angles_deg must have length {etl}")

    # Apply B1 scaling
    alpha_ex = 90.0 * float(b1)
    refocus_angles = refocus_angles * float(b1)

    # Initialize simulator
    sim = EPGSimulator(n_states=etl + 1, t1_ms=t1_ms, t2_ms=t2_ms)
    sim.reset(m0=1.0)

    # Excitation pulse
    sim.apply_rf(alpha_ex)

    signals = np.zeros(etl, dtype=np.float64)

    for i in range(etl):
        # TE/2 relaxation + dephasing
        sim.apply_relaxation(te_ms / 2.0)
        sim.apply_gradient_dephasing()

        # Refocusing pulse (variable angle)
        sim.apply_rf(refocus_angles[i])

        # TE/2 relaxation + rephasing
        sim.apply_gradient_rephasing()
        sim.apply_relaxation(te_ms / 2.0)

        # Record echo
        signals[i] = sim.get_signal_magnitude()

    return signals


def decay_curve(
    *,
    t2_ms: float,
    t1_ms: float,
    te_ms: float,
    etl: int,
    alpha_deg: float = 180.0,
    beta_deg: float = 180.0,
    b1: float = 1.0,
    backend: str = "native",
) -> NDArray[np.float64]:
    """Compute normalized spin echo decay curve with EPG correction.

    This function provides a unified interface for computing EPG-corrected
    decay curves, compatible with the DECAES implementation.

    Parameters
    ----------
    t2_ms : float
        T2 relaxation time in milliseconds.
    t1_ms : float
        T1 relaxation time in milliseconds.
    te_ms : float
        Echo spacing in milliseconds.
    etl : int
        Echo train length.
    alpha_deg : float, optional
        Refocusing flip angle in degrees (default: 180).
    beta_deg : float, optional
        Alternative refocusing angle for echoes 2+ (default: 180).
        Only used with backend="decaes".
    b1 : float, optional
        B1 scaling factor (default: 1.0).
    backend : {"native", "decaes"}, optional
        Backend implementation (default: "native").

    Returns
    -------
    ndarray
        Normalized decay curve of length ``etl``.
    """
    import numpy as np

    backend_norm = str(backend).lower().strip()

    if backend_norm == "decaes":
        # Use DECAES-compatible implementation
        from qmrpy.models.t2.decaes_t2 import epg_decay_curve as _decaes_epg

        return _decaes_epg(
            etl=etl,
            alpha_deg=float(alpha_deg) * float(b1),
            te_ms=te_ms,
            t2_ms=t2_ms,
            t1_ms=t1_ms,
            beta_deg=beta_deg,
        )
    elif backend_norm == "native":
        # Use native EPG simulation
        signal = cpmg(
            t2_ms=t2_ms,
            t1_ms=t1_ms,
            te_ms=te_ms,
            n_echoes=etl,
            refocus_deg=alpha_deg,
            b1=b1,
        )
        # Normalize to first echo
        if signal[0] > 0:
            return signal / signal[0]
        return signal
    else:
        raise ValueError("backend must be 'native' or 'decaes'")
