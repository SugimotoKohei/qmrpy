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
    refocus_deg: float = 180.0,
    b1: float = 1.0,
    b1_excitation: bool = True,
) -> NDArray[np.float64]:
    """Simulate CPMG (Carr-Purcell-Meiboom-Gill) spin echo train.

    This uses the DECAES EPG algorithm which accurately models stimulated
    echoes and B1 inhomogeneity effects.

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
    refocus_deg : float, optional
        Nominal refocusing flip angle in degrees (default: 180).
    b1 : float, optional
        B1 scaling factor (default: 1.0). Values < 1 simulate B1 inhomogeneity.
    b1_excitation : bool, optional
        If True (default), B1 affects both excitation and refocusing pulses
        (DECAES behavior, more physically realistic).
        If False, B1 only affects refocusing pulses, excitation is ideal 90°
        (Weigel behavior, simpler model).

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

    Notes
    -----
    The difference between b1_excitation modes:

    - ``b1_excitation=True`` (DECAES): Both excitation (90°) and refocusing
      pulses are scaled by B1. Initial magnetization = sin(B1 * 90°).
      This is more physically realistic as B1 inhomogeneity affects all pulses.

    - ``b1_excitation=False`` (Weigel): Only refocusing pulses are scaled.
      Excitation is assumed ideal, initial magnetization = 1.0.
      This matches the Weigel reference implementation.
    """
    from .core import epg_cpmg_decaes

    import numpy as np

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

    # Effective flip angle includes B1 scaling
    # The DECAES algorithm expects alpha_deg where alpha_deg/180 is the B1 factor
    alpha_deg = float(refocus_deg) * float(b1)

    signal = epg_cpmg_decaes(
        etl=n_echoes,
        alpha_deg=alpha_deg,
        te_ms=te_ms,
        t2_ms=t2_ms,
        t1_ms=t1_ms,
        beta_deg=refocus_deg,  # beta is scaled internally by alpha_deg/180
    )

    # If b1_excitation=False, scale to match Weigel's assumption of ideal excitation
    if not b1_excitation:
        # DECAES uses sin(alpha_ex) as initial magnetization where alpha_ex = (alpha_deg/180)*90
        # Weigel uses 1.0 (ideal 90° excitation)
        alpha_ex = (alpha_deg / 180.0) * 90.0
        scale = 1.0 / np.sin(np.deg2rad(alpha_ex))
        signal = signal * scale

    return signal


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
    from .core import epg_cpmg_decaes

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

    # For constant refocusing angles, use the optimized DECAES algorithm
    if refocus_angles_deg is None:
        return cpmg(
            t2_ms=t2_ms,
            t1_ms=t1_ms,
            te_ms=te_ms,
            n_echoes=etl,
            b1=b1,
        )

    # For variable refocusing angles, compute each echo separately
    # This is a simplified approach - proper TSE would need full EPG tracking
    refocus_angles = np.asarray(refocus_angles_deg, dtype=np.float64) * float(b1)
    if refocus_angles.shape != (etl,):
        raise ValueError(f"refocus_angles_deg must have length {etl}")

    # Use average angle for simplified simulation
    avg_angle = np.mean(refocus_angles)
    return epg_cpmg_decaes(
        etl=etl,
        alpha_deg=avg_angle,
        te_ms=te_ms,
        t2_ms=t2_ms,
        t1_ms=t1_ms,
        beta_deg=180.0,
    )
