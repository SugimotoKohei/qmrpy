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


def se(
    *,
    t2_ms: float,
    t1_ms: float,
    te_ms: float,
    n_echoes: int = 1,
    refocus_deg: float | NDArray[np.float64] | list[float] = 180.0,
    b1: float = 1.0,
    cpmg: bool = True,
) -> NDArray[np.float64]:
    """Simulate Spin Echo sequence.

    This uses Weigel's EPG algorithm which accurately models stimulated
    echoes, B1 inhomogeneity effects, and CP/CPMG phase cycling.

    Parameters
    ----------
    t2_ms : float
        T2 relaxation time in milliseconds.
    t1_ms : float
        T1 relaxation time in milliseconds.
    te_ms : float
        Echo spacing in milliseconds.
    n_echoes : int, optional
        Number of echoes to simulate (default: 1).
    refocus_deg : float or array-like, optional
        Nominal refocusing flip angle(s) in degrees (default: 180).
        If a scalar, the same angle is used for all refocusing pulses.
        If an array of length ``n_echoes``, each echo uses its own angle
        (variable flip angle / TSE mode).
    b1 : float, optional
        B1 scaling factor (default: 1.0). Values < 1 simulate B1 inhomogeneity.
    cpmg : bool, optional
        If True (default), use CPMG phase cycling (refocusing pulse 90° from
        excitation pulse). If False, use CP phase cycling (same phase).

    Returns
    -------
    ndarray
        Signal magnitude at each echo time, shape (n_echoes,).

    Examples
    --------
    >>> from qmrpy.epg import epg_se
    >>> # Single spin echo
    >>> signal = epg_se.se(t2_ms=80, t1_ms=1000, te_ms=10)
    >>> signal.shape
    (1,)

    >>> # Multi-echo (CPMG)
    >>> signal = epg_se.se(t2_ms=80, t1_ms=1000, te_ms=10, n_echoes=32)
    >>> signal.shape
    (32,)

    >>> # Variable flip angle (TSE/FSE)
    >>> angles = [180, 160, 140, 120, 100, 80, 60, 40]
    >>> signal = epg_se.se(t2_ms=80, t1_ms=1000, te_ms=10, n_echoes=8,
    ...                    refocus_deg=angles)

    >>> # CP mode (no CPMG phase cycling)
    >>> signal = epg_se.se(t2_ms=80, t1_ms=1000, te_ms=10, n_echoes=32, cpmg=False)

    Notes
    -----
    **CP vs CPMG phase cycling:**

    - **CP (Carr-Purcell)**: 90°x - 180°x - 180°x - ...
      Refocusing pulses have the same phase as excitation.
      B1 errors accumulate with each refocusing pulse.

    - **CPMG (Carr-Purcell-Meiboom-Gill)**: 90°x - 180°y - 180°y - ...
      Refocusing pulses are 90° phase-shifted from excitation.
      Even echoes self-compensate for B1 errors.

    References
    ----------
    .. [1] Weigel M (2015). Extended phase graphs: dephasing, RF pulses, and echoes -
           pure and simple. J Magn Reson Imaging, 41(2):266-295.
    """
    from .core import epg_weigel

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

    # Convert refocus_deg to array for validation
    refocus_arr = np.atleast_1d(np.asarray(refocus_deg, dtype=np.float64))
    if refocus_arr.size != 1 and refocus_arr.size != n_echoes:
        raise ValueError(
            f"refocus_deg must be a scalar or array of length n_echoes ({n_echoes}), "
            f"got length {refocus_arr.size}"
        )

    # Pass to epg_weigel; B1 scaling is applied inside
    return epg_weigel(
        n_echoes=n_echoes,
        alpha_deg=refocus_deg,
        te_ms=te_ms,
        t2_ms=t2_ms,
        t1_ms=t1_ms,
        b1=float(b1),
        cpmg=cpmg,
    )


def tse(
    *,
    t2_ms: float,
    t1_ms: float,
    te_ms: float,
    etl: int,
    refocus_angles_deg: NDArray[np.float64] | list[float] | None = None,
    b1: float = 1.0,
    cpmg: bool = True,
) -> NDArray[np.float64]:
    """Simulate Turbo Spin Echo (TSE) / Fast Spin Echo (FSE) sequence.

    TSE/FSE uses variable refocusing flip angles to optimize image contrast
    and reduce SAR (Specific Absorption Rate).

    This is a convenience wrapper around :func:`se` with variable flip angles.

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
    cpmg : bool, optional
        If True (default), use CPMG phase cycling.

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
    refocus_deg = 180.0 if refocus_angles_deg is None else refocus_angles_deg

    return se(
        t2_ms=t2_ms,
        t1_ms=t1_ms,
        te_ms=te_ms,
        n_echoes=etl,
        refocus_deg=refocus_deg,
        b1=b1,
        cpmg=cpmg,
    )
