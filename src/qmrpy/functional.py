from __future__ import annotations

from typing import Any


def vfa_t1_forward(
    *, m0: float, t1_ms: float, flip_angle_deg: Any, tr_ms: float, b1: Any | float = 1.0
):
    """Functional wrapper for VFA T1 forward model (ms).

    Parameters
    ----------
    m0 : float
        Proton density scale.
    t1_ms : float
        T1 in milliseconds.
    flip_angle_deg : array-like
        Flip angles in degrees.
    tr_ms : float
        Repetition time in milliseconds.
    b1 : float or array-like, optional
        B1 scaling (scalar or array).

    Returns
    -------
    ndarray
        Simulated SPGR signal array.
    """
    from qmrpy.models.t1.vfa_t1 import VfaT1

    return VfaT1(flip_angle_deg=flip_angle_deg, tr_ms=tr_ms, b1=b1).forward(m0=m0, t1_ms=t1_ms)


def vfa_t1_fit(
    signal: Any,
    *,
    flip_angle_deg: Any,
    tr_ms: float,
    b1: Any | float = 1.0,
    **kwargs: Any,
) -> dict[str, float]:
    """Functional wrapper for VFA T1 fit (ms).

    Parameters
    ----------
    signal : array-like
        Observed signal array.
    flip_angle_deg : array-like
        Flip angles in degrees.
    tr_ms : float
        Repetition time in milliseconds.
    b1 : float or array-like, optional
        B1 scaling (scalar or array).
    **kwargs
        Passed to ``VfaT1.fit``.

    Returns
    -------
    dict
        Fit results dict (e.g., ``t1_ms``, ``m0``, ``n_points``).
    """
    from qmrpy.models.t1.vfa_t1 import VfaT1

    return VfaT1(flip_angle_deg=flip_angle_deg, tr_ms=tr_ms, b1=b1).fit(signal, **kwargs)


def vfa_t1_fit_linear(
    signal: Any,
    *,
    flip_angle_deg: Any,
    tr_ms: float,
    b1: Any | float = 1.0,
    **kwargs: Any,
) -> dict[str, float]:
    """Functional wrapper for VFA T1 linear fit (ms).

    Parameters
    ----------
    signal : array-like
        Observed signal array.
    flip_angle_deg : array-like
        Flip angles in degrees.
    tr_ms : float
        Repetition time in milliseconds.
    b1 : float or array-like, optional
        B1 scaling (scalar or array).
    **kwargs
        Passed to ``VfaT1.fit_linear``.

    Returns
    -------
    dict
        Fit results dict (e.g., ``t1_ms``, ``m0``, ``n_points``).
    """
    from qmrpy.models.t1.vfa_t1 import VfaT1

    return VfaT1(flip_angle_deg=flip_angle_deg, tr_ms=tr_ms, b1=b1).fit_linear(signal, **kwargs)


def inversion_recovery_forward(
    *, t1_ms: float, ra: float, rb: float, ti_ms: Any, magnitude: bool = False
):
    """Functional wrapper for inversion recovery forward model (ms).

    Parameters
    ----------
    t1_ms : float
        T1 in milliseconds.
    ra : float
        Offset term.
    rb : float
        Amplitude term.
    ti_ms : array-like
        Inversion times in milliseconds.
    magnitude : bool, optional
        If True, return magnitude signal.

    Returns
    -------
    ndarray
        Simulated inversion recovery signal.
    """
    from qmrpy.models.t1.inversion_recovery import InversionRecovery

    return InversionRecovery(ti_ms=ti_ms).forward(t1_ms=t1_ms, ra=ra, rb=rb, magnitude=magnitude)


def inversion_recovery_fit(
    signal: Any, *, ti_ms: Any, **kwargs: Any
) -> dict[str, float]:
    """Functional wrapper for inversion recovery fit (ms).

    Parameters
    ----------
    signal : array-like
        Observed signal array.
    ti_ms : array-like
        Inversion times in milliseconds.
    **kwargs
        Passed to ``InversionRecovery.fit``.

    Returns
    -------
    dict
        Fit results dict (e.g., ``t1_ms``, ``ra``, ``rb``).
    """
    from qmrpy.models.t1.inversion_recovery import InversionRecovery

    return InversionRecovery(ti_ms=ti_ms).fit(signal, **kwargs)


def mono_t2_forward(*, m0: float, t2_ms: float, te_ms: Any):
    """Functional wrapper for mono-exponential T2 forward model (ms).

    Parameters
    ----------
    m0 : float
        Proton density scale.
    t2_ms : float
        T2 in milliseconds.
    te_ms : array-like
        Echo times in milliseconds.

    Returns
    -------
    ndarray
        Simulated mono-exponential signal array.
    """
    from qmrpy.models.t2.mono_t2 import MonoT2

    return MonoT2(te_ms=te_ms).forward(m0=m0, t2_ms=t2_ms)


def mono_t2_fit(signal: Any, *, te_ms: Any, **kwargs: Any) -> dict[str, float]:
    """Functional wrapper for mono-exponential T2 fit (ms).

    Parameters
    ----------
    signal : array-like
        Observed signal array.
    te_ms : array-like
        Echo times in milliseconds.
    **kwargs
        Passed to ``MonoT2.fit``.

    Returns
    -------
    dict
        Fit results dict (e.g., ``t2_ms``, ``m0``).
    """
    from qmrpy.models.t2.mono_t2 import MonoT2

    return MonoT2(te_ms=te_ms).fit(signal, **kwargs)


def epg_t2_forward(
    *,
    m0: float,
    t2_ms: float,
    n_te: int,
    te_ms: float,
    t1_ms: float = 1000.0,
    alpha_deg: float = 180.0,
    beta_deg: float = 180.0,
    b1: float | None = None,
    offset: float = 0.0,
    epg_backend: str = "decaes",
):
    """Functional wrapper for EPG-corrected T2 forward model (ms).

    Parameters
    ----------
    m0 : float
        Proton density scale.
    t2_ms : float
        T2 in milliseconds.
    n_te : int
        Number of echoes.
    te_ms : float
        Echo spacing in milliseconds.
    t1_ms : float, optional
        T1 in milliseconds.
    alpha_deg : float, optional
        Refocusing flip angle in degrees.
    beta_deg : float, optional
        Refocusing phase angle in degrees.
    b1 : float, optional
        B1 scaling factor.
    offset : float, optional
        Constant offset term.
    epg_backend : {"decaes"}, optional
        Backend for EPG decay computation.

    Returns
    -------
    ndarray
        Simulated multi-echo signal array.
    """
    from qmrpy.models.t2.epg_t2 import EpgT2

    return EpgT2(
        n_te=n_te,
        te_ms=te_ms,
        t1_ms=t1_ms,
        alpha_deg=alpha_deg,
        beta_deg=beta_deg,
        epg_backend=epg_backend,
    ).forward(m0=m0, t2_ms=t2_ms, offset=offset, b1=b1)


def epg_t2_fit(
    signal: Any,
    *,
    n_te: int,
    te_ms: float,
    t1_ms: float = 1000.0,
    alpha_deg: float = 180.0,
    beta_deg: float = 180.0,
    b1: float | None = None,
    epg_backend: str = "decaes",
    **kwargs: Any,
) -> dict[str, float]:
    """Functional wrapper for EPG-corrected T2 fit (ms).

    Parameters
    ----------
    signal : array-like
        Observed signal array (length ``n_te``).
    n_te : int
        Number of echoes.
    te_ms : float
        Echo spacing in milliseconds.
    t1_ms : float, optional
        T1 in milliseconds.
    alpha_deg : float, optional
        Refocusing flip angle in degrees.
    beta_deg : float, optional
        Refocusing phase angle in degrees.
    b1 : float, optional
        B1 scaling factor.
    epg_backend : {"decaes"}, optional
        Backend for EPG decay computation.
    **kwargs
        Passed to ``EpgT2.fit``.

    Returns
    -------
    dict
        Fit results dict (e.g., ``t2_ms``, ``m0``).
    """
    from qmrpy.models.t2.epg_t2 import EpgT2

    model = EpgT2(
        n_te=n_te,
        te_ms=te_ms,
        t1_ms=t1_ms,
        alpha_deg=alpha_deg,
        beta_deg=beta_deg,
        epg_backend=epg_backend,
    )
    return model.fit(signal, b1=b1, **kwargs)


def mwf_fit(signal: Any, *, te_ms: Any, **kwargs: Any) -> dict[str, Any]:
    """Functional wrapper for multi-component T2 (MWF) fit (ms).

    Parameters
    ----------
    signal : array-like
        Observed signal array.
    te_ms : array-like
        Echo times in milliseconds.
    **kwargs
        Passed to ``MultiComponentT2.fit``.

    Returns
    -------
    dict
        Fit results dict including spectrum and MWF metrics.
    """
    from qmrpy.models.t2.mwf import MultiComponentT2

    return MultiComponentT2(te_ms=te_ms).fit(signal, **kwargs)


def decaes_t2map_fit(
    signal: Any,
    *,
    n_te: int,
    te_ms: float,
    n_t2: int,
    t2_range_ms: tuple[float, float],
    reg: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Functional wrapper for DECAES T2 map fit (ms).

    Parameters
    ----------
    signal : array-like
        Observed signal array (length ``n_te``).
    n_te : int
        Number of echoes.
    te_ms : float
        Echo spacing in milliseconds.
    n_t2 : int
        Number of T2 components.
    t2_range_ms : tuple of float
        T2 range in milliseconds.
    reg : str
        Regularization method (``none``, ``lcurve``, ``gcv``, ``chi2``, ``mdp``).
    **kwargs
        Passed to ``DecaesT2Map``.

    Returns
    -------
    dict
        Fit results dict including spectrum and diagnostics.
    """
    from qmrpy.models.t2.decaes_t2 import DecaesT2Map

    model = DecaesT2Map(
        n_te=n_te,
        te_ms=te_ms,
        n_t2=n_t2,
        t2_range_ms=t2_range_ms,
        reg=reg,
        **kwargs,
    )
    return model.fit(signal)


def decaes_t2map_spectrum(
    signal: Any,
    *,
    n_te: int,
    te_ms: float,
    n_t2: int,
    t2_range_ms: tuple[float, float],
    reg: str,
    **kwargs: Any,
) -> Any:
    """Return T2 spectrum (distribution) from DECAES T2 map fit (ms).

    Parameters
    ----------
    signal : array-like
        Observed signal array (length ``n_te``).
    n_te : int
        Number of echoes.
    te_ms : float
        Echo spacing in milliseconds.
    n_t2 : int
        Number of T2 components.
    t2_range_ms : tuple of float
        T2 range in milliseconds.
    reg : str
        Regularization method (``none``, ``lcurve``, ``gcv``, ``chi2``, ``mdp``).
    **kwargs
        Passed to ``DecaesT2Map``.

    Returns
    -------
    ndarray
        T2 distribution array.
    """
    fit = decaes_t2map_fit(
        signal,
        n_te=n_te,
        te_ms=te_ms,
        n_t2=n_t2,
        t2_range_ms=t2_range_ms,
        reg=reg,
        **kwargs,
    )
    return fit.get("distribution")
