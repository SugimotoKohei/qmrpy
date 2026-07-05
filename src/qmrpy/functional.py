from __future__ import annotations

from typing import Any

from numpy.typing import ArrayLike

from qmrpy.core import FitResult
from qmrpy.models import (
    B0DualEcho,
    B0MultiEcho,
    MRFDictionary,
    MTR,
    MTsat,
    T1DESPOT1HIFI,
    T1InversionRecovery,
    T1MP2RAGE,
    T1Rho,
    T1VFA,
    T2DECAESMap,
    T2EMC,
    T2EPG,
    T2Mono,
    T2MultiComponent,
    T2WaterFat,
    T2StarComplexR2,
    T2StarMonoR2,
)

MODEL_REGISTRY: dict[str, type[Any]] = {
    "t1_vfa": T1VFA,
    "t1_inversion_recovery": T1InversionRecovery,
    "t1_despot1_hifi": T1DESPOT1HIFI,
    "t1_mp2rage": T1MP2RAGE,
    "t1rho": T1Rho,
    "t2_mono": T2Mono,
    "t2_epg": T2EPG,
    "t2_emc": T2EMC,
    "t2_decaes_map": T2DECAESMap,
    "t2_multi_component": T2MultiComponent,
    "t2_water_fat": T2WaterFat,
    "t2star_mono_r2": T2StarMonoR2,
    "t2star_complex_r2": T2StarComplexR2,
    "b0_dual_echo": B0DualEcho,
    "b0_multi_echo": B0MultiEcho,
    "mrf_dictionary": MRFDictionary,
    "mtr": MTR,
    "mtsat": MTsat,
}


def _build_model(model_id: str, **kwargs: Any) -> Any:
    model_cls = MODEL_REGISTRY[model_id]
    return model_cls(**kwargs)


def simulate_t1_vfa(
    *,
    m0: float,
    t1_ms: float,
    flip_angle_deg: ArrayLike,
    tr_ms: float,
    b1: ArrayLike | float = 1.0,
) -> Any:
    return _build_model("t1_vfa", flip_angle_deg=flip_angle_deg, tr_ms=tr_ms, b1=b1).forward(
        m0=m0, t1_ms=t1_ms
    )


def fit_t1_vfa(
    signal: ArrayLike,
    *,
    flip_angle_deg: ArrayLike,
    tr_ms: float,
    b1: ArrayLike | float = 1.0,
    **kwargs: Any,
) -> FitResult:
    return _build_model("t1_vfa", flip_angle_deg=flip_angle_deg, tr_ms=tr_ms, b1=b1).fit(
        signal, **kwargs
    )


def fit_t1_vfa_linear(
    signal: ArrayLike,
    *,
    flip_angle_deg: ArrayLike,
    tr_ms: float,
    b1: ArrayLike | float = 1.0,
    **kwargs: Any,
) -> FitResult:
    return _build_model("t1_vfa", flip_angle_deg=flip_angle_deg, tr_ms=tr_ms, b1=b1).fit_linear(
        signal, **kwargs
    )


def simulate_t1_inversion_recovery(
    *,
    t1_ms: float,
    ra: float,
    rb: float,
    ti_ms: ArrayLike,
    magnitude: bool = False,
) -> Any:
    return _build_model("t1_inversion_recovery", ti_ms=ti_ms).forward(
        t1_ms=t1_ms, ra=ra, rb=rb, magnitude=magnitude
    )


def fit_t1_inversion_recovery(
    signal: ArrayLike,
    *,
    ti_ms: ArrayLike,
    **kwargs: Any,
) -> FitResult:
    return _build_model("t1_inversion_recovery", ti_ms=ti_ms).fit(signal, **kwargs)


def fit_t1_despot1_hifi(
    signal: ArrayLike,
    *,
    flip_angle_deg: ArrayLike,
    tr_ms: float,
    **kwargs: Any,
) -> FitResult:
    return _build_model("t1_despot1_hifi", flip_angle_deg=flip_angle_deg, tr_ms=tr_ms).fit(
        signal, **kwargs
    )


def fit_t1_mp2rage(
    signal: ArrayLike,
    *,
    ti1_ms: float,
    ti2_ms: float,
    alpha1_deg: float,
    alpha2_deg: float,
    **kwargs: Any,
) -> FitResult:
    model = _build_model(
        "t1_mp2rage",
        ti1_ms=ti1_ms,
        ti2_ms=ti2_ms,
        alpha1_deg=alpha1_deg,
        alpha2_deg=alpha2_deg,
    )
    return model.fit(signal, **kwargs)


def simulate_t1rho(
    *,
    m0: float,
    t1rho_ms: float,
    tsl_ms: ArrayLike,
) -> Any:
    return _build_model("t1rho", tsl_ms=tsl_ms).forward(m0=m0, t1rho_ms=t1rho_ms)


def fit_t1rho(
    signal: ArrayLike,
    *,
    tsl_ms: ArrayLike,
    **kwargs: Any,
) -> FitResult:
    return _build_model("t1rho", tsl_ms=tsl_ms).fit(signal, **kwargs)


def simulate_t2_mono(
    *,
    m0: float,
    t2_ms: float,
    te_ms: ArrayLike,
) -> Any:
    return _build_model("t2_mono", te_ms=te_ms).forward(m0=m0, t2_ms=t2_ms)


def fit_t2_mono(
    signal: ArrayLike,
    *,
    te_ms: ArrayLike,
    **kwargs: Any,
) -> FitResult:
    return _build_model("t2_mono", te_ms=te_ms).fit(signal, **kwargs)


def simulate_t2_epg(
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
) -> Any:
    return _build_model(
        "t2_epg",
        n_te=n_te,
        te_ms=te_ms,
        t1_ms=t1_ms,
        alpha_deg=alpha_deg,
        beta_deg=beta_deg,
        epg_backend=epg_backend,
    ).forward(m0=m0, t2_ms=t2_ms, offset=offset, b1=b1)


def fit_t2_epg(
    signal: ArrayLike,
    *,
    n_te: int,
    te_ms: float,
    t1_ms: float = 1000.0,
    alpha_deg: float = 180.0,
    beta_deg: float = 180.0,
    b1: float | None = None,
    epg_backend: str = "decaes",
    **kwargs: Any,
) -> FitResult:
    model = _build_model(
        "t2_epg",
        n_te=n_te,
        te_ms=te_ms,
        t1_ms=t1_ms,
        alpha_deg=alpha_deg,
        beta_deg=beta_deg,
        epg_backend=epg_backend,
    )
    return model.fit(signal, b1=b1, **kwargs)


def fit_t2_emc(
    signal: ArrayLike,
    *,
    n_te: int,
    te_ms: float,
    **kwargs: Any,
) -> FitResult:
    return _build_model("t2_emc", n_te=n_te, te_ms=te_ms).fit(signal, **kwargs)


def fit_t2_multi_component(
    signal: ArrayLike,
    *,
    te_ms: ArrayLike,
    **kwargs: Any,
) -> FitResult:
    return _build_model("t2_multi_component", te_ms=te_ms).fit(signal, **kwargs)


def fit_t2_decaes_map(
    signal: ArrayLike,
    *,
    n_te: int,
    te_ms: float,
    n_t2: int,
    t2_range_ms: tuple[float, float],
    reg: str,
    **kwargs: Any,
) -> FitResult:
    model = _build_model(
        "t2_decaes_map",
        n_te=n_te,
        te_ms=te_ms,
        n_t2=n_t2,
        t2_range_ms=t2_range_ms,
        reg=reg,
        **kwargs,
    )
    return model.fit(signal)


def simulate_t2_water_fat(
    *,
    te_ms: ArrayLike,
    water_amplitude: float,
    fat_amplitude: float,
    water_t2_ms: float,
    fat_t2_ms: float,
) -> Any:
    return _build_model("t2_water_fat", te_ms=te_ms).forward(
        water_amplitude=water_amplitude,
        fat_amplitude=fat_amplitude,
        water_t2_ms=water_t2_ms,
        fat_t2_ms=fat_t2_ms,
    )


def fit_t2_water_fat(
    signal: ArrayLike,
    *,
    te_ms: ArrayLike,
    water_t2_grid_ms: ArrayLike,
    fat_t2_grid_ms: ArrayLike,
) -> FitResult:
    return _build_model("t2_water_fat", te_ms=te_ms).fit(
        signal,
        water_t2_grid_ms=water_t2_grid_ms,
        fat_t2_grid_ms=fat_t2_grid_ms,
    )


def fit_t2star_mono_r2(
    signal: ArrayLike,
    *,
    te_ms: ArrayLike,
    **kwargs: Any,
) -> FitResult:
    return _build_model("t2star_mono_r2", te_ms=te_ms).fit(signal, **kwargs)


def fit_t2star_complex_r2(
    signal: ArrayLike,
    *,
    te_ms: ArrayLike,
) -> FitResult:
    return _build_model("t2star_complex_r2", te_ms=te_ms).fit(signal)


def fit_b0_dual_echo(
    signal: ArrayLike,
    *,
    te1_ms: float,
    te2_ms: float,
    unwrap_phase: bool = False,
) -> FitResult:
    return _build_model(
        "b0_dual_echo",
        te1_ms=te1_ms,
        te2_ms=te2_ms,
        unwrap_phase=unwrap_phase,
    ).fit(signal)


def fit_b0_multi_echo(
    signal: ArrayLike,
    *,
    te_ms: ArrayLike,
    unwrap_phase: bool = True,
) -> FitResult:
    return _build_model("b0_multi_echo", te_ms=te_ms, unwrap_phase=unwrap_phase).fit(signal)


def simulate_mtr(
    *,
    s0: float,
    mtr: float,
) -> Any:
    return _build_model("mtr").forward(s0=s0, mtr=mtr)


def fit_mtr(signal: ArrayLike) -> FitResult:
    return _build_model("mtr").fit(signal)


def simulate_mtsat(
    *,
    m0: float,
    t1_ms: float,
    mtsat: float,
    flip_angle_deg: float,
    tr_ms: float,
) -> Any:
    return _build_model("mtsat", flip_angle_deg=flip_angle_deg, tr_ms=tr_ms).forward(
        m0=m0,
        t1_ms=t1_ms,
        mtsat=mtsat,
    )


def fit_mtsat(
    signal: ArrayLike,
    *,
    flip_angle_deg: float,
    tr_ms: float,
    m0: float,
    t1_ms: float,
) -> FitResult:
    return _build_model("mtsat", flip_angle_deg=flip_angle_deg, tr_ms=tr_ms).fit(
        signal,
        m0=m0,
        t1_ms=t1_ms,
    )


def simulate_mrf_dictionary(
    *,
    m0: float,
    t1_ms: float,
    t2_ms: float,
    flip_angle_deg: ArrayLike,
    tr_ms: ArrayLike,
    te_ms: ArrayLike | float | None = None,
    b1: float = 1.0,
) -> Any:
    return _build_model("mrf_dictionary", flip_angle_deg=flip_angle_deg, tr_ms=tr_ms, te_ms=te_ms).forward(
        m0=m0,
        t1_ms=t1_ms,
        t2_ms=t2_ms,
        b1=b1,
    )


def fit_mrf_dictionary(
    signal: ArrayLike,
    *,
    flip_angle_deg: ArrayLike,
    tr_ms: ArrayLike,
    t1_grid_ms: ArrayLike,
    t2_grid_ms: ArrayLike,
    te_ms: ArrayLike | float | None = None,
    b1: float = 1.0,
) -> FitResult:
    return _build_model("mrf_dictionary", flip_angle_deg=flip_angle_deg, tr_ms=tr_ms, te_ms=te_ms).fit(
        signal,
        t1_grid_ms=t1_grid_ms,
        t2_grid_ms=t2_grid_ms,
        b1=b1,
    )
