from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .simulation import SimulationProtocol


def _require_pypulseq() -> Any:
    try:
        from pypulseq.Sequence.sequence import Sequence  # noqa: F401
        from pypulseq.calc_duration import calc_duration  # noqa: F401
        from pypulseq.make_adc import make_adc  # noqa: F401
        from pypulseq.make_delay import make_delay  # noqa: F401
        from pypulseq.make_sinc_pulse import make_sinc_pulse  # noqa: F401
        from pypulseq.make_trapezoid import make_trapezoid  # noqa: F401
        from pypulseq.opts import Opts  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("pypulseq が必要です。`uv add pypulseq` を実行してください。") from exc

    from pypulseq.Sequence.sequence import Sequence
    from pypulseq.calc_duration import calc_duration
    from pypulseq.make_adc import make_adc
    from pypulseq.make_delay import make_delay
    from pypulseq.make_sinc_pulse import make_sinc_pulse
    from pypulseq.make_trapezoid import make_trapezoid
    from pypulseq.opts import Opts

    return Sequence, calc_duration, make_adc, make_delay, make_sinc_pulse, make_trapezoid, Opts


def _default_system() -> Any:
    _, _, _, _, _, _, Opts = _require_pypulseq()
    return Opts(
        max_grad=28,
        grad_unit="mT/m",
        max_slew=150,
        slew_unit="T/m/s",
        rf_ringdown_time=30e-6,
        rf_dead_time=100e-6,
        adc_dead_time=10e-6,
    )


def _quantize(t: float, raster: float) -> float:
    import numpy as np

    return float(np.round(float(t) / float(raster)) * float(raster))


def _format_token(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def _default_seq_path(name: str) -> Path:
    out_dir = Path("output/seq")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{name}.seq"


def _write_sequence(seq: Any, path: str | Path | None, *, fallback_name: str) -> str:
    seq_path = _default_seq_path(fallback_name) if path is None else Path(path)
    seq_path.parent.mkdir(parents=True, exist_ok=True)
    seq.write(str(seq_path))
    return str(seq_path)

def build_cpmg_sequence(
    *,
    te_ms: float = 10.0,
    n_echo: int = 16,
    adc_samples: int = 1,
    refoc_angle_deg: float = 180.0,
    slice_thickness_mm: float = 5.0,
    fov_mm: float = 200.0,
    matrix: tuple[int, int] = (128, 128),
    readout_bw_hz_per_pixel: float | None = 300.0,
    phase_encode_duration_ms: float = 1.0,
    readout_prephaser_duration_ms: float = 1.0,
    phase_encode_mode: str = "cpmg",
    phase_encode_index: int = 0,
    repeat_phase_encodes: bool = False,
    auto_adc_samples: bool = True,
    rf_duration_ms: float = 3.0,
    adc_duration_ms: float = 3.2,
    apodization: float = 0.5,
    time_bw_product: float = 4.0,
    include_readout: bool = False,
    system: Any | None = None,
) -> Any:
    """Build a standard CPMG sequence using pypulseq.

    If include_readout is True, add phase-encode (Gy) and readout (Gx) gradients
    suitable for 2D imaging. readout_bw_hz_per_pixel is interpreted as Hz/pixel;
    ADC duration is derived as 1 / readout_bw_hz_per_pixel.

    phase_encode_mode:
      - "cpmg": fixed phase-encode amplitude for all echoes (use phase_encode_index).
      - "tse":  vary phase-encode amplitude across echoes (echo train sampling).
    repeat_phase_encodes:
      - If True, repeat the echo train for every phase-encode line (TR loop).
    """
    Sequence, calc_duration, make_adc, make_delay, make_sinc_pulse, make_trapezoid, _ = _require_pypulseq()
    if n_echo <= 0:
        raise ValueError("n_echo must be >= 1")
    if te_ms <= 0:
        raise ValueError("te_ms must be > 0")
    if adc_samples <= 0:
        raise ValueError("adc_samples must be >= 1")
    if fov_mm <= 0:
        raise ValueError("fov_mm must be > 0")
    if phase_encode_duration_ms <= 0 or readout_prephaser_duration_ms <= 0:
        raise ValueError("phase_encode_duration_ms/readout_prephaser_duration_ms must be > 0")

    system = _default_system() if system is None else system
    seq = Sequence(system)

    te = float(te_ms) * 1e-3
    slice_thickness = float(slice_thickness_mm) * 1e-3
    rf_duration = float(rf_duration_ms) * 1e-3
    adc_duration = float(adc_duration_ms) * 1e-3

    rf90, gz90, _ = make_sinc_pulse(
        flip_angle=float(90.0) * (3.141592653589793 / 180.0),
        duration=rf_duration,
        phase_offset=0.0,
        slice_thickness=slice_thickness,
        apodization=float(apodization),
        time_bw_product=float(time_bw_product),
        system=system,
        return_gz=True,
    )
    gz90_rephaser = make_trapezoid(channel="z", area=-gz90.area / 2, duration=1e-3, system=system)

    rf180, gz180, _ = make_sinc_pulse(
        flip_angle=float(refoc_angle_deg) * (3.141592653589793 / 180.0),
        duration=rf_duration,
        phase_offset=3.141592653589793 / 2.0,
        slice_thickness=slice_thickness,
        apodization=float(apodization),
        time_bw_product=float(time_bw_product),
        system=system,
        return_gz=True,
    )

    matrix_x, matrix_y = int(matrix[0]), int(matrix[1])
    if matrix_x <= 0 or matrix_y <= 0:
        raise ValueError("matrix must be positive")

    raster = system.block_duration_raster
    if include_readout:
        if auto_adc_samples and adc_samples <= 1:
            adc_samples = matrix_x
        if readout_bw_hz_per_pixel is None:
            adc_duration = float(adc_duration_ms) * 1e-3
        else:
            bw_pp = float(readout_bw_hz_per_pixel)
            if bw_pp <= 0:
                raise ValueError("readout_bw_hz_per_pixel must be > 0")
            adc_duration = 1.0 / bw_pp
        adc_duration = _quantize(adc_duration, raster)

    adc = make_adc(num_samples=int(adc_samples), duration=adc_duration, system=system, phase_offset=0.0)

    gx_pre = gx = gx_reph = None
    ky_values = None
    pre_dur = readout_dur = reph_dur = 0.0
    pe_dur = 0.0
    if include_readout:
        fov_m = float(fov_mm) * 1e-3
        delta_k = 1.0 / fov_m
        k_width = float(matrix_x) * delta_k
        ky_values = (np.arange(matrix_y, dtype=np.float64) - (matrix_y / 2.0)) * delta_k

        pre_dur = _quantize(float(readout_prephaser_duration_ms) * 1e-3, raster)
        pe_dur = _quantize(float(phase_encode_duration_ms) * 1e-3, raster)
        gx_pre = make_trapezoid(channel="x", area=-k_width / 2.0, duration=pre_dur, system=system)
        gx = make_trapezoid(channel="x", flat_area=k_width, flat_time=adc_duration, system=system)
        gx_reph = make_trapezoid(channel="x", area=-k_width / 2.0, duration=pre_dur, system=system)

        pre_dur = max(calc_duration(gx_pre), pe_dur)
        readout_dur = max(calc_duration(gx), calc_duration(adc))
        reph_dur = max(calc_duration(gx_reph), pe_dur)

    tau = te / 2.0
    rf90_dur = calc_duration(rf90, gz90)
    gz90_reph_dur = calc_duration(gz90_rephaser)
    rf180_dur = calc_duration(rf180, gz180)
    adc_dur = calc_duration(adc)

    seq.add_block(rf90, gz90)
    seq.add_block(gz90_rephaser)

    delay1 = tau - (rf90_dur / 2) - gz90_reph_dur - (rf180_dur / 2)
    if delay1 < 0:
        raise ValueError("te_ms is too short for the chosen RF/ADC durations")
    if delay1 > 0:
        seq.add_block(make_delay(_quantize(delay1, raster)))

    if include_readout and repeat_phase_encodes:
        phase_indices = list(range(len(ky_values))) if ky_values is not None else [0]
    else:
        phase_indices = [int(phase_encode_index)]

    mode = str(phase_encode_mode).lower().strip()
    if mode not in {"cpmg", "tse"}:
        raise ValueError("phase_encode_mode must be 'cpmg' or 'tse'")

    for pe_idx in phase_indices:
        for echo_idx in range(int(n_echo)):
            seq.add_block(rf180, gz180)
            if include_readout:
                delay2 = tau - (rf180_dur / 2) - pre_dur - (readout_dur / 2)
            else:
                delay2 = tau - (rf180_dur / 2) - (adc_dur / 2)
            if delay2 < 0:
                raise ValueError("te_ms is too short for the chosen RF/ADC durations")
            if delay2 > 0:
                seq.add_block(make_delay(_quantize(delay2, raster)))

            if include_readout:
                assert gx_pre is not None
                assert gx is not None
                assert gx_reph is not None
                assert ky_values is not None
                if mode == "cpmg":
                    if pe_idx < 0 or pe_idx >= len(ky_values):
                        raise ValueError("phase_encode_index out of range")
                    ky = float(ky_values[pe_idx])
                else:
                    ky = float(ky_values[(echo_idx + pe_idx) % len(ky_values)])
                gy_pre = make_trapezoid(channel="y", area=ky, duration=pe_dur, system=system)
                gy_reph = make_trapezoid(channel="y", area=-ky, duration=pe_dur, system=system)
                seq.add_block(gx_pre, gy_pre)
                seq.add_block(gx, adc)
                seq.add_block(gx_reph, gy_reph)
                delay3 = tau - (readout_dur / 2) - reph_dur - (rf180_dur / 2)
            else:
                seq.add_block(adc)
                delay3 = tau - (adc_dur / 2) - (rf180_dur / 2)
            if delay3 < 0:
                raise ValueError("te_ms is too short for the chosen RF/ADC durations")
            if delay3 > 0:
                seq.add_block(make_delay(_quantize(delay3, raster)))

    return seq


def build_se_sequence(
    *,
    te_ms: float = 10.0,
    adc_samples: int = 1,
    system: Any | None = None,
) -> Any:
    """Standard spin-echo (SE) sequence template."""
    return build_cpmg_sequence(
        te_ms=te_ms,
        n_echo=1,
        adc_samples=adc_samples,
        refoc_angle_deg=180.0,
        system=system,
    )


def build_spgr_sequence(
    *,
    flip_angle_deg: float = 15.0,
    tr_ms: float = 15.0,
    adc_samples: int = 1,
    n_reps: int = 200,
    slice_thickness_mm: float = 5.0,
    rf_duration_ms: float = 1.0,
    adc_duration_ms: float = 3.2,
    apodization: float = 0.5,
    time_bw_product: float = 4.0,
    system: Any | None = None,
) -> Any:
    """Standard SPGR sequence template."""
    Sequence, calc_duration, make_adc, make_delay, make_sinc_pulse, make_trapezoid, _ = _require_pypulseq()
    if tr_ms <= 0:
        raise ValueError("tr_ms must be > 0")
    if n_reps <= 0:
        raise ValueError("n_reps must be >= 1")

    system = _default_system() if system is None else system
    seq = Sequence(system)

    tr = float(tr_ms) * 1e-3
    slice_thickness = float(slice_thickness_mm) * 1e-3
    rf_duration = float(rf_duration_ms) * 1e-3
    adc_duration = float(adc_duration_ms) * 1e-3

    rf, gz, _ = make_sinc_pulse(
        flip_angle=float(flip_angle_deg) * (3.141592653589793 / 180.0),
        duration=rf_duration,
        time_bw_product=float(time_bw_product),
        system=system,
        return_gz=True,
        slice_thickness=slice_thickness,
        apodization=float(apodization),
    )
    gz_reph = make_trapezoid(channel="z", area=-gz.area / 2, duration=1e-3, system=system)
    adc = make_adc(num_samples=int(adc_samples), duration=adc_duration, system=system)
    gz_spoil = make_trapezoid(channel="z", area=10.0 / slice_thickness, duration=2e-3, system=system)

    rf_dur = calc_duration(rf, gz)
    gz_reph_dur = calc_duration(gz_reph)
    adc_dur = calc_duration(adc)
    spoil_dur = calc_duration(gz_spoil)
    min_dur = rf_dur + gz_reph_dur + adc_dur + spoil_dur
    if tr < min_dur:
        raise ValueError("tr_ms is too short for the chosen RF/ADC durations")
    delay = make_delay(_quantize(tr - min_dur, system.block_duration_raster))

    for _ in range(int(n_reps)):
        seq.add_block(rf, gz)
        seq.add_block(gz_reph)
        seq.add_block(adc)
        seq.add_block(gz_spoil)
        seq.add_block(delay)

    return seq


def mrzero_single_voxel_data_factory(
    params: Mapping[str, float],
    *,
    size_mm: tuple[float, float, float] = (5.0, 5.0, 5.0),
) -> Any:
    """Default SimData factory for MRzero Bloch simulations (single voxel)."""
    from .mrzero import _import_mrzero

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("MRzero の SimData には torch が必要です。") from exc

    mr0 = _import_mrzero()

    m0 = float(params.get("m0", 1.0))
    t1_ms = float(params.get("t1_ms", 1000.0))
    t2_ms = float(params.get("t2_ms", 100.0))
    t2dash_ms = float(params.get("t2dash_ms", t2_ms))
    b1 = float(params.get("b1", 1.0))
    b0 = float(params.get("b0", 0.0))

    pd = torch.ones((1,)) * m0
    t1 = torch.ones((1,)) * (t1_ms * 1e-3)
    t2 = torch.ones((1,)) * (t2_ms * 1e-3)
    t2dash = torch.ones((1,)) * (t2dash_ms * 1e-3)
    d = torch.zeros((1, 3))
    b0_t = torch.ones((1,)) * b0
    b1_t = torch.ones((1, 1)) * b1
    coil_sens = torch.ones((1, 1))
    size = torch.tensor([s * 1e-3 for s in size_mm])
    voxel_pos = torch.zeros((1, 3))
    nyquist = torch.tensor([1, 1, 1])
    dephasing_func = lambda b0_in, t: torch.zeros_like(b0_in)

    return mr0.SimData(pd, t1, t2, t2dash, d, b0_t, b1_t, coil_sens, size, voxel_pos, nyquist, dephasing_func)


def mrzero_protocol_se(
    *,
    te_ms: float = 10.0,
    adc_samples: int = 1,
    spin_count: int = 5000,
    perfect_spoiling: bool = False,
    print_progress: bool = True,
    data_factory: Callable[[Mapping[str, float]], Any] = mrzero_single_voxel_data_factory,
    seq_path: str | Path | None = None,
) -> SimulationProtocol:
    """Standard MRzero SE protocol template."""
    seq = build_se_sequence(te_ms=te_ms, adc_samples=adc_samples)
    seq_or_path = _write_sequence(
        seq,
        seq_path,
        fallback_name=f"se_te{_format_token(te_ms)}",
    )
    return SimulationProtocol(
        simulation_backend="mrzero_bloch",
        fit=True,
        model_protocol={
            "model": {"te_ms": [float(te_ms)]},
            "mrzero": {
                "seq_or_path": seq_or_path,
                "data_factory": data_factory,
                "spin_count": int(spin_count),
                "perfect_spoiling": bool(perfect_spoiling),
                "print_progress": bool(print_progress),
            },
        },
    )


def mrzero_protocol_cpmg(
    *,
    te_ms: float = 10.0,
    n_echo: int = 16,
    adc_samples: int = 1,
    refoc_angle_deg: float = 180.0,
    slice_thickness_mm: float = 5.0,
    fov_mm: float = 200.0,
    matrix: tuple[int, int] = (128, 128),
    readout_bw_hz_per_pixel: float | None = 300.0,
    phase_encode_duration_ms: float = 1.0,
    readout_prephaser_duration_ms: float = 1.0,
    include_readout: bool = False,
    phase_encode_mode: str = "cpmg",
    phase_encode_index: int = 0,
    repeat_phase_encodes: bool = False,
    spin_count: int = 5000,
    perfect_spoiling: bool = False,
    print_progress: bool = True,
    data_factory: Callable[[Mapping[str, float]], Any] = mrzero_single_voxel_data_factory,
    seq_path: str | Path | None = None,
) -> SimulationProtocol:
    """Standard MRzero CPMG protocol template."""
    seq = build_cpmg_sequence(
        te_ms=te_ms,
        n_echo=n_echo,
        adc_samples=adc_samples,
        refoc_angle_deg=refoc_angle_deg,
        slice_thickness_mm=slice_thickness_mm,
        fov_mm=fov_mm,
        matrix=matrix,
        readout_bw_hz_per_pixel=readout_bw_hz_per_pixel,
        phase_encode_duration_ms=phase_encode_duration_ms,
        readout_prephaser_duration_ms=readout_prephaser_duration_ms,
        include_readout=include_readout,
        phase_encode_mode=phase_encode_mode,
        phase_encode_index=phase_encode_index,
        repeat_phase_encodes=repeat_phase_encodes,
    )
    seq_or_path = _write_sequence(
        seq,
        seq_path,
        fallback_name=f"cpmg_te{_format_token(te_ms)}_n{int(n_echo)}_ref{_format_token(refoc_angle_deg)}",
    )
    te_list = [float(te_ms) * (i + 1) for i in range(int(n_echo))]
    return SimulationProtocol(
        simulation_backend="mrzero_bloch",
        fit=True,
        model_protocol={
            "model": {"te_ms": te_list},
            "mrzero": {
                "seq_or_path": seq_or_path,
                "data_factory": data_factory,
                "spin_count": int(spin_count),
                "perfect_spoiling": bool(perfect_spoiling),
                "print_progress": bool(print_progress),
            },
        },
    )


def mrzero_protocol_spgr(
    *,
    flip_angle_deg: float = 15.0,
    tr_ms: float = 15.0,
    adc_samples: int = 1,
    n_reps: int = 200,
    spin_count: int = 5000,
    perfect_spoiling: bool = False,
    print_progress: bool = True,
    data_factory: Callable[[Mapping[str, float]], Any] = mrzero_single_voxel_data_factory,
    seq_path: str | Path | None = None,
) -> SimulationProtocol:
    """Standard MRzero SPGR protocol template."""
    seq = build_spgr_sequence(
        flip_angle_deg=flip_angle_deg,
        tr_ms=tr_ms,
        adc_samples=adc_samples,
        n_reps=n_reps,
    )
    seq_or_path = _write_sequence(
        seq,
        seq_path,
        fallback_name=f"spgr_fa{_format_token(flip_angle_deg)}_tr{_format_token(tr_ms)}",
    )
    return SimulationProtocol(
        simulation_backend="mrzero_bloch",
        fit=True,
        model_protocol={
            "model": {"flip_angle_deg": [float(flip_angle_deg)], "tr_ms": float(tr_ms)},
            "mrzero": {
                "seq_or_path": seq_or_path,
                "data_factory": data_factory,
                "spin_count": int(spin_count),
                "perfect_spoiling": bool(perfect_spoiling),
                "print_progress": bool(print_progress),
            },
        },
    )
