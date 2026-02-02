#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from qmrpy.sim.templates import build_cpmg_sequence


def _format_token(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot CPMG sequence using pypulseq visualization.")
    parser.add_argument("--te-ms", type=float, default=30.0)
    parser.add_argument("--n-echo", type=int, default=32)
    parser.add_argument("--adc-samples", type=int, default=1)
    parser.add_argument("--refoc-angle-deg", type=float, default=180.0)
    parser.add_argument("--slice-thickness-mm", type=float, default=5.0)
    parser.add_argument("--fov-cm", type=float, default=20.0)
    parser.add_argument("--matrix", type=int, nargs=2, default=[128, 128])
    parser.add_argument("--readout-bw-hz-per-pixel", type=float, default=300.0)
    parser.add_argument("--phase-encode-mode", type=str, default="cpmg", choices=["cpmg", "tse"])
    parser.add_argument("--phase-encode-index", type=int, default=None)
    parser.add_argument("--repeat-phase-encodes", action="store_true")
    parser.add_argument("--time-range-ms", type=float, nargs=2, default=None)
    parser.add_argument("--time-disp", type=str, default="ms", choices=["s", "ms", "us"])
    parser.add_argument("--grad-disp", type=str, default="mT/m", choices=["kHz/m", "mT/m"])
    parser.add_argument("--show-blocks", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("output/seq"))
    args = parser.parse_args()

    plt.close("all")
    matrix_x, matrix_y = int(args.matrix[0]), int(args.matrix[1])
    adc_samples = int(args.adc_samples)
    if adc_samples <= 1:
        adc_samples = matrix_x
    if args.phase_encode_index is None:
        phase_encode_index = matrix_y // 2
    else:
        phase_encode_index = int(args.phase_encode_index)

    seq = build_cpmg_sequence(
        te_ms=float(args.te_ms),
        n_echo=int(args.n_echo),
        adc_samples=adc_samples,
        refoc_angle_deg=float(args.refoc_angle_deg),
        slice_thickness_mm=float(args.slice_thickness_mm),
        fov_mm=float(args.fov_cm) * 10.0,
        matrix=(matrix_x, matrix_y),
        readout_bw_hz_per_pixel=float(args.readout_bw_hz_per_pixel),
        include_readout=True,
        phase_encode_mode=str(args.phase_encode_mode),
        phase_encode_index=phase_encode_index,
        repeat_phase_encodes=bool(args.repeat_phase_encodes),
    )

    if args.time_range_ms is None:
        time_range = (0.0, float("inf"))
    else:
        time_range = (float(args.time_range_ms[0]) * 1e-3, float(args.time_range_ms[1]) * 1e-3)

    seq.plot(
        show_blocks=bool(args.show_blocks),
        time_range=time_range,
        time_disp=str(args.time_disp),
        grad_disp=str(args.grad_disp),
        plot_now=False,
    )

    fig_nums = sorted(plt.get_fignums())
    if len(fig_nums) < 2:
        raise RuntimeError("pypulseq plot did not generate two figures as expected.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base = (
        f"cpmg_te{_format_token(float(args.te_ms))}"
        f"_n{int(args.n_echo)}"
        f"_slice{_format_token(float(args.slice_thickness_mm))}mm"
    )
    fig1 = plt.figure(fig_nums[0])
    fig2 = plt.figure(fig_nums[1])
    out1 = args.out_dir / f"{base}_rf_adc.png"
    out2 = args.out_dir / f"{base}_gradients.png"
    fig1.savefig(out1, dpi=150)
    fig2.savefig(out2, dpi=150)
    plt.close("all")

    print(f"Saved: {out1}")
    print(f"Saved: {out2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
