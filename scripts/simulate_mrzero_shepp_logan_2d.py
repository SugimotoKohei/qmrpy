#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from qmrpy.sim import shepp_logan_3d_maps
from qmrpy.sim.mrzero import simulate_bloch, simulate_pdg
from qmrpy.sim.templates import build_cpmg_sequence


@dataclass(frozen=True)
class MrzeroTseConfig:
    nx: int
    ny: int
    nz: int
    fov_mm: float
    te_ms: float
    n_echo: int
    slice_thickness_mm: float
    voxel_size_z_mm: float
    tr_ms: float
    t1_min_ms: float
    t1_max_ms: float
    t2_min_ms: float
    t2_max_ms: float
    pd_max: float
    matrix_x: int
    matrix_y: int
    readout_bw_hz_per_pixel: float
    spin_count: int
    backend: str
    pdg_max_states: int
    pdg_min_latent_signal: float
    pdg_min_emitted_signal: float
    stride: int
    phase_encode_mode: str
    full_kspace: bool
    ideal_encoding: bool
    seed: int
    save_every: int


def _build_run_dir(out_root: Path, tag: str) -> Path:
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d_%H%M%S")
    safe_tag = "".join(ch if (ch.isalnum() or ch in "-_") else "-" for ch in tag).strip("-")
    run_id = f"{ts}_{safe_tag}" if safe_tag else ts
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _slab_average(volume: np.ndarray, *, thickness_mm: float, center_mm: float, voxel_size_mm: float) -> np.ndarray:
    if volume.ndim != 3:
        raise ValueError("volume must be 3D (nz, ny, nx)")
    nz = volume.shape[0]
    z_mm = (np.arange(nz, dtype=np.float64) - (nz - 1) / 2.0) * float(voxel_size_mm)
    half = float(thickness_mm) / 2.0
    mask = np.abs(z_mm - float(center_mm)) <= half
    if not np.any(mask):
        idx = int(np.argmin(np.abs(z_mm - float(center_mm))))
        mask[idx] = True
    return np.mean(volume[mask], axis=0)


def _make_simdata(
    mr0: Any,
    *,
    pd: float,
    t1_ms: float,
    t2_ms: float,
    voxel_pos_mm: tuple[float, float, float],
    voxel_size_mm: tuple[float, float, float],
    coil_sens_dim: int = 1,
) -> Any:
    import torch

    t2dash_ms = t2_ms
    b1 = 1.0
    b0 = 0.0

    pd_t = torch.ones((1,)) * float(pd)
    t1 = torch.ones((1,)) * (float(t1_ms) * 1e-3)
    t2 = torch.ones((1,)) * (float(t2_ms) * 1e-3)
    t2dash = torch.ones((1,)) * (float(t2dash_ms) * 1e-3)
    d = torch.zeros((1, 3))
    b0_t = torch.ones((1,)) * float(b0)
    b1_t = torch.ones((int(coil_sens_dim), 1)) * float(b1)
    coil_sens = torch.ones((int(coil_sens_dim), 1))
    size = torch.tensor([float(s) * 1e-3 for s in voxel_size_mm])
    voxel_pos = torch.tensor([[p * 1e-3 for p in voxel_pos_mm]], dtype=torch.float32)
    nyquist = torch.tensor([1, 1, 1])
    dephasing_func = lambda b0_in, t: torch.zeros_like(b0_in)

    return mr0.SimData(pd_t, t1, t2, t2dash, d, b0_t, b1_t, coil_sens, size, voxel_pos, nyquist, dephasing_func)


def _plot_echoes(out_dir: Path, pd: np.ndarray, images: np.ndarray, *, echo_indices: list[int]) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    n = len(echo_indices) + 1
    ncol = n
    fig, axes = plt.subplots(1, ncol, figsize=(3.5 * ncol, 3.5))
    axes = np.atleast_1d(axes).ravel()

    im = axes[0].imshow(pd, cmap="viridis", origin="upper")
    axes[0].set_title("pd")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    for ax, echo_idx in zip(axes[1:], echo_indices):
        img = images[echo_idx]
        im = ax.imshow(img, cmap="viridis", origin="upper")
        ax.set_title(f"echo{echo_idx + 1}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig_path = out_dir / "mrzero_tse_echoes.png"
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return str(fig_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="MRzero TSE-style 2D simulation with Shepp-Logan phantom.")
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--nz", type=int, default=64)
    parser.add_argument("--fov-cm", type=float, default=20.0)
    parser.add_argument("--te-ms", type=float, default=30.0)
    parser.add_argument("--n-echo", type=int, default=32)
    parser.add_argument("--slice-thickness-mm", type=float, default=5.0)
    parser.add_argument("--voxel-size-z-mm", type=float, default=1.0)
    parser.add_argument("--tr-ms", type=float, default=2000.0)
    parser.add_argument("--t1-min-ms", type=float, default=600.0)
    parser.add_argument("--t1-max-ms", type=float, default=1400.0)
    parser.add_argument("--t2-min-ms", type=float, default=40.0)
    parser.add_argument("--t2-max-ms", type=float, default=120.0)
    parser.add_argument("--pd-max", type=float, default=1.0)
    parser.add_argument("--matrix", type=int, nargs=2, default=[128, 128])
    parser.add_argument("--readout-bw-hz-per-pixel", type=float, default=300.0)
    parser.add_argument("--spin-count", type=int, default=50)
    parser.add_argument("--backend", type=str, default="pdg", choices=["bloch", "pdg"])
    parser.add_argument("--pdg-max-states", type=int, default=200)
    parser.add_argument("--pdg-min-latent-signal", type=float, default=1e-4)
    parser.add_argument("--pdg-min-emitted-signal", type=float, default=1e-4)
    parser.add_argument("--stride", type=int, default=1, help="Voxel subsampling stride for speed.")
    parser.add_argument("--phase-encode-mode", type=str, default="cpmg", choices=["cpmg", "tse"])
    parser.add_argument(
        "--full-kspace",
        action="store_true",
        help="If set, simulate all phase-encode lines and reconstruct via k-space (very slow).",
    )
    parser.add_argument(
        "--ideal-encoding",
        action="store_true",
        help="Use PDG echo amplitudes and ideal FFT encoding (no readout/phase gradients in sequence).",
    )
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("output/sim/mrzero_tse_shepp_logan"))
    args = parser.parse_args()

    if args.stride <= 0:
        raise ValueError("stride must be >= 1")
    nx, ny, nz = int(args.nx), int(args.ny), int(args.nz)
    matrix_x, matrix_y = int(args.matrix[0]), int(args.matrix[1])
    if matrix_x != nx or matrix_y != ny:
        raise ValueError("matrix must match nx/ny for this script")

    cfg = MrzeroTseConfig(
        nx=nx,
        ny=ny,
        nz=nz,
        fov_mm=float(args.fov_cm) * 10.0,
        te_ms=float(args.te_ms),
        n_echo=int(args.n_echo),
        slice_thickness_mm=float(args.slice_thickness_mm),
        voxel_size_z_mm=float(args.voxel_size_z_mm),
        tr_ms=float(args.tr_ms),
        t1_min_ms=float(args.t1_min_ms),
        t1_max_ms=float(args.t1_max_ms),
        t2_min_ms=float(args.t2_min_ms),
        t2_max_ms=float(args.t2_max_ms),
        pd_max=float(args.pd_max),
        matrix_x=matrix_x,
        matrix_y=matrix_y,
        readout_bw_hz_per_pixel=float(args.readout_bw_hz_per_pixel),
        spin_count=int(args.spin_count),
        backend=str(args.backend),
        pdg_max_states=int(args.pdg_max_states),
        pdg_min_latent_signal=float(args.pdg_min_latent_signal),
        pdg_min_emitted_signal=float(args.pdg_min_emitted_signal),
        stride=int(args.stride),
        phase_encode_mode=str(args.phase_encode_mode),
        full_kspace=bool(args.full_kspace),
        ideal_encoding=bool(args.ideal_encoding),
        seed=int(args.seed),
        save_every=int(args.save_every),
    )

    if cfg.save_every <= 0:
        raise ValueError("save_every must be >= 1")

    if args.run_dir is not None:
        out_dir = Path(args.run_dir)
        if args.resume:
            if not out_dir.exists():
                raise FileNotFoundError(f"run-dir not found: {out_dir}")
        else:
            if out_dir.exists():
                raise FileExistsError(f"run-dir already exists: {out_dir}")
            out_dir.mkdir(parents=True, exist_ok=True)
    else:
        if args.resume:
            raise ValueError("--resume requires --run-dir")
        out_dir = _build_run_dir(Path(args.out_dir), "mrzero-tse")

    (out_dir / "arrays").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "seq").mkdir(parents=True, exist_ok=True)

    config_path = out_dir / "config.json"
    if args.resume:
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {out_dir}")
        loaded = _load_json(config_path)
        defaults = asdict(cfg)
        merged = defaults.copy()
        merged.update({k: v for k, v in loaded.items() if k in defaults})
        cfg = MrzeroTseConfig(**merged)
    else:
        _save_json(config_path, asdict(cfg))

    pd_3d, t1_3d, t2_3d = shepp_logan_3d_maps(
        nx=cfg.nx,
        ny=cfg.ny,
        nz=cfg.nz,
        t1_range_ms=(cfg.t1_min_ms, cfg.t1_max_ms),
        t2_range_ms=(cfg.t2_min_ms, cfg.t2_max_ms),
        pd_max=cfg.pd_max,
    )
    pd = _slab_average(
        pd_3d,
        thickness_mm=cfg.slice_thickness_mm,
        center_mm=0.0,
        voxel_size_mm=cfg.voxel_size_z_mm,
    )
    t1_ms = _slab_average(
        t1_3d,
        thickness_mm=cfg.slice_thickness_mm,
        center_mm=0.0,
        voxel_size_mm=cfg.voxel_size_z_mm,
    )
    t2_ms = _slab_average(
        t2_3d,
        thickness_mm=cfg.slice_thickness_mm,
        center_mm=0.0,
        voxel_size_mm=cfg.voxel_size_z_mm,
    )

    if cfg.ideal_encoding:
        adc_samples = 1
        seq = build_cpmg_sequence(
            te_ms=cfg.te_ms,
            n_echo=cfg.n_echo,
            adc_samples=adc_samples,
            refoc_angle_deg=180.0,
            slice_thickness_mm=cfg.slice_thickness_mm,
            include_readout=False,
            auto_adc_samples=False,
        )
    else:
        adc_samples = cfg.matrix_x if cfg.full_kspace else 1
        seq = build_cpmg_sequence(
            te_ms=cfg.te_ms,
            n_echo=cfg.n_echo,
            adc_samples=adc_samples,
            refoc_angle_deg=180.0,
            slice_thickness_mm=cfg.slice_thickness_mm,
            fov_mm=cfg.fov_mm,
            matrix=(cfg.matrix_x, cfg.matrix_y),
            readout_bw_hz_per_pixel=cfg.readout_bw_hz_per_pixel,
            include_readout=True,
            phase_encode_mode=cfg.phase_encode_mode,
            phase_encode_index=cfg.matrix_y // 2,
            repeat_phase_encodes=cfg.full_kspace,
            auto_adc_samples=cfg.full_kspace,
        )
    seq_path = out_dir / "seq" / "cpmg_tse.seq"
    seq.write(str(seq_path))

    try:
        import MRzeroCore as mr0  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover - optional
        raise ModuleNotFoundError("MRzeroCore が必要です。") from exc

    dx_mm = cfg.fov_mm / cfg.matrix_x
    dy_mm = cfg.fov_mm / cfg.matrix_y
    voxel_size_mm = (dx_mm, dy_mm, cfg.slice_thickness_mm)
    x_coords = (np.arange(cfg.matrix_x, dtype=np.float64) - (cfg.matrix_x - 1) / 2.0) * dx_mm
    y_coords = ((cfg.matrix_y - 1) / 2.0 - np.arange(cfg.matrix_y, dtype=np.float64)) * dy_mm

    mask = pd > 0
    if cfg.stride > 1:
        stride_mask = np.zeros_like(mask, dtype=bool)
        stride_mask[:: cfg.stride, :: cfg.stride] = True
        mask &= stride_mask

    indices = np.argwhere(mask)
    n_voxels = int(indices.shape[0])
    n_echo = cfg.n_echo
    if cfg.ideal_encoding:
        expected_len = n_echo
        images_complex_path = out_dir / "arrays" / "images_complex.npy"
        if args.resume and images_complex_path.exists():
            images_complex = np.load(images_complex_path)
        else:
            images_complex = np.zeros((n_echo, cfg.ny, cfg.nx), dtype=np.complex128)
        images_mag = None
        kspace = None
    elif cfg.full_kspace:
        n_pe = cfg.matrix_y
        expected_len = n_pe * n_echo * adc_samples
        kspace_path = out_dir / "arrays" / "kspace.npy"
        if args.resume and kspace_path.exists():
            kspace = np.load(kspace_path)
        else:
            kspace = np.zeros((n_echo, n_pe, adc_samples), dtype=np.complex128)
        images_mag = None
    else:
        expected_len = n_echo * adc_samples
        kspace = None
        images_path = out_dir / "arrays" / "images_mag.npy"
        if args.resume and images_path.exists():
            images_mag = np.load(images_path)
        else:
            images_mag = np.zeros((n_echo, cfg.ny, cfg.nx), dtype=np.float64)

    progress_path = out_dir / "progress.json"
    start_index = 0
    if args.resume and progress_path.exists():
        progress = _load_json(progress_path)
        start_index = int(progress.get("next_index", 0))

    for count, (iy, ix) in enumerate(indices[start_index:], start=start_index + 1):
        data = _make_simdata(
            mr0,
            pd=float(pd[iy, ix]),
            t1_ms=float(t1_ms[iy, ix]),
            t2_ms=float(t2_ms[iy, ix]),
            voxel_pos_mm=(float(x_coords[ix]), float(y_coords[iy]), 0.0),
            voxel_size_mm=voxel_size_mm,
            coil_sens_dim=3 if cfg.backend == "pdg" else 1,
        )
        if cfg.backend == "pdg":
            sig = simulate_pdg(
                str(seq_path),
                data,
                max_states=int(cfg.pdg_max_states),
                min_latent_signal=float(cfg.pdg_min_latent_signal),
                min_emitted_signal=float(cfg.pdg_min_emitted_signal),
                print_progress=False,
                return_graph=False,
            )
        else:
            sig = simulate_bloch(
                str(seq_path),
                data,
                spin_count=int(cfg.spin_count),
                perfect_spoiling=False,
                print_progress=False,
            )
        sig = np.asarray(sig)
        if sig.ndim > 1:
            sig = sig.reshape(sig.shape[0], -1)
            if sig.shape[1] > 1:
                sig = sig.mean(axis=1)
            else:
                sig = sig[:, 0]
        if sig.size != expected_len:
            raise ValueError(f"Unexpected signal length: {sig.size} != {expected_len}")
        if cfg.ideal_encoding:
            sig = sig.reshape(n_echo)
            images_complex[:, int(iy), int(ix)] = sig
        elif cfg.full_kspace:
            sig = sig.reshape(n_pe, n_echo, adc_samples)
            kspace += np.transpose(sig, (1, 0, 2))
        else:
            sig = sig.reshape(n_echo, adc_samples)
            sig_line = sig[:, adc_samples // 2]
            images_mag[:, int(iy), int(ix)] = np.abs(sig_line)
        if count % 200 == 0 or count == n_voxels:
            print(f"Simulated {count}/{n_voxels} voxels")
        if count % cfg.save_every == 0 or count == n_voxels:
            if cfg.ideal_encoding:
                np.save(out_dir / "arrays" / "images_complex.npy", images_complex)
            elif cfg.full_kspace and kspace is not None:
                np.save(out_dir / "arrays" / "kspace.npy", kspace)
            else:
                np.save(out_dir / "arrays" / "images_mag.npy", images_mag)
            _save_json(progress_path, {"next_index": count})

    if cfg.ideal_encoding:
        images_mag = np.abs(images_complex)
        if cfg.full_kspace:
            kspace = np.fft.fftshift(
                np.fft.fft2(np.fft.ifftshift(images_complex, axes=(-2, -1)), axes=(-2, -1)),
                axes=(-2, -1),
            )
    elif cfg.full_kspace:
        kspace_shift = np.fft.ifftshift(kspace, axes=(-2, -1))
        images = np.fft.ifft2(kspace_shift, axes=(-2, -1))
        images_mag = np.abs(images)

    echo_times_ms = np.asarray([cfg.te_ms * (i + 1) for i in range(cfg.n_echo)], dtype=np.float64)
    np.save(out_dir / "arrays" / "pd.npy", pd)
    np.save(out_dir / "arrays" / "t1_ms.npy", t1_ms)
    np.save(out_dir / "arrays" / "t2_ms.npy", t2_ms)
    if cfg.ideal_encoding:
        np.save(out_dir / "arrays" / "images_complex.npy", images_complex)
    if cfg.full_kspace and kspace is not None:
        np.save(out_dir / "arrays" / "kspace.npy", kspace)
    np.save(out_dir / "arrays" / "images_mag.npy", images_mag)
    np.save(out_dir / "arrays" / "echo_times_ms.npy", echo_times_ms)

    fig_path = _plot_echoes(out_dir / "figures", pd, images_mag, echo_indices=[0, n_echo // 2, n_echo - 1])

    report = {
        "config": asdict(cfg),
        "sequence": str(seq_path),
        "paths": {
            "pd": str(out_dir / "arrays" / "pd.npy"),
            "t1_ms": str(out_dir / "arrays" / "t1_ms.npy"),
            "t2_ms": str(out_dir / "arrays" / "t2_ms.npy"),
            "kspace": str(out_dir / "arrays" / "kspace.npy") if cfg.full_kspace else None,
            "images_mag": str(out_dir / "arrays" / "images_mag.npy"),
            "images_complex": str(out_dir / "arrays" / "images_complex.npy")
            if cfg.ideal_encoding
            else None,
            "echo_times_ms": str(out_dir / "arrays" / "echo_times_ms.npy"),
            "figure": fig_path,
        },
        "stats": {
            "n_voxels": n_voxels,
            "signal_max": float(np.max(np.abs(kspace)))
            if kspace is not None
            else float(np.max(images_mag)),
            "signal_mean": float(np.mean(np.abs(kspace)))
            if kspace is not None
            else float(np.mean(images_mag)),
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Output dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
