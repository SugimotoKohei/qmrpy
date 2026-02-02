#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from qmrpy.sim import shepp_logan_2d_maps, shepp_logan_3d_maps
from qmrpy.sim.noise import add_rician_noise


@dataclass(frozen=True)
class SheppLoganConfig:
    nx: int
    ny: int
    nz: int
    tr_ms: float
    te_ms: float
    n_echo: int
    snr: float
    t1_min_ms: float
    t1_max_ms: float
    t2_min_ms: float
    t2_max_ms: float
    pd_max: float
    slice_thickness_mm: float
    slice_center_mm: float
    voxel_size_z_mm: float
    seed: int


def _build_run_dir(out_root: Path, tag: str) -> Path:
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d_%H%M%S")
    safe_tag = "".join(ch if (ch.isalnum() or ch in "-_") else "-" for ch in tag).strip("-")
    run_id = f"{ts}_{safe_tag}" if safe_tag else ts
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_arrays(out_dir: Path, arrays: dict[str, Any]) -> dict[str, str]:
    paths: dict[str, str] = {}
    for name, arr in arrays.items():
        path = out_dir / f"{name}.npy"
        np.save(path, np.asarray(arr))
        paths[name] = str(path)
    return paths


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


def _plot_maps(out_dir: Path, maps: dict[str, np.ndarray]) -> str | None:
    try:
        import pandas as pd
        from plotnine import aes, facet_wrap, geom_raster, ggplot, scale_y_reverse, theme_void
        from plotnine import ggsave
    except Exception:
        return None

    records: list[dict[str, Any]] = []
    for name, img in maps.items():
        ny, nx = img.shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        records.append(
            pd.DataFrame(
                {
                    "x": xx.ravel(),
                    "y": yy.ravel(),
                    "value": img.ravel(),
                    "kind": name,
                }
            )
        )
    df = pd.concat(records, ignore_index=True)
    p = (
        ggplot(df, aes("x", "y", fill="value"))
        + geom_raster()
        + facet_wrap("~kind", ncol=2)
        + scale_y_reverse()
        + theme_void()
    )
    fig_path = out_dir / "shepp_logan_maps.png"
    ggsave(p, filename=str(fig_path), verbose=False, dpi=150)
    return str(fig_path)


def _plot_maps_individual(out_dir: Path, maps: dict[str, np.ndarray]) -> str | None:
    """Plot maps with per-panel normalization (plotnine doesn't support per-facet fill scales)."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    n = len(maps)
    ncol = 2
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, (name, img) in zip(axes, maps.items()):
        im = ax.imshow(img, cmap="viridis", origin="upper")
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[len(maps) :]:
        ax.axis("off")

    fig_path = out_dir / "shepp_logan_maps_individual.png"
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return str(fig_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Simulate 2D/3D Shepp-Logan phantom and slice-select CPMG signals."
    )
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--tr-ms", type=float, default=2000.0)
    parser.add_argument("--te-ms", type=float, default=80.0)
    parser.add_argument("--n-echo", type=int, default=1)
    parser.add_argument("--snr", type=float, default=50.0)
    parser.add_argument("--t1-min-ms", type=float, default=600.0)
    parser.add_argument("--t1-max-ms", type=float, default=1400.0)
    parser.add_argument("--t2-min-ms", type=float, default=40.0)
    parser.add_argument("--t2-max-ms", type=float, default=120.0)
    parser.add_argument("--pd-max", type=float, default=1.0)
    parser.add_argument("--nz", type=int, default=64)
    parser.add_argument("--slice-thickness-mm", type=float, default=5.0)
    parser.add_argument("--slice-center-mm", type=float, default=0.0)
    parser.add_argument("--voxel-size-z-mm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("output/sim/shepp_logan"))
    args = parser.parse_args()

    cfg = SheppLoganConfig(
        nx=int(args.nx),
        ny=int(args.ny),
        nz=int(args.nz),
        tr_ms=float(args.tr_ms),
        te_ms=float(args.te_ms),
        n_echo=int(args.n_echo),
        snr=float(args.snr),
        t1_min_ms=float(args.t1_min_ms),
        t1_max_ms=float(args.t1_max_ms),
        t2_min_ms=float(args.t2_min_ms),
        t2_max_ms=float(args.t2_max_ms),
        pd_max=float(args.pd_max),
        slice_thickness_mm=float(args.slice_thickness_mm),
        slice_center_mm=float(args.slice_center_mm),
        voxel_size_z_mm=float(args.voxel_size_z_mm),
        seed=int(args.seed),
    )

    out_dir = _build_run_dir(Path(args.out_dir), "shepp-logan")
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "arrays").mkdir(parents=True, exist_ok=True)

    if cfg.nz <= 1:
        pd, t1_ms, t2_ms = shepp_logan_2d_maps(
            nx=cfg.nx,
            ny=cfg.ny,
            t1_range_ms=(cfg.t1_min_ms, cfg.t1_max_ms),
            t2_range_ms=(cfg.t2_min_ms, cfg.t2_max_ms),
            pd_max=cfg.pd_max,
        )
        pd_3d = t1_3d = t2_3d = None
    else:
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
            center_mm=cfg.slice_center_mm,
            voxel_size_mm=cfg.voxel_size_z_mm,
        )
        t1_ms = _slab_average(
            t1_3d,
            thickness_mm=cfg.slice_thickness_mm,
            center_mm=cfg.slice_center_mm,
            voxel_size_mm=cfg.voxel_size_z_mm,
        )
        t2_ms = _slab_average(
            t2_3d,
            thickness_mm=cfg.slice_thickness_mm,
            center_mm=cfg.slice_center_mm,
            voxel_size_mm=cfg.voxel_size_z_mm,
        )

    mask = pd > 0
    if cfg.n_echo <= 0:
        raise ValueError("n_echo must be >= 1")

    e1 = np.zeros_like(t1_ms, dtype=np.float64)
    if np.any(mask):
        e1[mask] = np.exp(-cfg.tr_ms / t1_ms[mask])

    te_list = [float(cfg.te_ms) * (idx + 1) for idx in range(int(cfg.n_echo))]
    signal_stack = np.zeros((pd.shape[0], pd.shape[1], int(cfg.n_echo)), dtype=np.float64)
    if np.any(mask):
        for i, te in enumerate(te_list):
            e2 = np.zeros_like(t2_ms, dtype=np.float64)
            e2[mask] = np.exp(-float(te) / t2_ms[mask])
            signal_stack[..., i] = pd * (1.0 - e1) * e2

    rng = np.random.default_rng(cfg.seed)
    sigma = 0.0 if cfg.snr <= 0 else float(np.max(signal_stack)) / float(cfg.snr)
    signal_noisy_stack = add_rician_noise(signal_stack, sigma=sigma, rng=rng)

    if cfg.n_echo == 1:
        signal = signal_stack[..., 0]
        signal_noisy = signal_noisy_stack[..., 0]
    else:
        signal = signal_stack
        signal_noisy = signal_noisy_stack

    arrays: dict[str, Any] = {
        "pd": pd,
        "t1_ms": t1_ms,
        "t2_ms": t2_ms,
        "signal": signal,
        "signal_noisy": signal_noisy,
        "echo_times_ms": np.asarray(te_list, dtype=np.float64),
    }
    if pd_3d is not None and t1_3d is not None and t2_3d is not None:
        arrays.update({"pd_3d": pd_3d, "t1_ms_3d": t1_3d, "t2_ms_3d": t2_3d})
    array_paths = _save_arrays(out_dir / "arrays", arrays)

    plot_maps = {
        "pd": pd,
        "t1_ms": t1_ms,
        "t2_ms": t2_ms,
        "signal_echo1": signal_stack[..., 0],
    }
    figures_dir = out_dir / "figures"
    fig_path = _plot_maps(figures_dir, plot_maps)
    fig_path_individual = _plot_maps_individual(figures_dir, plot_maps)

    report = {
        "config": asdict(cfg),
        "paths": {"arrays": array_paths, "figure": fig_path, "figure_individual": fig_path_individual},
        "stats": {
            "signal_max": float(np.max(signal)),
            "signal_min": float(np.min(signal)),
            "signal_mean": float(np.mean(signal)),
            "signal_noisy_mean": float(np.mean(signal_noisy)),
            "snr": float(cfg.snr),
            "sigma": float(sigma),
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["stats"], indent=2, ensure_ascii=False))
    if fig_path is None:
        print("plotnine が見つからないため、図の保存はスキップしました。")
    else:
        print(f"Figure saved: {fig_path}")
    print(f"Output dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
