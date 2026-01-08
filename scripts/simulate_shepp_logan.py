#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from qmrpy.sim import shepp_logan_2d_maps
from qmrpy.sim.noise import add_rician_noise


@dataclass(frozen=True)
class SheppLoganConfig:
    nx: int
    ny: int
    tr_ms: float
    te_ms: float
    snr: float
    t1_min_ms: float
    t1_max_ms: float
    t2_min_ms: float
    t2_max_ms: float
    pd_max: float
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


def _plot_maps(out_dir: Path, maps: dict[str, np.ndarray]) -> str | None:
    try:
        import pandas as pd
        from plotnine import aes, facet_wrap, geom_raster, ggplot, theme_void
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
        + theme_void()
    )
    fig_path = out_dir / "shepp_logan_maps.png"
    ggsave(p, filename=str(fig_path), verbose=False, dpi=150)
    return str(fig_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulate 2D Shepp-Logan phantom with PD/T1/T2.")
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--tr-ms", type=float, default=2000.0)
    parser.add_argument("--te-ms", type=float, default=80.0)
    parser.add_argument("--snr", type=float, default=50.0)
    parser.add_argument("--t1-min-ms", type=float, default=600.0)
    parser.add_argument("--t1-max-ms", type=float, default=1400.0)
    parser.add_argument("--t2-min-ms", type=float, default=40.0)
    parser.add_argument("--t2-max-ms", type=float, default=120.0)
    parser.add_argument("--pd-max", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("output/sim/shepp_logan"))
    args = parser.parse_args()

    cfg = SheppLoganConfig(
        nx=int(args.nx),
        ny=int(args.ny),
        tr_ms=float(args.tr_ms),
        te_ms=float(args.te_ms),
        snr=float(args.snr),
        t1_min_ms=float(args.t1_min_ms),
        t1_max_ms=float(args.t1_max_ms),
        t2_min_ms=float(args.t2_min_ms),
        t2_max_ms=float(args.t2_max_ms),
        pd_max=float(args.pd_max),
        seed=int(args.seed),
    )

    out_dir = _build_run_dir(Path(args.out_dir), "shepp-logan")
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "arrays").mkdir(parents=True, exist_ok=True)

    pd, t1_ms, t2_ms = shepp_logan_2d_maps(
        nx=cfg.nx,
        ny=cfg.ny,
        t1_range_ms=(cfg.t1_min_ms, cfg.t1_max_ms),
        t2_range_ms=(cfg.t2_min_ms, cfg.t2_max_ms),
        pd_max=cfg.pd_max,
    )

    mask = pd > 0
    signal = np.zeros_like(pd, dtype=np.float64)
    if np.any(mask):
        e1 = np.zeros_like(t1_ms, dtype=np.float64)
        e2 = np.zeros_like(t2_ms, dtype=np.float64)
        e1[mask] = np.exp(-cfg.tr_ms / t1_ms[mask])
        e2[mask] = np.exp(-cfg.te_ms / t2_ms[mask])
        signal = pd * (1.0 - e1) * e2

    rng = np.random.default_rng(cfg.seed)
    sigma = 0.0 if cfg.snr <= 0 else float(np.max(signal)) / float(cfg.snr)
    signal_noisy = add_rician_noise(signal, sigma=sigma, rng=rng)

    arrays = {
        "pd": pd,
        "t1_ms": t1_ms,
        "t2_ms": t2_ms,
        "signal": signal,
        "signal_noisy": signal_noisy,
    }
    array_paths = _save_arrays(out_dir / "arrays", arrays)

    fig_path = _plot_maps(out_dir / "figures", arrays)

    report = {
        "config": asdict(cfg),
        "paths": {"arrays": array_paths, "figure": fig_path},
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
