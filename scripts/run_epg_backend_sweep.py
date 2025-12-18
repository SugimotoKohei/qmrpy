from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def _now_run_id(tag: str) -> str:
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d_%H%M%S")
    safe_tag = "".join(ch if (ch.isalnum() or ch in "-_") else "-" for ch in tag).strip("-")
    return f"{ts}_{safe_tag}" if safe_tag else ts


def _git_info() -> dict[str, object]:
    import subprocess

    def run(cmd: list[str]) -> str:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()

    try:
        commit = run(["git", "rev-parse", "HEAD"])
        status = run(["git", "status", "--porcelain=v1"])
        return {"commit": commit, "dirty": bool(status)}
    except Exception:
        return {"commit": None, "dirty": None}


def _ensure_dirs(run_dir: Path) -> dict[str, Path]:
    paths = {
        "run_dir": run_dir,
        "config_snapshot": run_dir / "config_snapshot",
        "env": run_dir / "env",
        "metrics": run_dir / "metrics",
        "figures": run_dir / "figures",
        "artifacts": run_dir / "artifacts",
        "logs": run_dir / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _setup_runtime_caches(env_dir: Path) -> dict[str, str]:
    """Set writable cache dirs to avoid warnings on macOS/CI."""
    mplconfig = env_dir / "matplotlib"
    xdg_cache = env_dir / "cache"
    mplconfig.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)

    env_updates = {
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(mplconfig),
        "XDG_CACHE_HOME": str(xdg_cache),
    }
    os.environ.update(env_updates)
    return env_updates


def _read_toml(path: Path) -> dict[str, object]:
    import tomllib

    return tomllib.loads(path.read_text(encoding="utf-8"))


def _require_plotnine() -> None:
    try:
        import plotnine  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "plotnine が必要です。`uv sync --extra viz` などで導入してください。"
        ) from exc


def _as_float_list(x: object, *, name: str) -> list[float]:
    if not isinstance(x, list) or not all(isinstance(v, (int, float)) for v in x):
        raise ValueError(f"{name} must be a list of numbers")
    return [float(v) for v in x]


def _as_int_list_or_int(x: object, *, name: str) -> list[int]:
    if isinstance(x, int):
        return [int(x)]
    if isinstance(x, list) and all(isinstance(v, int) for v in x):
        return [int(v) for v in x]
    raise ValueError(f"{name} must be int or list[int]")


def _as_float_list_or_float(x: object, *, name: str) -> list[float]:
    if isinstance(x, (int, float)):
        return [float(x)]
    return _as_float_list(x, name=name)


def _normalize_curve(y, *, mode: str):
    import numpy as np

    y = np.asarray(y, dtype=np.float64)
    if mode == "none":
        return y
    if mode == "first_echo":
        d = float(y[0]) if y.size else 0.0
        return y / d if d != 0.0 else y
    raise ValueError("normalize must be 'none' or 'first_echo'")


def _run_backend_sweep(cfg: dict[str, object], *, out_metrics: Path, out_figures: Path) -> dict[str, object]:
    import numpy as np

    _require_plotnine()
    from plotnine import (
        aes,
        facet_wrap,
        geom_histogram,
        geom_line,
        geom_point,
        ggplot,
        labs,
        scale_y_log10,
        theme_bw,
    )
    from plotnine import ggsave

    import pandas as pd

    sweep = cfg.get("epg_backend_sweep", {})
    if not isinstance(sweep, dict):
        raise ValueError("config format error: [epg_backend_sweep] must be a table")

    etls = _as_int_list_or_int(sweep.get("etl", 16), name="epg_backend_sweep.etl")
    te_ms_list = _as_float_list_or_float(sweep.get("te_ms", 10.0), name="epg_backend_sweep.te_ms")
    alpha_deg = _as_float_list(sweep.get("alpha_deg", [180.0]), name="epg_backend_sweep.alpha_deg")
    t1_s = _as_float_list(sweep.get("t1_s", [1.0]), name="epg_backend_sweep.t1_s")
    t2_ms = _as_float_list(sweep.get("t2_ms", [80.0]), name="epg_backend_sweep.t2_ms")
    beta_deg_list = _as_float_list_or_float(sweep.get("beta_deg", 180.0), name="epg_backend_sweep.beta_deg")
    normalize = str(sweep.get("normalize", "first_echo"))
    topk = int(sweep.get("topk", 3))

    from qmrpy.models.t2.decaes_t2 import epg_decay_curve

    rows: list[dict[str, float]] = []
    for etl in etls:
        for te_ms in te_ms_list:
            for beta_deg in beta_deg_list:
                for a in alpha_deg:
                    for t1 in t1_s:
                        for t2 in t2_ms:
                            y_epgpy = epg_decay_curve(
                                etl=int(etl),
                                alpha_deg=float(a),
                                te_s=float(te_ms) / 1000.0,
                                t2_s=float(t2) / 1000.0,
                                t1_s=float(t1),
                                beta_deg=float(beta_deg),
                                backend="epgpy",
                            )
                            y_decaes = epg_decay_curve(
                                etl=int(etl),
                                alpha_deg=float(a),
                                te_s=float(te_ms) / 1000.0,
                                t2_s=float(t2) / 1000.0,
                                t1_s=float(t1),
                                beta_deg=float(beta_deg),
                                backend="decaes",
                            )

                            y1 = _normalize_curve(y_epgpy, mode=normalize)
                            y2 = _normalize_curve(y_decaes, mode=normalize)

                            diff = y1 - y2
                            denom = float(np.mean(y2**2)) if float(np.mean(y2**2)) > 0 else 1.0
                            nmse = float(np.mean(diff**2) / denom)
                            max_abs = float(np.max(np.abs(diff)))
                            rows.append(
                                {
                                    "etl": float(etl),
                                    "te_ms": float(te_ms),
                                    "beta_deg": float(beta_deg),
                                    "alpha_deg": float(a),
                                    "t1_s": float(t1),
                                    "t2_ms": float(t2),
                                    "nmse": nmse,
                                    "max_abs_diff": max_abs,
                                }
                            )

    df = pd.DataFrame(rows)
    df.to_csv(out_metrics.parent / "epg_backend_sweep_per_point.csv", index=False)

    nmse = df["nmse"].to_numpy(dtype=float)
    summary = {
        "etl": [int(x) for x in etls],
        "te_ms": [float(x) for x in te_ms_list],
        "beta_deg": [float(x) for x in beta_deg_list],
        "normalize": str(normalize),
        "n_points": int(df.shape[0]),
        "nmse_mean": float(np.mean(nmse)),
        "nmse_median": float(np.median(nmse)),
        "nmse_p95": float(np.quantile(nmse, 0.95)),
        "nmse_max": float(np.max(nmse)),
        "max_abs_diff_max": float(df["max_abs_diff"].max()),
    }
    out_metrics.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # (A) data check: parameter grid scatter
    df["group"] = (
        df["etl"].astype(int).astype(str)
        + ", TE="
        + df["te_ms"].astype(float).astype(str)
        + "ms, beta="
        + df["beta_deg"].astype(float).astype(str)
        + "deg"
    )
    fig_a = (
        ggplot(df, aes(x="t2_ms", y="t1_s"))
        + geom_point(aes(color="alpha_deg"), size=2.5, alpha=0.8)
        + scale_y_log10()
        + facet_wrap("group", ncol=1)
        + theme_bw()
        + labs(
            title="(A) Data check: sweep grid (t2_ms vs t1_s; color=alpha_deg)",
            x="T2 [ms]",
            y="T1 [s] (log)",
        )
    )
    ggsave(fig_a, filename=str(out_figures / "data_check__grid.png"), verbose=False, dpi=150)

    # (B) primary: nmse distribution and stratification by alpha
    fig_b = (
        ggplot(df, aes(x="nmse"))
        + geom_histogram(bins=40)
        + facet_wrap("group", ncol=1)
        + theme_bw()
        + labs(title="(B) Result: NMSE(epgpy vs decaes) distribution", x="NMSE", y="count")
    )
    ggsave(fig_b, filename=str(out_figures / "result__nmse_hist.png"), verbose=False, dpi=150)

    # (C) failure analysis: overlay top-k worst curves
    df_sorted = df.sort_values("nmse", ascending=False).head(max(topk, 1))
    curve_rows: list[dict[str, object]] = []
    for i, row in enumerate(df_sorted.itertuples(index=False), start=1):
        etl = int(getattr(row, "etl"))
        te_ms = float(getattr(row, "te_ms"))
        beta_deg = float(getattr(row, "beta_deg"))
        a = float(row.alpha_deg)
        t1 = float(row.t1_s)
        t2 = float(row.t2_ms)
        y_epgpy = epg_decay_curve(
            etl=etl,
            alpha_deg=a,
            te_s=float(te_ms) / 1000.0,
            t2_s=float(t2) / 1000.0,
            t1_s=t1,
            beta_deg=float(beta_deg),
            backend="epgpy",
        )
        y_decaes = epg_decay_curve(
            etl=etl,
            alpha_deg=a,
            te_s=float(te_ms) / 1000.0,
            t2_s=float(t2) / 1000.0,
            t1_s=t1,
            beta_deg=float(beta_deg),
            backend="decaes",
        )
        y_epgpy = _normalize_curve(y_epgpy, mode=normalize)
        y_decaes = _normalize_curve(y_decaes, mode=normalize)

        label = f"case{i}: ETL={etl}, TE={te_ms:g}ms, beta={beta_deg:g}, a={a:.0f}, T1={t1:g}s, T2={t2:g}ms"
        for echo_idx in range(etl):
            curve_rows.append(
                {"case": label, "backend": "epgpy", "echo": echo_idx + 1, "signal": float(y_epgpy[echo_idx])}
            )
            curve_rows.append(
                {"case": label, "backend": "decaes", "echo": echo_idx + 1, "signal": float(y_decaes[echo_idx])}
            )

    df_curve = pd.DataFrame(curve_rows)
    fig_c = (
        ggplot(df_curve, aes(x="echo", y="signal", color="backend"))
        + geom_line()
        + facet_wrap("case", ncol=1)
        + theme_bw()
        + labs(title="(C) Failure analysis: top-k worst cases (normalized)", x="echo index", y="signal (norm.)")
    )
    ggsave(fig_c, filename=str(out_figures / "failure__topk_curves.png"), verbose=False, dpi=150)

    return {"metrics": summary, "figures": [p.name for p in out_figures.glob("*.png")]}


def shlex_quote(s: str) -> str:
    import shlex

    return shlex.quote(s)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="TOML config (e.g. configs/exp/epg_backend_sweep.toml)")
    parser.add_argument("--out-root", type=str, default="output/runs", help="output root (default: output/runs)")
    parser.add_argument("--run-id", type=str, default=None, help="override run_id (default: auto)")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    cfg = _read_toml(config_path)

    run_cfg = cfg.get("run", {})
    if not isinstance(run_cfg, dict):
        run_cfg = {}

    tag = str(run_cfg.get("tag", "epg-backend-sweep"))
    run_id = args.run_id or _now_run_id(tag)
    out_root = Path(args.out_root)
    run_dir = out_root / run_id

    paths = _ensure_dirs(run_dir)
    env_updates = _setup_runtime_caches(paths["env"])

    # snapshot config
    shutil.copy2(config_path, paths["config_snapshot"] / config_path.name)

    metrics_path = paths["metrics"] / "epg_backend_sweep_metrics.json"
    result = _run_backend_sweep(cfg, out_metrics=metrics_path, out_figures=paths["figures"])

    run_json = {
        "run_id": run_id,
        "command": " ".join([shlex_quote(x) for x in [sys.executable, *sys.argv]]),
        "config": str(config_path),
        "config_snapshot": str((paths["config_snapshot"] / config_path.name).relative_to(run_dir)),
        "model": "epg_backend_sweep",
        "seed": int(run_cfg.get("seed", 0)) if isinstance(run_cfg.get("seed", 0), int) else 0,
        "git": _git_info(),
        "env": {
            "python": sys.version,
            "platform": sys.platform,
            "env_updates": env_updates,
        },
        "outputs": {
            "metrics": str((paths["metrics"]).relative_to(run_dir)),
            "figures": str((paths["figures"]).relative_to(run_dir)),
            "logs": str((paths["logs"]).relative_to(run_dir)),
            "artifacts": str((paths["artifacts"]).relative_to(run_dir)),
        },
        "model_config": cfg.get("epg_backend_sweep", {}),
        "result": result,
    }
    (run_dir / "run.json").write_text(json.dumps(run_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
