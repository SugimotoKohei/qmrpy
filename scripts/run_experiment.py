from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
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
            "plotnine が必要です。`uv add plotnine` または extras を導入してください（例: `uv sync`）。"
        ) from exc


@dataclass(frozen=True)
class MonoT2Config:
    te_ms: list[float]
    n_samples: int
    m0: float
    t2_min_ms: float
    t2_max_ms: float
    noise_model: str
    noise_sigma: float
    seed: int
    fit_type: str
    drop_first_echo: bool
    offset_term: bool


def _parse_mono_t2_config(config: dict[str, object]) -> MonoT2Config:
    run_cfg = config.get("run", {})
    mono_cfg = config.get("mono_t2", {})
    if not isinstance(run_cfg, dict) or not isinstance(mono_cfg, dict):
        raise ValueError("config format error: [run] and [mono_t2] must be tables")

    te_ms = mono_cfg.get("te_ms")
    if not isinstance(te_ms, list) or not all(isinstance(x, (int, float)) for x in te_ms):
        raise ValueError("mono_t2.te_ms must be a list of numbers")

    n_samples = int(mono_cfg.get("n_samples", 200))
    m0 = float(mono_cfg.get("m0", 1000.0))
    t2_range = mono_cfg.get("t2_range_ms", [20.0, 200.0])
    if (
        not isinstance(t2_range, list)
        or len(t2_range) != 2
        or not all(isinstance(x, (int, float)) for x in t2_range)
    ):
        raise ValueError("mono_t2.t2_range_ms must be [min, max]")
    t2_min_ms, t2_max_ms = float(t2_range[0]), float(t2_range[1])
    noise_model = str(mono_cfg.get("noise_model", "gaussian"))
    noise_sigma = float(mono_cfg.get("noise_sigma", 0.0))
    fit_type = str(mono_cfg.get("fit_type", "exponential"))
    drop_first_echo = bool(mono_cfg.get("drop_first_echo", False))
    offset_term = bool(mono_cfg.get("offset_term", False))
    seed = int(run_cfg.get("seed", 0))

    return MonoT2Config(
        te_ms=[float(x) for x in te_ms],
        n_samples=n_samples,
        m0=m0,
        t2_min_ms=t2_min_ms,
        t2_max_ms=t2_max_ms,
        noise_model=noise_model,
        noise_sigma=noise_sigma,
        seed=seed,
        fit_type=fit_type,
        drop_first_echo=drop_first_echo,
        offset_term=offset_term,
    )


def _run_mono_t2(cfg: MonoT2Config, *, out_metrics: Path, out_figures: Path) -> dict[str, object]:
    import numpy as np

    _require_plotnine()
    from plotnine import aes, geom_abline, geom_histogram, geom_point, ggplot, labs, theme_bw
    from plotnine import ggsave

    from qmrpy.models.t2 import MonoT2
    from qmrpy.sim.noise import add_gaussian_noise, add_rician_noise

    rng = np.random.default_rng(cfg.seed)
    t2_true = rng.uniform(cfg.t2_min_ms, cfg.t2_max_ms, size=cfg.n_samples).astype(float)
    m0_true = np.full(cfg.n_samples, cfg.m0, dtype=float)

    model = MonoT2(te=np.array(cfg.te_ms, dtype=float))
    signal_clean = np.stack([model.forward(m0=float(m0_true[i]), t2=float(t2_true[i])) for i in range(cfg.n_samples)])
    if cfg.noise_model == "gaussian":
        signal = add_gaussian_noise(signal_clean, sigma=cfg.noise_sigma, rng=rng)
    elif cfg.noise_model == "rician":
        signal = add_rician_noise(signal_clean, sigma=cfg.noise_sigma, rng=rng)
    else:
        raise ValueError(f"unknown noise_model for mono_t2: {cfg.noise_model}")

    fitted_m0 = np.empty(cfg.n_samples, dtype=float)
    fitted_t2 = np.empty(cfg.n_samples, dtype=float)
    fitted_offset = np.full(cfg.n_samples, np.nan, dtype=float)
    for i in range(cfg.n_samples):
        fitted = model.fit(
            signal[i],
            fit_type=cfg.fit_type,
            drop_first_echo=cfg.drop_first_echo,
            offset_term=cfg.offset_term,
        )
        fitted_m0[i] = fitted["m0"]
        fitted_t2[i] = fitted["t2"]
        if "offset" in fitted:
            fitted_offset[i] = float(fitted["offset"])

    t2_err = fitted_t2 - t2_true
    m0_err = fitted_m0 - m0_true

    metrics = {
        "n_samples": int(cfg.n_samples),
        "te_ms": [float(x) for x in cfg.te_ms],
        "noise_model": str(cfg.noise_model),
        "noise_sigma": float(cfg.noise_sigma),
        "fit_type": str(cfg.fit_type),
        "drop_first_echo": bool(cfg.drop_first_echo),
        "offset_term": bool(cfg.offset_term),
        "t2_mae": float(np.mean(np.abs(t2_err))),
        "t2_rmse": float(np.sqrt(np.mean(t2_err**2))),
        "m0_mae": float(np.mean(np.abs(m0_err))),
        "m0_rmse": float(np.sqrt(np.mean(m0_err**2))),
        "t2_rel_mae": float(np.mean(np.abs(t2_err) / t2_true)),
    }
    out_metrics.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    import pandas as pd

    df = pd.DataFrame({"t2_true": t2_true, "t2_hat": fitted_t2, "t2_err": t2_err})

    fig_a = (
        ggplot(df, aes(x="t2_true"))
        + geom_histogram(bins=30)
        + theme_bw()
        + labs(title="(A) Data check: T2 true distribution", x="T2 true [ms]", y="count")
    )
    ggsave(fig_a, filename=str(out_figures / "data_check__t2_true_hist.png"), verbose=False, dpi=150)

    fig_b = (
        ggplot(df, aes(x="t2_true", y="t2_hat"))
        + geom_point(alpha=0.6)
        + geom_abline(intercept=0.0, slope=1.0)
        + theme_bw()
        + labs(title="(B) Result: T2 fitted vs true", x="T2 true [ms]", y="T2 fitted [ms]")
    )
    ggsave(fig_b, filename=str(out_figures / "result__t2_true_vs_hat.png"), verbose=False, dpi=150)

    fig_c = (
        ggplot(df, aes(x="t2_err"))
        + geom_histogram(bins=30)
        + theme_bw()
        + labs(title="(C) Failure analysis: T2 residual distribution", x="T2 error (hat - true) [ms]", y="count")
    )
    ggsave(fig_c, filename=str(out_figures / "failure__t2_error_hist.png"), verbose=False, dpi=150)

    return {"metrics": metrics, "figures": [p.name for p in out_figures.glob("*.png")]}


@dataclass(frozen=True)
class VfaT1Config:
    flip_angle_deg: list[float]
    tr_s: float
    n_samples: int
    m0: float
    t1_min_s: float
    t1_max_s: float
    b1: float
    b1_range: tuple[float, float] | None
    noise_model: str
    noise_sigma: float
    seed: int
    robust_linear: bool
    huber_k: float
    min_signal: float | None


def _parse_vfa_t1_config(config: dict[str, object]) -> VfaT1Config:
    run_cfg = config.get("run", {})
    vfa_cfg = config.get("vfa_t1", {})
    if not isinstance(run_cfg, dict) or not isinstance(vfa_cfg, dict):
        raise ValueError("config format error: [run] and [vfa_t1] must be tables")

    fa = vfa_cfg.get("flip_angle_deg")
    if not isinstance(fa, list) or not all(isinstance(x, (int, float)) for x in fa):
        raise ValueError("vfa_t1.flip_angle_deg must be a list of numbers")
    tr_s = float(vfa_cfg.get("tr_s", 0.015))
    n_samples = int(vfa_cfg.get("n_samples", 200))
    m0 = float(vfa_cfg.get("m0", 2000.0))
    t1_range_s = vfa_cfg.get("t1_range_s", [0.2, 2.0])
    if (
        not isinstance(t1_range_s, list)
        or len(t1_range_s) != 2
        or not all(isinstance(x, (int, float)) for x in t1_range_s)
    ):
        raise ValueError("vfa_t1.t1_range_s must be [min, max]")
    t1_min_s, t1_max_s = float(t1_range_s[0]), float(t1_range_s[1])
    b1 = float(vfa_cfg.get("b1", 1.0))
    b1_range = vfa_cfg.get("b1_range")
    if b1_range is None:
        b1_range_parsed = None
    else:
        if (
            not isinstance(b1_range, list)
            or len(b1_range) != 2
            or not all(isinstance(x, (int, float)) for x in b1_range)
        ):
            raise ValueError("vfa_t1.b1_range must be [min, max]")
        b1_range_parsed = (float(b1_range[0]), float(b1_range[1]))
    noise_model = str(vfa_cfg.get("noise_model", "gaussian"))
    noise_sigma = float(vfa_cfg.get("noise_sigma", 0.0))
    robust_linear = bool(vfa_cfg.get("robust_linear", False))
    huber_k = float(vfa_cfg.get("huber_k", 1.345))
    min_signal = vfa_cfg.get("min_signal")
    min_signal_parsed = None if min_signal is None else float(min_signal)
    seed = int(run_cfg.get("seed", 0))
    return VfaT1Config(
        flip_angle_deg=[float(x) for x in fa],
        tr_s=tr_s,
        n_samples=n_samples,
        m0=m0,
        t1_min_s=t1_min_s,
        t1_max_s=t1_max_s,
        b1=b1,
        b1_range=b1_range_parsed,
        noise_model=noise_model,
        noise_sigma=noise_sigma,
        seed=seed,
        robust_linear=robust_linear,
        huber_k=huber_k,
        min_signal=min_signal_parsed,
    )


def _run_vfa_t1(cfg: VfaT1Config, *, out_metrics: Path, out_figures: Path) -> dict[str, object]:
    import numpy as np

    _require_plotnine()
    from plotnine import aes, geom_abline, geom_histogram, geom_point, ggplot, labs, theme_bw
    from plotnine import ggsave

    from qmrpy.models.t1 import VfaT1
    from qmrpy.sim.noise import add_gaussian_noise, add_rician_noise

    rng = np.random.default_rng(cfg.seed)
    t1_true = rng.uniform(cfg.t1_min_s, cfg.t1_max_s, size=cfg.n_samples).astype(float)
    m0_true = np.full(cfg.n_samples, cfg.m0, dtype=float)
    if cfg.b1_range is not None:
        b1_true = rng.uniform(cfg.b1_range[0], cfg.b1_range[1], size=cfg.n_samples).astype(float)
    else:
        b1_true = np.full(cfg.n_samples, cfg.b1, dtype=float)

    model_nominal = VfaT1(flip_angle_deg=np.array(cfg.flip_angle_deg, dtype=float), tr_s=cfg.tr_s, b1=1.0)
    signal_clean = np.stack(
        [
            VfaT1(
                flip_angle_deg=model_nominal.flip_angle_deg,
                tr_s=cfg.tr_s,
                b1=float(b1_true[i]),
            ).forward(m0=float(m0_true[i]), t1_s=float(t1_true[i]))
            for i in range(cfg.n_samples)
        ]
    )
    if cfg.noise_model == "gaussian":
        signal = add_gaussian_noise(signal_clean, sigma=cfg.noise_sigma, rng=rng)
    elif cfg.noise_model == "rician":
        signal = add_rician_noise(signal_clean, sigma=cfg.noise_sigma, rng=rng)
    else:
        raise ValueError(f"unknown noise_model for vfa_t1: {cfg.noise_model}")

    fitted_m0 = np.empty(cfg.n_samples, dtype=float)
    fitted_t1 = np.empty(cfg.n_samples, dtype=float)
    for i in range(cfg.n_samples):
        if cfg.min_signal is not None and float(np.max(signal[i])) < float(cfg.min_signal):
            fitted_m0[i] = np.nan
            fitted_t1[i] = np.nan
            continue
        fitted = VfaT1(
            flip_angle_deg=model_nominal.flip_angle_deg,
            tr_s=cfg.tr_s,
            b1=float(b1_true[i]),
        ).fit_linear(signal[i], robust=cfg.robust_linear, huber_k=cfg.huber_k)
        fitted_m0[i] = fitted["m0"]
        fitted_t1[i] = fitted["t1_s"]

    valid = np.isfinite(fitted_t1) & np.isfinite(fitted_m0)
    t1_err = fitted_t1[valid] - t1_true[valid]
    m0_err = fitted_m0[valid] - m0_true[valid]

    metrics = {
        "n_samples": int(cfg.n_samples),
        "n_valid": int(np.sum(valid)),
        "flip_angle_deg": [float(x) for x in cfg.flip_angle_deg],
        "tr_s": float(cfg.tr_s),
        "b1": float(cfg.b1),
        "b1_range": None if cfg.b1_range is None else [float(cfg.b1_range[0]), float(cfg.b1_range[1])],
        "noise_model": str(cfg.noise_model),
        "noise_sigma": float(cfg.noise_sigma),
        "robust_linear": bool(cfg.robust_linear),
        "huber_k": float(cfg.huber_k),
        "min_signal": cfg.min_signal,
        "t1_mae": float(np.mean(np.abs(t1_err))),
        "t1_rmse": float(np.sqrt(np.mean(t1_err**2))),
        "m0_mae": float(np.mean(np.abs(m0_err))),
        "m0_rmse": float(np.sqrt(np.mean(m0_err**2))),
        "t1_rel_mae": float(np.mean(np.abs(t1_err) / t1_true[valid])),
    }
    out_metrics.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    import pandas as pd

    df = pd.DataFrame({"t1_true": t1_true, "t1_hat": fitted_t1, "t1_err": t1_err})

    fig_a = (
        ggplot(df, aes(x="t1_true"))
        + geom_histogram(bins=30)
        + theme_bw()
        + labs(title="(A) Data check: T1 true distribution", x="T1 true [s]", y="count")
    )
    ggsave(fig_a, filename=str(out_figures / "data_check__t1_true_hist.png"), verbose=False, dpi=150)

    fig_b = (
        ggplot(df, aes(x="t1_true", y="t1_hat"))
        + geom_point(alpha=0.6)
        + geom_abline(intercept=0.0, slope=1.0)
        + theme_bw()
        + labs(title="(B) Result: T1 fitted vs true", x="T1 true [s]", y="T1 fitted [s]")
    )
    ggsave(fig_b, filename=str(out_figures / "result__t1_true_vs_hat.png"), verbose=False, dpi=150)

    fig_c = (
        ggplot(df, aes(x="t1_err"))
        + geom_histogram(bins=30)
        + theme_bw()
        + labs(title="(C) Failure analysis: T1 residual distribution", x="T1 error (hat - true) [s]", y="count")
    )
    ggsave(fig_c, filename=str(out_figures / "failure__t1_error_hist.png"), verbose=False, dpi=150)

    return {"metrics": metrics, "figures": [p.name for p in out_figures.glob("*.png")]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="configs/exp/*.toml")
    parser.add_argument("--run-id", type=str, default=None, help="YYYY-MM-DD_HHMMSS_tag (default: now)")
    parser.add_argument("--out-root", type=str, default="output/runs", help="output root (default: output/runs)")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = _read_toml(config_path)
    tag = str(config.get("run", {}).get("tag", "exp")) if isinstance(config.get("run", {}), dict) else "exp"

    run_id = args.run_id or _now_run_id(tag)
    run_dir = Path(args.out_root) / run_id
    paths = _ensure_dirs(run_dir)
    env_updates = _setup_runtime_caches(paths["env"])

    (paths["logs"] / "run.log").write_text("", encoding="utf-8")
    def log(msg: str) -> None:
        with (paths["logs"] / "run.log").open("a", encoding="utf-8") as f:
            f.write(msg.rstrip() + "\n")
        print(msg, flush=True)

    log(f"run_id={run_id}")
    log(f"config={config_path}")
    log(f"run_dir={run_dir}")

    shutil.copy2(config_path, paths["config_snapshot"] / config_path.name)

    run_cfg = config.get("run", {})
    model_name = str(run_cfg.get("model", "mono_t2")) if isinstance(run_cfg, dict) else "mono_t2"

    if model_name == "mono_t2":
        model_cfg = _parse_mono_t2_config(config)
        result = _run_mono_t2(
            model_cfg,
            out_metrics=paths["metrics"] / "mono_t2_metrics.json",
            out_figures=paths["figures"],
        )
        model_cfg_dict = asdict(model_cfg)
    elif model_name == "vfa_t1":
        model_cfg = _parse_vfa_t1_config(config)
        result = _run_vfa_t1(
            model_cfg,
            out_metrics=paths["metrics"] / "vfa_t1_metrics.json",
            out_figures=paths["figures"],
        )
        model_cfg_dict = asdict(model_cfg)
    else:
        raise ValueError(f"unknown model: {model_name}")

    run_json = {
        "run_id": run_id,
        "command": " ".join([shlex_quote(x) for x in [sys.executable, *sys.argv]]),
        "config": str(config_path),
        "config_snapshot": str((paths["config_snapshot"] / config_path.name).relative_to(run_dir)),
        "model": model_name,
        "seed": int(model_cfg_dict.get("seed", 0)),
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
        "model_config": model_cfg_dict,
        "result": result,
    }
    (run_dir / "run.json").write_text(json.dumps(run_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log("done")
    return 0


def shlex_quote(s: str) -> str:
    import shlex

    return shlex.quote(s)


if __name__ == "__main__":
    raise SystemExit(main())
