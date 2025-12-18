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
    noise_sigma: float
    seed: int


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
    noise_sigma = float(mono_cfg.get("noise_sigma", 0.0))
    seed = int(run_cfg.get("seed", 0))

    return MonoT2Config(
        te_ms=[float(x) for x in te_ms],
        n_samples=n_samples,
        m0=m0,
        t2_min_ms=t2_min_ms,
        t2_max_ms=t2_max_ms,
        noise_sigma=noise_sigma,
        seed=seed,
    )


def _run_mono_t2(cfg: MonoT2Config, *, out_metrics: Path, out_figures: Path) -> dict[str, object]:
    import numpy as np

    _require_plotnine()
    from plotnine import aes, geom_abline, geom_histogram, geom_point, ggplot, labs, theme_bw
    from plotnine import ggsave

    from qmrpy.models.t2 import MonoT2

    rng = np.random.default_rng(cfg.seed)
    t2_true = rng.uniform(cfg.t2_min_ms, cfg.t2_max_ms, size=cfg.n_samples).astype(float)
    m0_true = np.full(cfg.n_samples, cfg.m0, dtype=float)

    model = MonoT2(te=np.array(cfg.te_ms, dtype=float))
    signal_clean = np.stack([model.forward(m0=float(m0_true[i]), t2=float(t2_true[i])) for i in range(cfg.n_samples)])
    noise = rng.normal(loc=0.0, scale=cfg.noise_sigma, size=signal_clean.shape)
    signal = signal_clean + noise

    fitted_m0 = np.empty(cfg.n_samples, dtype=float)
    fitted_t2 = np.empty(cfg.n_samples, dtype=float)
    for i in range(cfg.n_samples):
        fitted = model.fit(signal[i])
        fitted_m0[i] = fitted["m0"]
        fitted_t2[i] = fitted["t2"]

    t2_err = fitted_t2 - t2_true
    m0_err = fitted_m0 - m0_true

    metrics = {
        "n_samples": int(cfg.n_samples),
        "te_ms": [float(x) for x in cfg.te_ms],
        "noise_sigma": float(cfg.noise_sigma),
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
        + labs(title="(A) 入力確認: T2真値分布", x="T2 true [ms]", y="count")
    )
    ggsave(fig_a, filename=str(out_figures / "data_check__t2_true_hist.png"), verbose=False, dpi=150)

    fig_b = (
        ggplot(df, aes(x="t2_true", y="t2_hat"))
        + geom_point(alpha=0.6)
        + geom_abline(intercept=0.0, slope=1.0)
        + theme_bw()
        + labs(title="(B) 主要結果: T2 推定（真値 vs 推定）", x="T2 true [ms]", y="T2 fitted [ms]")
    )
    ggsave(fig_b, filename=str(out_figures / "result__t2_true_vs_hat.png"), verbose=False, dpi=150)

    fig_c = (
        ggplot(df, aes(x="t2_err"))
        + geom_histogram(bins=30)
        + theme_bw()
        + labs(title="(C) 失敗解析: T2 残差分布", x="T2 error (hat - true) [ms]", y="count")
    )
    ggsave(fig_c, filename=str(out_figures / "failure__t2_error_hist.png"), verbose=False, dpi=150)

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

    (paths["logs"] / "run.log").write_text("", encoding="utf-8")
    def log(msg: str) -> None:
        with (paths["logs"] / "run.log").open("a", encoding="utf-8") as f:
            f.write(msg.rstrip() + "\n")
        print(msg, flush=True)

    log(f"run_id={run_id}")
    log(f"config={config_path}")
    log(f"run_dir={run_dir}")

    shutil.copy2(config_path, paths["config_snapshot"] / config_path.name)

    mono_cfg = _parse_mono_t2_config(config)
    result = _run_mono_t2(
        mono_cfg,
        out_metrics=paths["metrics"] / "mono_t2_metrics.json",
        out_figures=paths["figures"],
    )

    run_json = {
        "run_id": run_id,
        "command": " ".join([shlex_quote(x) for x in [sys.executable, *sys.argv]]),
        "config": str(config_path),
        "config_snapshot": str((paths["config_snapshot"] / config_path.name).relative_to(run_dir)),
        "seed": mono_cfg.seed,
        "git": _git_info(),
        "env": {
            "python": sys.version,
            "platform": sys.platform,
        },
        "outputs": {
            "metrics": str((paths["metrics"]).relative_to(run_dir)),
            "figures": str((paths["figures"]).relative_to(run_dir)),
            "logs": str((paths["logs"]).relative_to(run_dir)),
            "artifacts": str((paths["artifacts"]).relative_to(run_dir)),
        },
        "mono_t2": asdict(mono_cfg),
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

