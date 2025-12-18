from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def _now_id(tag: str) -> str:
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d_%H%M%S")
    safe_tag = "".join(ch if (ch.isalnum() or ch in "-_") else "-" for ch in tag).strip("-")
    return f"{ts}_{safe_tag}" if safe_tag else ts


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def _label_from_run(run: dict) -> str:
    cfg = run.get("model_config", {}) if isinstance(run.get("model_config", {}), dict) else {}
    model = str(run.get("model", ""))
    noise = str(cfg.get("noise_model", ""))
    sigma = cfg.get("noise_sigma", None)
    b1_range = cfg.get("b1_range", None)
    robust = cfg.get("robust_linear", None)
    outlier = cfg.get("outlier_reject", None)

    parts = [model]
    if noise:
        parts.append(f"noise={noise}")
    if sigma is not None:
        parts.append(f"sigma={sigma}")
    if b1_range is not None:
        parts.append(f"b1_range={b1_range}")
    if robust:
        parts.append("robust")
    if outlier:
        parts.append("outlier")
    return ", ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="run.json paths or run directories (output/runs/<run_id>)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="output directory (default: output/reports/<timestamp>_compare)",
    )
    parser.add_argument("--tag", type=str, default="compare", help="tag for default output dir name")
    args = parser.parse_args(argv)

    import pandas as pd

    rows: list[dict] = []
    for item in args.runs:
        p = Path(item)
        run_json_path = p / "run.json" if p.is_dir() else p
        run = _read_json(run_json_path)
        metrics = run.get("result", {}).get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}

        row = {
            "run_id": run.get("run_id"),
            "model": run.get("model"),
            "label": _label_from_run(run),
            "config": run.get("config"),
            "noise_model": run.get("model_config", {}).get("noise_model") if isinstance(run.get("model_config", {}), dict) else None,
            "noise_sigma": run.get("model_config", {}).get("noise_sigma") if isinstance(run.get("model_config", {}), dict) else None,
        }
        for k, v in metrics.items():
            row[k] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no runs loaded")

    out_dir = Path(args.out) if args.out else (Path("output/reports") / _now_id(args.tag))
    metrics_dir = out_dir / "metrics"
    figures_dir = out_dir / "figures"
    env_dir = out_dir / "env"
    _ensure_dir(metrics_dir)
    _ensure_dir(figures_dir)
    _ensure_dir(env_dir)
    env_updates = _setup_runtime_caches(env_dir)

    df.to_csv(metrics_dir / "summary.csv", index=False)
    (metrics_dir / "summary.json").write_text(
        json.dumps({"rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # figures (at least 3)
    from plotnine import aes, coord_flip, geom_col, geom_point, ggplot, labs, theme_bw
    from plotnine import ggsave

    if "t1_rel_mae" in df.columns:
        fig1 = (
            ggplot(df, aes(x="label", y="t1_rel_mae"))
            + geom_col()
            + coord_flip()
            + theme_bw()
            + labs(title="(B) Compare: t1_rel_mae by run", x="run", y="t1_rel_mae")
        )
        ggsave(fig1, filename=str(figures_dir / "compare__t1_rel_mae.png"), verbose=False, dpi=150)

    if "t1_rmse" in df.columns and "noise_sigma" in df.columns:
        fig2 = (
            ggplot(df, aes(x="noise_sigma", y="t1_rmse"))
            + geom_point(size=3)
            + theme_bw()
            + labs(title="(B) Compare: t1_rmse vs noise_sigma", x="noise_sigma", y="t1_rmse")
        )
        ggsave(fig2, filename=str(figures_dir / "compare__t1_rmse_vs_noise_sigma.png"), verbose=False, dpi=150)

    if "n_valid" in df.columns:
        fig3 = (
            ggplot(df, aes(x="label", y="n_valid"))
            + geom_col()
            + coord_flip()
            + theme_bw()
            + labs(title="(C) Diagnostics: n_valid by run", x="run", y="n_valid")
        )
        ggsave(fig3, filename=str(figures_dir / "diagnostic__n_valid.png"), verbose=False, dpi=150)

    if "n_points_mean" in df.columns:
        fig4 = (
            ggplot(df, aes(x="label", y="n_points_mean"))
            + geom_col()
            + coord_flip()
            + theme_bw()
            + labs(title="(C) Diagnostics: mean used points", x="run", y="n_points_mean")
        )
        ggsave(fig4, filename=str(figures_dir / "diagnostic__n_points_mean.png"), verbose=False, dpi=150)

    (out_dir / "report.json").write_text(
        json.dumps(
            {
                "type": "compare_runs",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "runs": [str(r) for r in args.runs],
                "outputs": {
                    "metrics": str(metrics_dir),
                    "figures": str(figures_dir),
                    "env": str(env_dir),
                },
                "env_updates": env_updates,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
