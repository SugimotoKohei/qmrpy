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


def _find_run_dir(item: str) -> Path:
    p = Path(item)
    if p.is_dir():
        return p
    if p.name == "run.json":
        return p.parent
    return p


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
    parser.add_argument(
        "--t1-range",
        nargs=2,
        type=float,
        default=None,
        metavar=("MIN", "MAX"),
        help="fixed range for T1 binning (e.g. 0.2 2.0). If set, uses equal-width bins.",
    )
    parser.add_argument(
        "--t1-bins",
        type=int,
        default=10,
        help="number of fixed bins for T1 (used with --t1-range). default: 10",
    )
    parser.add_argument(
        "--t1-edges",
        type=str,
        default=None,
        help="explicit comma-separated T1 bin edges (overrides --t1-range/--t1-bins).",
    )
    parser.add_argument(
        "--b1-range",
        nargs=2,
        type=float,
        default=None,
        metavar=("MIN", "MAX"),
        help="fixed range for B1 binning (e.g. 0.8 1.2). If set, uses equal-width bins.",
    )
    parser.add_argument(
        "--b1-bins",
        type=int,
        default=10,
        help="number of fixed bins for B1 (used with --b1-range). default: 10",
    )
    parser.add_argument(
        "--b1-edges",
        type=str,
        default=None,
        help="explicit comma-separated B1 bin edges (overrides --b1-range/--b1-bins).",
    )
    args = parser.parse_args(argv)

    import pandas as pd

    rows: list[dict] = []
    per_sample_frames = []
    for item in args.runs:
        run_dir = _find_run_dir(item)
        run_json_path = run_dir / "run.json"
        run = _read_json(run_json_path)
        metrics = run.get("result", {}).get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}

        label = _label_from_run(run)
        row = {
            "run_id": run.get("run_id"),
            "model": run.get("model"),
            "label": label,
            "config": run.get("config"),
            "noise_model": run.get("model_config", {}).get("noise_model")
            if isinstance(run.get("model_config", {}), dict)
            else None,
            "noise_sigma": run.get("model_config", {}).get("noise_sigma")
            if isinstance(run.get("model_config", {}), dict)
            else None,
        }
        for k, v in metrics.items():
            row[k] = v
        rows.append(row)

        model = str(run.get("model", ""))
        if model == "vfa_t1":
            per_sample_path = run_dir / "metrics" / "vfa_t1_per_sample.csv"
        elif model == "mono_t2":
            per_sample_path = run_dir / "metrics" / "mono_t2_per_sample.csv"
        else:
            per_sample_path = None

        if per_sample_path is not None and per_sample_path.exists():
            df_ps = pd.read_csv(per_sample_path)
            df_ps["run_id"] = str(run.get("run_id"))
            df_ps["label"] = label
            df_ps["model"] = model
            per_sample_frames.append(df_ps)

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
    from plotnine import (
        aes,
        coord_flip,
        facet_wrap,
        geom_col,
        geom_histogram,
        geom_point,
        ggplot,
        labs,
        theme_bw,
    )
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

    if per_sample_frames:
        df_ps_all = pd.concat(per_sample_frames, ignore_index=True)
        # ensure numeric columns are treated as numeric for plotnine/stat_smooth
        for col in ["t1_true", "t1_hat", "t1_err", "b1_true", "n_points"]:
            if col in df_ps_all.columns:
                df_ps_all[col] = pd.to_numeric(df_ps_all[col], errors="coerce")

        if "t1_err" in df_ps_all.columns:
            fig5 = (
                ggplot(df_ps_all, aes(x="t1_err"))
                + geom_histogram(bins=40)
                + facet_wrap("label", scales="free_y")
                + theme_bw()
                + labs(title="(C) Failure analysis: T1 residual distribution", x="t1_err [s]", y="count")
            )
            ggsave(fig5, filename=str(figures_dir / "failure__t1_err_hist_by_run.png"), verbose=False, dpi=150)

        if "t1_true" in df_ps_all.columns and "t1_err" in df_ps_all.columns:
            df_ps_all["abs_t1_err"] = (df_ps_all["t1_err"]).abs()
            fig6 = (
                ggplot(df_ps_all, aes(x="t1_true", y="abs_t1_err"))
                + geom_point(alpha=0.35, size=1.0)
                + facet_wrap("label", scales="free_y")
                + theme_bw()
                + labs(title="(C) Failure analysis: |T1 error| vs T1 true", x="t1_true [s]", y="|t1_err| [s]")
            )
            ggsave(fig6, filename=str(figures_dir / "failure__abs_t1_err_vs_t1_true.png"), verbose=False, dpi=150)

        if "b1_true" in df_ps_all.columns and "t1_err" in df_ps_all.columns:
            if "abs_t1_err" not in df_ps_all.columns:
                df_ps_all["abs_t1_err"] = (df_ps_all["t1_err"]).abs()
            fig7 = (
                ggplot(df_ps_all, aes(x="b1_true", y="abs_t1_err"))
                + geom_point(alpha=0.35, size=1.0)
                + facet_wrap("label", scales="free_y")
                + theme_bw()
                + labs(title="(C) Failure analysis: |T1 error| vs B1", x="b1_true", y="|t1_err| [s]")
            )
            ggsave(fig7, filename=str(figures_dir / "failure__abs_t1_err_vs_b1.png"), verbose=False, dpi=150)

        # Stratified summaries (binning)
        if "abs_t1_err" in df_ps_all.columns and "t1_true" in df_ps_all.columns:
            df_bin = df_ps_all[["label", "t1_true", "abs_t1_err"]].dropna()
            if not df_bin.empty:
                t1_edges = None
                if args.t1_edges:
                    t1_edges = [float(x) for x in args.t1_edges.split(",") if x.strip()]
                elif args.t1_range is not None:
                    t1_min, t1_max = float(args.t1_range[0]), float(args.t1_range[1])
                    if t1_max <= t1_min:
                        raise SystemExit("--t1-range must satisfy MAX > MIN")
                    import numpy as np

                    t1_edges = np.linspace(t1_min, t1_max, int(args.t1_bins) + 1).tolist()

                if t1_edges is not None:
                    # fixed bins across runs
                    df_bin["t1_bin"] = pd.cut(df_bin["t1_true"], bins=t1_edges, include_lowest=True)
                else:
                    # default: quantile bins (stable within a run set)
                    n_bins = 10 if df_bin["t1_true"].nunique() >= 10 else max(2, int(df_bin["t1_true"].nunique()))
                    df_bin["t1_bin"] = pd.qcut(df_bin["t1_true"], q=n_bins, duplicates="drop")

                def p95(x):
                    return x.quantile(0.95)

                agg = (
                    df_bin.groupby(["label", "t1_bin"], dropna=False, observed=True)
                    .agg(
                        n=("abs_t1_err", "size"),
                        abs_t1_err_mean=("abs_t1_err", "mean"),
                        abs_t1_err_median=("abs_t1_err", "median"),
                        abs_t1_err_p95=("abs_t1_err", p95),
                    )
                    .reset_index()
                )
                # plotnine/matplotlib: avoid categorical bin objects causing axis warnings
                agg["t1_bin"] = agg["t1_bin"].astype(str)
                agg.to_csv(metrics_dir / "failure__abs_t1_err_by_t1_bin.csv", index=False)

                agg_plot = agg[agg["n"] > 0].copy()
                fig8 = (
                    ggplot(agg_plot, aes(x="t1_bin", y="abs_t1_err_mean"))
                    + geom_col()
                    + coord_flip()
                    + facet_wrap("label", scales="free_y")
                    + theme_bw()
                    + labs(
                        title="(C) Failure analysis: mean |T1 error| by T1 bin",
                        x="T1 true bins [s]",
                        y="mean |t1_err| [s]",
                    )
                )
                ggsave(fig8, filename=str(figures_dir / "failure__abs_t1_err_by_t1_bin.png"), verbose=False, dpi=150)

                fig8b = (
                    ggplot(agg_plot, aes(x="t1_bin", y="abs_t1_err_p95"))
                    + geom_col()
                    + coord_flip()
                    + facet_wrap("label", scales="free_y")
                    + theme_bw()
                    + labs(
                        title="(C) Failure analysis: p95 |T1 error| by T1 bin",
                        x="T1 true bins [s]",
                        y="p95 |t1_err| [s]",
                    )
                )
                ggsave(fig8b, filename=str(figures_dir / "failure__abs_t1_err_p95_by_t1_bin.png"), verbose=False, dpi=150)

        if "abs_t1_err" in df_ps_all.columns and "b1_true" in df_ps_all.columns:
            df_bin = df_ps_all[["label", "b1_true", "abs_t1_err"]].dropna()
            if not df_bin.empty:
                b1_edges = None
                if args.b1_edges:
                    b1_edges = [float(x) for x in args.b1_edges.split(",") if x.strip()]
                elif args.b1_range is not None:
                    b1_min, b1_max = float(args.b1_range[0]), float(args.b1_range[1])
                    if b1_max <= b1_min:
                        raise SystemExit("--b1-range must satisfy MAX > MIN")
                    import numpy as np

                    b1_edges = np.linspace(b1_min, b1_max, int(args.b1_bins) + 1).tolist()

                if b1_edges is not None:
                    df_bin["b1_bin"] = pd.cut(df_bin["b1_true"], bins=b1_edges, include_lowest=True)
                else:
                    n_bins = 10 if df_bin["b1_true"].nunique() >= 10 else max(2, int(df_bin["b1_true"].nunique()))
                    df_bin["b1_bin"] = pd.qcut(df_bin["b1_true"], q=n_bins, duplicates="drop")

                def p95(x):
                    return x.quantile(0.95)

                agg = (
                    df_bin.groupby(["label", "b1_bin"], dropna=False, observed=True)
                    .agg(
                        n=("abs_t1_err", "size"),
                        abs_t1_err_mean=("abs_t1_err", "mean"),
                        abs_t1_err_median=("abs_t1_err", "median"),
                        abs_t1_err_p95=("abs_t1_err", p95),
                    )
                    .reset_index()
                )
                agg["b1_bin"] = agg["b1_bin"].astype(str)
                agg.to_csv(metrics_dir / "failure__abs_t1_err_by_b1_bin.csv", index=False)

                agg_plot = agg[agg["n"] > 0].copy()
                fig9 = (
                    ggplot(agg_plot, aes(x="b1_bin", y="abs_t1_err_mean"))
                    + geom_col()
                    + coord_flip()
                    + facet_wrap("label", scales="free_y")
                    + theme_bw()
                    + labs(
                        title="(C) Failure analysis: mean |T1 error| by B1 bin",
                        x="B1 bins",
                        y="mean |t1_err| [s]",
                    )
                )
                ggsave(fig9, filename=str(figures_dir / "failure__abs_t1_err_by_b1_bin.png"), verbose=False, dpi=150)

                fig9b = (
                    ggplot(agg_plot, aes(x="b1_bin", y="abs_t1_err_p95"))
                    + geom_col()
                    + coord_flip()
                    + facet_wrap("label", scales="free_y")
                    + theme_bw()
                    + labs(
                        title="(C) Failure analysis: p95 |T1 error| by B1 bin",
                        x="B1 bins",
                        y="p95 |t1_err| [s]",
                    )
                )
                ggsave(fig9b, filename=str(figures_dir / "failure__abs_t1_err_p95_by_b1_bin.png"), verbose=False, dpi=150)

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
                "binning": {
                    "t1_range": args.t1_range,
                    "t1_bins": args.t1_bins,
                    "t1_edges": args.t1_edges,
                    "b1_range": args.b1_range,
                    "b1_bins": args.b1_bins,
                    "b1_edges": args.b1_edges,
                },
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
