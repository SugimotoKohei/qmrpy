#!/usr/bin/env python3
"""Generate publication-ready figures for the qmrpy manuscript using plotnine.

Figures
-------
Figure 1 : Normalized validation margin (metric / threshold) for all core cases.
Figure 2 : Parameter recovery (estimate vs ground truth) for representative models.
Figure 3 : Synthetic 2D phantom mono-exponential T2 mapping (truth vs estimate).

All figures are regenerated deterministically from fixed seeds and from the
validation settings in ``configs/exp/validation_core.toml``.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    coord_flip,
    element_blank,
    element_text,
    facet_wrap,
    geom_abline,
    geom_hline,
    geom_point,
    geom_segment,
    geom_text,
    geom_tile,
    ggplot,
    labs,
    scale_fill_cmap,
    scale_shape_manual,
    scale_y_log10,
    theme,
    theme_minimal,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output" / "paper_figures"
REPORTS_DIR = ROOT_DIR / "output" / "reports" / "parity_summary"
CONFIG_PATH = ROOT_DIR / "configs" / "exp" / "validation_core.toml"

# Lower plotting bound for the log-scaled margin axis. Cases that recover the
# ground truth exactly (ratio == 0) are drawn at this floor and annotated.
RATIO_FLOOR = 1e-12

BASE_THEME = theme_minimal() + theme(
    text=element_text(family="sans-serif", size=9),
    axis_title=element_text(size=10),
    strip_text=element_text(size=9, weight="bold"),
    legend_title=element_text(size=9),
    legend_key_size=10,
)


def _load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("rb") as f:
        return tomllib.load(f)


def _save(plot: Any, stem: str, *, width: float, height: float) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUTPUT_DIR / f"{stem}.png"
    pdf = OUTPUT_DIR / f"{stem}.pdf"
    plot.save(str(png), width=width, height=height, dpi=300, verbose=False)
    plot.save(str(pdf), width=width, height=height, verbose=False)
    print(f"saved: {png.relative_to(ROOT_DIR)}")
    print(f"saved: {pdf.relative_to(ROOT_DIR)}")


def _margin_ratio(value: float, threshold: float) -> float:
    """Return metric/threshold, treating an exact-zero threshold as exact recovery."""
    if threshold > 0.0:
        return float(value) / float(threshold)
    return 0.0 if float(value) == 0.0 else float("inf")


def generate_figure1_validation_margin() -> None:
    """Figure 1: primary metric normalized by its pass threshold, log scale."""
    csv_path = REPORTS_DIR / "core_validation.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run scripts/summarize_parity.py --suite core first."
        )

    df = pd.read_csv(csv_path)
    df["ratio"] = [
        _margin_ratio(v, t)
        for v, t in zip(df["primary_value"], df["primary_threshold"], strict=True)
    ]
    df["plot_ratio"] = df["ratio"].clip(lower=RATIO_FLOOR)
    df["exactness"] = np.where(df["ratio"] == 0.0, "exact recovery", "finite margin")
    df["label"] = df["domain"] + ": " + df["model"]
    df["annotation"] = np.where(df["ratio"] == 0.0, "0 (exact)", "")

    df = df.sort_values(by=["domain", "model"], ascending=[False, False])
    df["label"] = pd.Categorical(df["label"], categories=df["label"].unique(), ordered=True)

    plot = (
        ggplot(df, aes(x="label", y="plot_ratio"))
        + geom_hline(yintercept=1.0, linetype="dashed", color="#c0392b", size=0.6)
        + geom_segment(aes(xend="label", color="domain"), yend=RATIO_FLOOR, size=0.7)
        + geom_point(aes(color="domain", shape="exactness"), size=2.6)
        + geom_text(aes(label="annotation"), nudge_x=0.38, size=6.5, color="#555555")
        + scale_y_log10(limits=(RATIO_FLOOR, 3.0))
        + scale_shape_manual(values={"exact recovery": "o", "finite margin": "D"})
        + coord_flip()
        + BASE_THEME
        + theme(panel_grid_major_y=element_blank(), legend_position="right")
        + labs(
            x="qMRI domain and model",
            y="Primary metric / pass threshold (log scale; dashed line = threshold)",
            color="Domain",
            shape="Case type",
        )
    )
    _save(plot, "fig1_validation_margin", width=8.0, height=6.0)


def _recover_vfa_t1(cfg: dict[str, Any], n: int) -> pd.DataFrame:
    from qmrpy.functional import fit_t1_vfa, simulate_t1_vfa
    from qmrpy.sim.noise import add_gaussian_noise

    c = cfg["core"]["vfa_t1"]
    rng = np.random.default_rng(20260211)
    fa = np.asarray(c["flip_angle_deg"], dtype=float)
    t1_true = np.linspace(c["t1_range_ms"][0], c["t1_range_ms"][1], n)

    rows = []
    for t1 in t1_true:
        sig = simulate_t1_vfa(m0=c["m0"], t1_ms=float(t1), flip_angle_deg=fa, tr_ms=c["tr_ms"])
        sig = add_gaussian_noise(sig, sigma=c["noise_sigma"], rng=rng)
        res = fit_t1_vfa(sig, flip_angle_deg=fa, tr_ms=c["tr_ms"])
        rows.append(
            {
                "panel": "VFA T1 (ms)",
                "truth": float(t1),
                "estimate": float(res["t1_ms"]),
            }
        )
    return pd.DataFrame(rows)


def _recover_t1rho(cfg: dict[str, Any], n: int) -> pd.DataFrame:
    from qmrpy.functional import fit_t1rho, simulate_t1rho
    from qmrpy.sim.noise import add_gaussian_noise

    c = cfg["core"]["t1rho"]
    rng = np.random.default_rng(20260212)
    tsl = np.asarray(c["tsl_ms"], dtype=float)
    truth = np.linspace(c["t1rho_range_ms"][0], c["t1rho_range_ms"][1], n)

    rows = []
    for value in truth:
        sig = simulate_t1rho(m0=c["m0"], t1rho_ms=float(value), tsl_ms=tsl)
        sig = add_gaussian_noise(sig, sigma=c["noise_sigma"], rng=rng)
        res = fit_t1rho(sig, tsl_ms=tsl)
        rows.append(
            {
                "panel": "Spin-lock T1rho (ms)",
                "truth": float(value),
                "estimate": float(res["t1rho_ms"]),
            }
        )
    return pd.DataFrame(rows)


def _recover_mono_t2(cfg: dict[str, Any], n: int) -> pd.DataFrame:
    from qmrpy.functional import fit_t2_mono, simulate_t2_mono
    from qmrpy.sim.noise import add_gaussian_noise

    c = cfg["core"]["mono_t2"]
    rng = np.random.default_rng(20260213)
    te = np.asarray(c["te_ms"], dtype=float)
    truth = np.linspace(c["t2_range_ms"][0], c["t2_range_ms"][1], n)

    rows = []
    for value in truth:
        sig = simulate_t2_mono(m0=c["m0"], t2_ms=float(value), te_ms=te)
        sig = add_gaussian_noise(sig, sigma=c["noise_sigma"], rng=rng)
        res = fit_t2_mono(sig, te_ms=te)
        rows.append(
            {
                "panel": "Mono-exponential T2 (ms)",
                "truth": float(value),
                "estimate": float(res["t2_ms"]),
            }
        )
    return pd.DataFrame(rows)


def _recover_mwf(cfg: dict[str, Any], n: int) -> pd.DataFrame:
    from qmrpy.models.t2 import T2MultiComponent
    from qmrpy.sim.noise import add_gaussian_noise

    c = cfg["core"]["mwf"]
    rng = np.random.default_rng(20260214)
    te = np.asarray(c["te_ms"], dtype=float)
    truth = np.linspace(c["mwf_range"][0], c["mwf_range"][1], n)
    model = T2MultiComponent(te_ms=te)

    rows = []
    for value in truth:
        sig = c["m0"] * (
            value * np.exp(-te / c["t2mw_ms"]) + (1.0 - value) * np.exp(-te / c["t2iew_ms"])
        )
        sig = add_gaussian_noise(sig, sigma=c["noise_sigma"], rng=rng)
        res = model.fit(
            sig,
            regularization_mode=c["regularization_mode"],
            qmrlab_sigma=c["qmrlab_sigma"],
            cutoff_ms=c["cutoff_ms"],
            upper_cutoff_iew_ms=c["upper_cutoff_iew_ms"],
        )
        rows.append(
            {
                "panel": "Myelin water fraction (-)",
                "truth": float(value),
                "estimate": float(res["params"]["mwf"]),
            }
        )
    return pd.DataFrame(rows)


def generate_figure2_parameter_recovery(n_points: int = 24) -> None:
    """Figure 2: estimate vs ground truth for four representative models."""
    cfg = _load_config()
    frames = [
        _recover_vfa_t1(cfg, n_points),
        _recover_t1rho(cfg, n_points),
        _recover_mono_t2(cfg, n_points),
        _recover_mwf(cfg, n_points),
    ]
    df = pd.concat(frames, ignore_index=True)
    order = [
        "VFA T1 (ms)",
        "Spin-lock T1rho (ms)",
        "Mono-exponential T2 (ms)",
        "Myelin water fraction (-)",
    ]
    df["panel"] = pd.Categorical(df["panel"], categories=order, ordered=True)

    stats = df.assign(
        abs_err=lambda d: (d["estimate"] - d["truth"]).abs(),
        rel_err=lambda d: (d["estimate"] - d["truth"]).abs() / d["truth"],
    ).groupby("panel", observed=True)
    for panel, group in stats:
        print(
            f"Figure 2 [{panel}] range={group['truth'].min():.4g}-{group['truth'].max():.4g} "
            f"MAE={group['abs_err'].mean():.4g} relMAE={group['rel_err'].mean():.4g}"
        )

    plot = (
        ggplot(df, aes(x="truth", y="estimate"))
        + geom_abline(slope=1.0, intercept=0.0, linetype="dashed", color="#c0392b", size=0.5)
        + geom_point(color="#2c6fbb", size=1.8, alpha=0.85)
        + facet_wrap("panel", scales="free", ncol=2)
        + BASE_THEME
        + theme(panel_spacing_y=0.08, panel_spacing_x=0.05)
        + labs(
            x="Ground-truth parameter value",
            y="qmrpy estimate",
        )
    )
    _save(plot, "fig2_parameter_recovery", width=7.0, height=5.6)


def generate_figure3_phantom_map(seed: int = 42) -> None:
    """Figure 3: synthetic 2D phantom mono-exponential T2 map (truth vs estimate)."""
    from qmrpy.models.t2 import T2Mono

    rng = np.random.default_rng(seed)
    shape = (64, 64)
    t2_true = np.zeros(shape, dtype=float)
    m0_true = np.zeros(shape, dtype=float)

    yy, xx = np.ogrid[: shape[0], : shape[1]]
    disks = [
        ((20, 20), 10, 30.0, 1000.0),
        ((44, 20), 10, 60.0, 1100.0),
        ((20, 44), 10, 90.0, 1200.0),
        ((44, 44), 10, 130.0, 1300.0),
    ]
    for (cx, cy), radius, t2, m0 in disks:
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
        t2_true[mask] = t2
        m0_true[mask] = m0

    te_ms = np.array([10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0, 120.0])
    active = t2_true > 0

    data = np.zeros((*shape, te_ms.size), dtype=float)
    for i, te in enumerate(te_ms):
        sig = np.zeros(shape, dtype=float)
        sig[active] = m0_true[active] * np.exp(-te / t2_true[active])
        data[..., i] = np.maximum(0.0, sig + rng.normal(0.0, 15.0, size=shape))

    maps = T2Mono(te_ms=te_ms).fit_image(data, mask=active, verbose=False)
    t2_est = np.asarray(maps["t2_ms"], dtype=float)

    mae = float(np.mean(np.abs(t2_est[active] - t2_true[active])))
    rel_mae = float(np.mean(np.abs(t2_est[active] - t2_true[active]) / t2_true[active]))
    print(f"Figure 3 phantom MAE: {mae:.4g} ms (relative {rel_mae:.4g})")

    ys, xs = np.nonzero(active)
    df = pd.concat(
        [
            pd.DataFrame({"x": xs, "y": ys, "t2": t2_true[ys, xs], "panel": "Ground truth"}),
            pd.DataFrame({"x": xs, "y": ys, "t2": t2_est[ys, xs], "panel": "qmrpy estimate"}),
        ],
        ignore_index=True,
    )
    df["panel"] = pd.Categorical(
        df["panel"], categories=["Ground truth", "qmrpy estimate"], ordered=True
    )

    plot = (
        ggplot(df, aes(x="x", y="y", fill="t2"))
        + geom_tile()
        + facet_wrap("panel", ncol=2)
        + scale_fill_cmap(cmap_name="viridis")
        + BASE_THEME
        + theme(
            axis_text=element_blank(),
            axis_ticks=element_blank(),
            panel_grid=element_blank(),
            aspect_ratio=1.0,
        )
        + labs(x="", y="", fill="T2 (ms)")
    )
    _save(plot, "fig3_t2_phantom_map", width=7.0, height=3.4)


def main() -> None:
    generate_figure1_validation_margin()
    generate_figure2_parameter_recovery()
    generate_figure3_phantom_map()


if __name__ == "__main__":
    main()
