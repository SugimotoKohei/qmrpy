#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).parent.parent
TEST_DATA_DIR = ROOT_DIR / "tests" / "data"
DEFAULT_OUT_DIR = ROOT_DIR / "output" / "reports" / "parity_summary"
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "exp" / "validation_core.toml"


CORE_CASE_COLUMNS = [
    "domain",
    "model",
    "case",
    "seed",
    "n_samples",
    "primary_metric",
    "primary_value",
    "primary_threshold",
    "primary_unit",
    "pass",
]

CORE_METRIC_COLUMNS = [
    "domain",
    "model",
    "case",
    "metric",
    "value",
    "threshold",
    "unit",
    "pass",
]

DECAES_COLUMNS = [
    "case",
    "reg",
    "alpha_abs_deg",
    "mu_abs",
    "chi2factor_abs",
    "dist_max_abs",
    "dist_rms",
    "n_te",
    "n_t2",
]

QMRLAB_COLUMNS = [
    "report_path",
    "mwf_abs_diff_pct",
    "t2mw_abs_diff_ms",
    "t2iew_abs_diff_ms",
    "noise_model",
    "noise_sigma",
    "cutoff_ms",
    "regularization_alpha",
]


def _load_csv_1d(path: Path) -> np.ndarray:
    x = np.loadtxt(path, delimiter=",")
    return np.atleast_1d(x).astype(np.float64)


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return f"{value:.6g}"
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row.get(col, "") for col in columns])


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_markdown_table(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join(["---"] * len(columns)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(_format_value(row.get(col)) for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def _parse_formats(formats_raw: str) -> set[str]:
    items = {x.strip().lower() for x in str(formats_raw).split(",") if x.strip()}
    valid = {"csv", "markdown", "json"}
    unknown = items - valid
    if unknown:
        raise ValueError(f"unknown formats: {sorted(unknown)}")
    if not items:
        return {"csv", "markdown", "json"}
    return items


def _load_validation_config(path: Path) -> dict[str, Any]:
    import tomllib

    if not path.exists():
        raise FileNotFoundError(f"validation config not found: {path}")
    with path.open("rb") as f:
        cfg = tomllib.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("validation config must be a TOML table")
    return cfg


def _as_float_list(value: Any, *, name: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    out: list[float] = []
    for x in value:
        if not isinstance(x, (int, float)):
            raise ValueError(f"{name} must contain only numbers")
        out.append(float(x))
    return out


def _as_pair(value: Any, *, name: str) -> tuple[float, float]:
    values = _as_float_list(value, name=name)
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly 2 numbers")
    return float(values[0]), float(values[1])


def _phase_diff(a_rad: np.ndarray, b_rad: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * (a_rad - b_rad))).astype(np.float64)


def _seed_for(base_seed: int, offset: int) -> int:
    return int(base_seed + 1009 * int(offset))


def _add_noise(signal: np.ndarray, *, model: str, sigma: float, rng: np.random.Generator) -> np.ndarray:
    from qmrpy.sim.noise import add_gaussian_noise, add_rician_noise

    sigma = float(sigma)
    if sigma < 0:
        raise ValueError("noise sigma must be >= 0")

    model_norm = str(model).lower().strip()
    if model_norm in {"none", "", "no"}:
        return np.asarray(signal, dtype=np.float64)
    if model_norm == "gaussian":
        return add_gaussian_noise(signal, sigma=sigma, rng=rng)
    if model_norm == "rician":
        return add_rician_noise(signal, sigma=sigma, rng=rng)
    raise ValueError(f"unknown noise_model: {model}")


def _metric_row(
    *,
    domain: str,
    model: str,
    case: str,
    metric: str,
    value: float,
    threshold: float,
    unit: str,
) -> dict[str, Any]:
    value_f = float(value)
    threshold_f = float(threshold)
    return {
        "domain": str(domain),
        "model": str(model),
        "case": str(case),
        "metric": str(metric),
        "value": value_f,
        "threshold": threshold_f,
        "unit": str(unit),
        "pass": int(value_f <= threshold_f),
    }


def _case_row(
    *,
    domain: str,
    model: str,
    case: str,
    seed: int,
    n_samples: int,
    primary_metric: str,
    metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    if not metrics:
        raise ValueError("metrics must not be empty")

    by_metric = {str(m["metric"]): m for m in metrics}
    if primary_metric not in by_metric:
        raise ValueError(f"primary_metric not found in metrics: {primary_metric}")

    primary = by_metric[primary_metric]
    status = int(all(int(m.get("pass", 0)) == 1 for m in metrics))

    return {
        "domain": str(domain),
        "model": str(model),
        "case": str(case),
        "seed": int(seed),
        "n_samples": int(n_samples),
        "primary_metric": str(primary_metric),
        "primary_value": float(primary["value"]),
        "primary_threshold": float(primary["threshold"]),
        "primary_unit": str(primary["unit"]),
        "pass": status,
    }


def _validate_mono_t2(cfg: dict[str, Any], *, seed: int, default_n_samples: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from qmrpy.models.t2 import T2Mono

    rng = np.random.default_rng(seed)
    n_samples = int(cfg.get("n_samples", default_n_samples))
    te_ms = np.asarray(_as_float_list(cfg.get("te_ms", [10.0, 20.0, 40.0, 80.0]), name="core.mono_t2.te_ms"))
    t2_lo, t2_hi = _as_pair(cfg.get("t2_range_ms", [30.0, 150.0]), name="core.mono_t2.t2_range_ms")
    m0 = float(cfg.get("m0", 1000.0))
    noise_model = str(cfg.get("noise_model", "gaussian"))
    noise_sigma = float(cfg.get("noise_sigma", 0.0))

    fit_type = str(cfg.get("fit_type", "exponential"))
    drop_first_echo = bool(cfg.get("drop_first_echo", False))
    offset_term = bool(cfg.get("offset_term", False))

    t2_true = rng.uniform(t2_lo, t2_hi, size=n_samples)
    m0_true = np.full(n_samples, m0, dtype=np.float64)

    model = T2Mono(te_ms=te_ms)
    signals = np.stack([model.forward(m0=float(m0_true[i]), t2_ms=float(t2_true[i])) for i in range(n_samples)])
    signals = _add_noise(signals, model=noise_model, sigma=noise_sigma, rng=rng)

    t2_hat = np.empty(n_samples, dtype=np.float64)
    m0_hat = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        out = model.fit(
            signals[i],
            fit_type=fit_type,
            drop_first_echo=drop_first_echo,
            offset_term=offset_term,
        )
        t2_hat[i] = float(out["params"]["t2_ms"])
        m0_hat[i] = float(out["params"]["m0"])

    t2_rel_mae = float(np.mean(np.abs(t2_hat - t2_true) / t2_true))
    m0_rel_mae = float(np.mean(np.abs(m0_hat - m0_true) / np.maximum(m0_true, 1e-12)))

    metrics = [
        _metric_row(
            domain="T2",
            model="mono_t2",
            case="mono_t2_gaussian",
            metric="t2_rel_mae",
            value=t2_rel_mae,
            threshold=float(cfg.get("threshold_t2_rel_mae", 0.1)),
            unit="ratio",
        ),
        _metric_row(
            domain="T2",
            model="mono_t2",
            case="mono_t2_gaussian",
            metric="m0_rel_mae",
            value=m0_rel_mae,
            threshold=float(cfg.get("threshold_m0_rel_mae", 0.1)),
            unit="ratio",
        ),
    ]
    case_row = _case_row(
        domain="T2",
        model="mono_t2",
        case="mono_t2_gaussian",
        seed=seed,
        n_samples=n_samples,
        primary_metric="t2_rel_mae",
        metrics=metrics,
    )
    return case_row, metrics


def _validate_vfa_t1(cfg: dict[str, Any], *, seed: int, default_n_samples: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from qmrpy.models.t1 import T1VFA

    rng = np.random.default_rng(seed)
    n_samples = int(cfg.get("n_samples", default_n_samples))

    flip_angle_deg = np.asarray(_as_float_list(cfg.get("flip_angle_deg", [3.0, 8.0, 15.0, 25.0]), name="core.vfa_t1.flip_angle_deg"))
    tr_ms = float(cfg.get("tr_ms", 15.0))
    m0 = float(cfg.get("m0", 2000.0))
    t1_lo, t1_hi = _as_pair(cfg.get("t1_range_ms", [500.0, 2000.0]), name="core.vfa_t1.t1_range_ms")
    b1_lo, b1_hi = _as_pair(cfg.get("b1_range", [0.9, 1.1]), name="core.vfa_t1.b1_range")

    noise_model = str(cfg.get("noise_model", "gaussian"))
    noise_sigma = float(cfg.get("noise_sigma", 0.0))
    robust = bool(cfg.get("robust", False))
    outlier_reject = bool(cfg.get("outlier_reject", False))

    t1_true = rng.uniform(t1_lo, t1_hi, size=n_samples)
    b1_true = rng.uniform(b1_lo, b1_hi, size=n_samples)
    m0_true = np.full(n_samples, m0, dtype=np.float64)

    t1_hat = np.empty(n_samples, dtype=np.float64)
    m0_hat = np.empty(n_samples, dtype=np.float64)

    for i in range(n_samples):
        model = T1VFA(flip_angle_deg=flip_angle_deg, tr_ms=tr_ms, b1=float(b1_true[i]))
        signal = model.forward(m0=float(m0_true[i]), t1_ms=float(t1_true[i]))
        signal_noisy = _add_noise(signal, model=noise_model, sigma=noise_sigma, rng=rng)
        out = model.fit(signal_noisy, robust=robust, outlier_reject=outlier_reject)
        t1_hat[i] = float(out["params"]["t1_ms"])
        m0_hat[i] = float(out["params"]["m0"])

    t1_rel_mae = float(np.mean(np.abs(t1_hat - t1_true) / t1_true))
    m0_rel_mae = float(np.mean(np.abs(m0_hat - m0_true) / np.maximum(m0_true, 1e-12)))

    metrics = [
        _metric_row(
            domain="T1",
            model="vfa_t1",
            case="vfa_t1_b1_range",
            metric="t1_rel_mae",
            value=t1_rel_mae,
            threshold=float(cfg.get("threshold_t1_rel_mae", 0.1)),
            unit="ratio",
        ),
        _metric_row(
            domain="T1",
            model="vfa_t1",
            case="vfa_t1_b1_range",
            metric="m0_rel_mae",
            value=m0_rel_mae,
            threshold=float(cfg.get("threshold_m0_rel_mae", 0.1)),
            unit="ratio",
        ),
    ]
    case_row = _case_row(
        domain="T1",
        model="vfa_t1",
        case="vfa_t1_b1_range",
        seed=seed,
        n_samples=n_samples,
        primary_metric="t1_rel_mae",
        metrics=metrics,
    )
    return case_row, metrics


def _validate_inversion_recovery(
    cfg: dict[str, Any], *, seed: int, default_n_samples: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from qmrpy.models.t1 import T1InversionRecovery

    rng = np.random.default_rng(seed)
    n_samples = int(cfg.get("n_samples", default_n_samples))

    ti_ms = np.asarray(
        _as_float_list(
            cfg.get("ti_ms", [350.0, 500.0, 650.0, 800.0, 950.0, 1100.0, 1250.0, 1400.0, 1700.0]),
            name="core.inversion_recovery.ti_ms",
        )
    )
    t1_lo, t1_hi = _as_pair(cfg.get("t1_range_ms", [500.0, 1800.0]), name="core.inversion_recovery.t1_range_ms")
    ra = float(cfg.get("ra", 500.0))
    rb = float(cfg.get("rb", -1000.0))
    noise_model = str(cfg.get("noise_model", "gaussian"))
    noise_sigma = float(cfg.get("noise_sigma", 0.0))

    method = str(cfg.get("method", "magnitude"))
    solver = str(cfg.get("solver", "rdnls"))

    model = T1InversionRecovery(ti_ms=ti_ms)
    t1_true = rng.uniform(t1_lo, t1_hi, size=n_samples)
    t1_hat = np.empty(n_samples, dtype=np.float64)

    magnitude = method.lower().strip() == "magnitude"
    for i in range(n_samples):
        signal = model.forward(t1_ms=float(t1_true[i]), ra=ra, rb=rb, magnitude=magnitude)
        signal_noisy = _add_noise(signal, model=noise_model, sigma=noise_sigma, rng=rng)
        out = model.fit(signal_noisy, method=method, solver=solver)
        t1_hat[i] = float(out["params"]["t1_ms"])

    t1_rel_mae = float(np.mean(np.abs(t1_hat - t1_true) / t1_true))
    metrics = [
        _metric_row(
            domain="T1",
            model="inversion_recovery",
            case="inversion_recovery_magnitude",
            metric="t1_rel_mae",
            value=t1_rel_mae,
            threshold=float(cfg.get("threshold_t1_rel_mae", 0.1)),
            unit="ratio",
        )
    ]
    case_row = _case_row(
        domain="T1",
        model="inversion_recovery",
        case="inversion_recovery_magnitude",
        seed=seed,
        n_samples=n_samples,
        primary_metric="t1_rel_mae",
        metrics=metrics,
    )
    return case_row, metrics


def _validate_epg_t2(cfg: dict[str, Any], *, seed: int, default_n_samples: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from qmrpy.models.t2 import T2EPG

    rng = np.random.default_rng(seed)
    n_samples = int(cfg.get("n_samples", default_n_samples))

    n_te = int(cfg.get("n_te", 16))
    te_ms = float(cfg.get("te_ms", 10.0))
    t1_ms = float(cfg.get("t1_ms", 1000.0))
    alpha_deg = float(cfg.get("alpha_deg", 180.0))
    beta_deg = float(cfg.get("beta_deg", 180.0))
    m0 = float(cfg.get("m0", 1000.0))

    t2_lo, t2_hi = _as_pair(cfg.get("t2_range_ms", [40.0, 120.0]), name="core.epg_t2.t2_range_ms")
    b1_lo, b1_hi = _as_pair(cfg.get("b1_range", [0.9, 1.1]), name="core.epg_t2.b1_range")

    noise_model = str(cfg.get("noise_model", "gaussian"))
    noise_sigma = float(cfg.get("noise_sigma", 0.0))

    model = T2EPG(n_te=n_te, te_ms=te_ms, t1_ms=t1_ms, alpha_deg=alpha_deg, beta_deg=beta_deg)

    t2_true = rng.uniform(t2_lo, t2_hi, size=n_samples)
    b1_true = rng.uniform(b1_lo, b1_hi, size=n_samples)
    m0_true = np.full(n_samples, m0, dtype=np.float64)

    t2_hat = np.empty(n_samples, dtype=np.float64)
    m0_hat = np.empty(n_samples, dtype=np.float64)

    for i in range(n_samples):
        signal = model.forward(m0=float(m0_true[i]), t2_ms=float(t2_true[i]), b1=float(b1_true[i]))
        signal_noisy = _add_noise(signal, model=noise_model, sigma=noise_sigma, rng=rng)
        out = model.fit(signal_noisy, b1=float(b1_true[i]))
        t2_hat[i] = float(out["params"]["t2_ms"])
        m0_hat[i] = float(out["params"]["m0"])

    t2_rel_mae = float(np.mean(np.abs(t2_hat - t2_true) / t2_true))
    m0_rel_mae = float(np.mean(np.abs(m0_hat - m0_true) / np.maximum(m0_true, 1e-12)))

    metrics = [
        _metric_row(
            domain="T2",
            model="epg_t2",
            case="epg_t2_b1_range",
            metric="t2_rel_mae",
            value=t2_rel_mae,
            threshold=float(cfg.get("threshold_t2_rel_mae", 0.12)),
            unit="ratio",
        ),
        _metric_row(
            domain="T2",
            model="epg_t2",
            case="epg_t2_b1_range",
            metric="m0_rel_mae",
            value=m0_rel_mae,
            threshold=float(cfg.get("threshold_m0_rel_mae", 0.12)),
            unit="ratio",
        ),
    ]
    case_row = _case_row(
        domain="T2",
        model="epg_t2",
        case="epg_t2_b1_range",
        seed=seed,
        n_samples=n_samples,
        primary_metric="t2_rel_mae",
        metrics=metrics,
    )
    return case_row, metrics


def _validate_mwf(cfg: dict[str, Any], *, seed: int, default_n_samples: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from qmrpy.models.t2 import T2MultiComponent

    rng = np.random.default_rng(seed)
    n_samples = int(cfg.get("n_samples", default_n_samples))

    te_ms = np.asarray(
        _as_float_list(
            cfg.get(
                "te_ms",
                [
                    10.0,
                    20.0,
                    30.0,
                    40.0,
                    50.0,
                    60.0,
                    70.0,
                    80.0,
                    90.0,
                    100.0,
                    110.0,
                    120.0,
                    130.0,
                    140.0,
                    150.0,
                    160.0,
                    170.0,
                    180.0,
                    190.0,
                    200.0,
                    210.0,
                    220.0,
                    230.0,
                    240.0,
                    250.0,
                    260.0,
                    270.0,
                    280.0,
                    290.0,
                    300.0,
                    310.0,
                    320.0,
                ],
            ),
            name="core.mwf.te_ms",
        )
    )

    m0 = float(cfg.get("m0", 1000.0))
    mwf_lo, mwf_hi = _as_pair(cfg.get("mwf_range", [0.08, 0.25]), name="core.mwf.mwf_range")
    t2mw_ms = float(cfg.get("t2mw_ms", 20.0))
    t2iew_ms = float(cfg.get("t2iew_ms", 80.0))

    noise_model = str(cfg.get("noise_model", "gaussian"))
    noise_sigma = float(cfg.get("noise_sigma", 0.0))

    regularization_mode = str(cfg.get("regularization_mode", "qmrlab_regnnls"))
    qmrlab_sigma = float(cfg.get("qmrlab_sigma", 20.0))
    cutoff_ms = float(cfg.get("cutoff_ms", 40.0))
    upper_cutoff_iew_ms = float(cfg.get("upper_cutoff_iew_ms", 200.0))

    mwf_true = rng.uniform(mwf_lo, mwf_hi, size=n_samples)

    model = T2MultiComponent(te_ms=te_ms)

    mwf_hat = np.empty(n_samples, dtype=np.float64)
    t2mw_hat = np.empty(n_samples, dtype=np.float64)
    t2iew_hat = np.empty(n_samples, dtype=np.float64)

    for i in range(n_samples):
        signal = m0 * (
            mwf_true[i] * np.exp(-te_ms / t2mw_ms) + (1.0 - mwf_true[i]) * np.exp(-te_ms / t2iew_ms)
        )
        signal_noisy = _add_noise(signal, model=noise_model, sigma=noise_sigma, rng=rng)
        out = model.fit(
            signal_noisy,
            regularization_mode=regularization_mode,
            qmrlab_sigma=qmrlab_sigma,
            cutoff_ms=cutoff_ms,
            upper_cutoff_iew_ms=upper_cutoff_iew_ms,
        )
        mwf_hat[i] = float(out["params"]["mwf"])
        t2mw_hat[i] = float(out["params"]["t2mw_ms"])
        t2iew_hat[i] = float(out["params"]["t2iew_ms"])

    mwf_mae_abs = float(np.mean(np.abs(mwf_hat - mwf_true)))
    t2mw_mae_ms = float(np.mean(np.abs(t2mw_hat - t2mw_ms)))
    t2iew_mae_ms = float(np.mean(np.abs(t2iew_hat - t2iew_ms)))

    metrics = [
        _metric_row(
            domain="T2",
            model="mwf",
            case="mwf_two_component",
            metric="mwf_mae_abs",
            value=mwf_mae_abs,
            threshold=float(cfg.get("threshold_mwf_mae_abs", 0.08)),
            unit="fraction",
        ),
        _metric_row(
            domain="T2",
            model="mwf",
            case="mwf_two_component",
            metric="t2mw_mae_ms",
            value=t2mw_mae_ms,
            threshold=float(cfg.get("threshold_t2mw_mae_ms", 25.0)),
            unit="ms",
        ),
        _metric_row(
            domain="T2",
            model="mwf",
            case="mwf_two_component",
            metric="t2iew_mae_ms",
            value=t2iew_mae_ms,
            threshold=float(cfg.get("threshold_t2iew_mae_ms", 25.0)),
            unit="ms",
        ),
    ]
    case_row = _case_row(
        domain="T2",
        model="mwf",
        case="mwf_two_component",
        seed=seed,
        n_samples=n_samples,
        primary_metric="mwf_mae_abs",
        metrics=metrics,
    )
    return case_row, metrics


def _validate_b1_dam(cfg: dict[str, Any], *, seed: int, default_n_samples: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from qmrpy.models.b1 import B1DAM

    rng = np.random.default_rng(seed)
    n_samples = int(cfg.get("n_samples", default_n_samples))

    alpha_deg = float(cfg.get("alpha_deg", 60.0))
    m0 = float(cfg.get("m0", 1000.0))
    b1_lo, b1_hi = _as_pair(cfg.get("b1_range", [0.7, 1.3]), name="core.b1_dam.b1_range")
    noise_model = str(cfg.get("noise_model", "gaussian"))
    noise_sigma = float(cfg.get("noise_sigma", 0.0))

    model = B1DAM(alpha_deg=alpha_deg)

    b1_true = rng.uniform(b1_lo, b1_hi, size=n_samples)
    b1_hat = np.empty(n_samples, dtype=np.float64)
    spurious = np.empty(n_samples, dtype=np.float64)

    for i in range(n_samples):
        signal = model.forward(m0=m0, b1=float(b1_true[i]))
        signal_noisy = _add_noise(signal, model=noise_model, sigma=noise_sigma, rng=rng)
        out = model.fit(signal_noisy)
        b1_hat[i] = float(out["params"]["b1_raw"])
        spurious[i] = float(out["diagnostics"]["spurious"])

    b1_mae_abs = float(np.nanmean(np.abs(b1_hat - b1_true)))
    spurious_rate = float(np.nanmean(spurious))

    metrics = [
        _metric_row(
            domain="B1",
            model="b1_dam",
            case="b1_dam_noise",
            metric="b1_mae_abs",
            value=b1_mae_abs,
            threshold=float(cfg.get("threshold_b1_mae_abs", 0.1)),
            unit="ratio",
        ),
        _metric_row(
            domain="B1",
            model="b1_dam",
            case="b1_dam_noise",
            metric="spurious_rate",
            value=spurious_rate,
            threshold=float(cfg.get("threshold_spurious_rate", 0.1)),
            unit="ratio",
        ),
    ]
    case_row = _case_row(
        domain="B1",
        model="b1_dam",
        case="b1_dam_noise",
        seed=seed,
        n_samples=n_samples,
        primary_metric="b1_mae_abs",
        metrics=metrics,
    )
    return case_row, metrics


def _validate_qsm(cfg: dict[str, Any], *, seed: int, default_n_samples: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from qmrpy.models.qsm import QSMSplitBregman

    rng = np.random.default_rng(seed)
    _ = default_n_samples

    shape_vals = cfg.get("shape", [6, 6, 6])
    if not isinstance(shape_vals, list) or len(shape_vals) != 3:
        raise ValueError("core.qsm.shape must be [sx, sy, sz]")
    shape = tuple(int(v) for v in shape_vals)

    pad_vals = cfg.get("pad_size", [1, 1, 1])
    if not isinstance(pad_vals, list) or len(pad_vals) != 3:
        raise ValueError("core.qsm.pad_size must be [px, py, pz]")
    pad_size = tuple(int(v) for v in pad_vals)

    lambda_l2 = float(cfg.get("lambda_l2", 1e-2))

    phase = rng.normal(0.0, 1.0, size=shape)
    mask = np.ones(shape, dtype=np.float64)

    model = QSMSplitBregman(
        sharp_filter=False,
        l1_regularized=False,
        l2_regularized=True,
        no_regularization=False,
        pad_size=pad_size,
        lambda_l2=lambda_l2,
    )

    out1 = model.fit(phase=phase, mask=mask, image_resolution_mm=np.array([1.0, 1.0, 1.0]))
    out2 = model.fit(phase=phase, mask=mask, image_resolution_mm=np.array([1.0, 1.0, 1.0]))

    chi1 = np.asarray(out1["params"]["chi_l2"], dtype=np.float64)
    chi2 = np.asarray(out2["params"]["chi_l2"], dtype=np.float64)

    repro_rmse = float(np.sqrt(np.mean((chi1 - chi2) ** 2)))
    nan_ratio = float(1.0 - np.mean(np.isfinite(chi1)))

    metrics = [
        _metric_row(
            domain="QSM",
            model="qsm",
            case="qsm_l2_reproducibility",
            metric="chi_l2_repro_rmse",
            value=repro_rmse,
            threshold=float(cfg.get("threshold_repro_rmse", 1e-12)),
            unit="a.u.",
        ),
        _metric_row(
            domain="QSM",
            model="qsm",
            case="qsm_l2_reproducibility",
            metric="chi_l2_nan_ratio",
            value=nan_ratio,
            threshold=float(cfg.get("threshold_nan_ratio", 0.0)),
            unit="ratio",
        ),
    ]
    case_row = _case_row(
        domain="QSM",
        model="qsm",
        case="qsm_l2_reproducibility",
        seed=seed,
        n_samples=1,
        primary_metric="chi_l2_repro_rmse",
        metrics=metrics,
    )
    return case_row, metrics


def _validate_simulation(
    cfg: dict[str, Any], *, seed: int, default_n_samples: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from qmrpy.models.t2 import T2Mono
    from qmrpy.sim import sensitivity_analysis

    _ = default_n_samples
    rng = np.random.default_rng(seed)

    te_ms = np.asarray(_as_float_list(cfg.get("te_ms", [10.0, 20.0, 40.0, 80.0]), name="core.simulation.te_ms"))
    t2_nominal_ms = float(cfg.get("t2_nominal_ms", 60.0))
    t2_lo, t2_hi = _as_pair(cfg.get("t2_range_ms", [40.0, 100.0]), name="core.simulation.t2_range_ms")
    m0 = float(cfg.get("m0", 1000.0))
    n_steps = int(cfg.get("n_steps", 5))
    n_runs = int(cfg.get("n_runs", 8))

    noise_model = str(cfg.get("noise_model", "gaussian"))
    noise_sigma = float(cfg.get("noise_sigma", 0.0))
    noise_snr_raw = cfg.get("noise_snr", 80.0)
    noise_snr = None if noise_snr_raw is None else float(noise_snr_raw)

    model = T2Mono(te_ms=te_ms)
    out = sensitivity_analysis(
        model,
        nominal_params={"m0": m0, "t2_ms": t2_nominal_ms},
        vary_param="t2_ms",
        lb=t2_lo,
        ub=t2_hi,
        n_steps=n_steps,
        n_runs=n_runs,
        simulation_backend="analytic",
        noise_model=noise_model,
        noise_sigma=noise_sigma,
        noise_snr=noise_snr,
        rng=rng,
    )

    x_true = np.asarray(out["x"], dtype=np.float64)
    t2_hat_mean = np.asarray(out["mean"]["params"]["t2_ms"], dtype=np.float64)
    t2_hat_std = np.asarray(out["std"]["params"]["t2_ms"], dtype=np.float64)

    t2_rel_mae = float(np.mean(np.abs(t2_hat_mean - x_true) / x_true))
    t2_std_mean_ms = float(np.mean(t2_hat_std))

    metrics = [
        _metric_row(
            domain="Simulation",
            model="simulation",
            case="simulation_sensitivity_mono_t2",
            metric="t2_rel_mae",
            value=t2_rel_mae,
            threshold=float(cfg.get("threshold_t2_rel_mae", 0.15)),
            unit="ratio",
        ),
        _metric_row(
            domain="Simulation",
            model="simulation",
            case="simulation_sensitivity_mono_t2",
            metric="t2_std_mean_ms",
            value=t2_std_mean_ms,
            threshold=float(cfg.get("threshold_t2_std_mean_ms", 20.0)),
            unit="ms",
        ),
    ]
    case_row = _case_row(
        domain="Simulation",
        model="simulation",
        case="simulation_sensitivity_mono_t2",
        seed=seed,
        n_samples=int(n_steps * n_runs),
        primary_metric="t2_rel_mae",
        metrics=metrics,
    )
    return case_row, metrics


def _validate_b0_dual_echo(
    cfg: dict[str, Any], *, seed: int, default_n_samples: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from qmrpy.models import B0DualEcho

    rng = np.random.default_rng(seed)
    n_samples = int(cfg.get("n_samples", default_n_samples))

    te1_ms = float(cfg.get("te1_ms", 4.0))
    te2_ms = float(cfg.get("te2_ms", 6.0))
    b0_lo, b0_hi = _as_pair(cfg.get("b0_range_hz", [-80.0, 80.0]), name="core.b0_dual_echo.b0_range_hz")
    phase_noise = float(cfg.get("phase_noise_rad", 0.0))

    model = B0DualEcho(te1_ms=te1_ms, te2_ms=te2_ms)
    b0_true = rng.uniform(b0_lo, b0_hi, size=n_samples)
    phase0_true = rng.uniform(-np.pi, np.pi, size=n_samples)

    b0_hat = np.empty(n_samples, dtype=np.float64)
    phase0_hat = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        phase = model.forward(b0_hz=float(b0_true[i]), phase0_rad=float(phase0_true[i]))
        if phase_noise > 0:
            phase = np.angle(np.exp(1j * (phase + rng.normal(0.0, phase_noise, size=2))))
        signal = np.exp(1j * phase)
        out = model.fit(signal)
        b0_hat[i] = float(out["params"]["b0_hz"])
        phase0_hat[i] = float(out["params"]["phase0_rad"])

    b0_mae_hz = float(np.mean(np.abs(b0_hat - b0_true)))
    phase0_mae = float(np.mean(np.abs(_phase_diff(phase0_hat, phase0_true))))

    metrics = [
        _metric_row(
            domain="B0",
            model="b0_dual_echo",
            case="b0_dual_echo_phase_diff",
            metric="b0_mae_hz",
            value=b0_mae_hz,
            threshold=float(cfg.get("threshold_b0_mae_hz", 2.0)),
            unit="Hz",
        ),
        _metric_row(
            domain="B0",
            model="b0_dual_echo",
            case="b0_dual_echo_phase_diff",
            metric="phase0_mae_rad",
            value=phase0_mae,
            threshold=float(cfg.get("threshold_phase0_mae_rad", 0.2)),
            unit="rad",
        ),
    ]
    case_row = _case_row(
        domain="B0",
        model="b0_dual_echo",
        case="b0_dual_echo_phase_diff",
        seed=seed,
        n_samples=n_samples,
        primary_metric="b0_mae_hz",
        metrics=metrics,
    )
    return case_row, metrics


def _validate_b0_multi_echo(
    cfg: dict[str, Any], *, seed: int, default_n_samples: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from qmrpy.models import B0MultiEcho

    rng = np.random.default_rng(seed)
    n_samples = int(cfg.get("n_samples", default_n_samples))

    te_ms = np.asarray(_as_float_list(cfg.get("te_ms", [4.0, 8.0, 12.0, 16.0]), name="core.b0_multi_echo.te_ms"))
    b0_lo, b0_hi = _as_pair(cfg.get("b0_range_hz", [-120.0, 120.0]), name="core.b0_multi_echo.b0_range_hz")
    phase_noise = float(cfg.get("phase_noise_rad", 0.0))

    model = B0MultiEcho(te_ms=te_ms, unwrap_phase=True)
    b0_true = rng.uniform(b0_lo, b0_hi, size=n_samples)
    phase0_true = rng.uniform(-np.pi, np.pi, size=n_samples)
    te_s = te_ms / 1000.0

    b0_hat = np.empty(n_samples, dtype=np.float64)
    phase0_hat = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        phase = phase0_true[i] + 2.0 * np.pi * b0_true[i] * te_s
        phase = np.angle(np.exp(1j * phase))
        if phase_noise > 0:
            phase = np.angle(np.exp(1j * (phase + rng.normal(0.0, phase_noise, size=te_ms.size))))
        out = model.fit(phase)
        b0_hat[i] = float(out["params"]["b0_hz"])
        phase0_hat[i] = float(out["params"]["phase0_rad"])

    b0_mae_hz = float(np.mean(np.abs(b0_hat - b0_true)))
    phase0_mae = float(np.mean(np.abs(_phase_diff(phase0_hat, phase0_true))))

    metrics = [
        _metric_row(
            domain="B0",
            model="b0_multi_echo",
            case="b0_multi_echo_linear_phase",
            metric="b0_mae_hz",
            value=b0_mae_hz,
            threshold=float(cfg.get("threshold_b0_mae_hz", 1.5)),
            unit="Hz",
        ),
        _metric_row(
            domain="B0",
            model="b0_multi_echo",
            case="b0_multi_echo_linear_phase",
            metric="phase0_mae_rad",
            value=phase0_mae,
            threshold=float(cfg.get("threshold_phase0_mae_rad", 0.2)),
            unit="rad",
        ),
    ]
    case_row = _case_row(
        domain="B0",
        model="b0_multi_echo",
        case="b0_multi_echo_linear_phase",
        seed=seed,
        n_samples=n_samples,
        primary_metric="b0_mae_hz",
        metrics=metrics,
    )
    return case_row, metrics


def _validate_b1_bloch_siegert(
    cfg: dict[str, Any], *, seed: int, default_n_samples: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from qmrpy.models import B1BlochSiegert

    rng = np.random.default_rng(seed)
    n_samples = int(cfg.get("n_samples", default_n_samples))

    k_bs = float(cfg.get("k_bs_rad_per_b1sq", 2.0))
    b1_lo, b1_hi = _as_pair(cfg.get("b1_range", [0.2, 1.0]), name="core.b1_bloch_siegert.b1_range")
    phase_noise = float(cfg.get("phase_noise_rad", 0.0))

    model = B1BlochSiegert(k_bs_rad_per_b1sq=k_bs)
    b1_true = rng.uniform(b1_lo, b1_hi, size=n_samples)

    b1_hat = np.empty(n_samples, dtype=np.float64)
    spurious = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        phase = model.forward(b1=float(b1_true[i]), phase0_rad=float(rng.uniform(-np.pi, np.pi)))
        if phase_noise > 0:
            phase = np.angle(np.exp(1j * (phase + rng.normal(0.0, phase_noise, size=2))))
        out = model.fit(np.exp(1j * phase))
        b1_hat[i] = float(out["params"]["b1_raw"])
        spurious[i] = float(out["diagnostics"]["spurious"])

    b1_mae = float(np.mean(np.abs(b1_hat - b1_true)))
    spurious_rate = float(np.mean(spurious > 0.5))

    metrics = [
        _metric_row(
            domain="B1",
            model="b1_bloch_siegert",
            case="b1_bloch_siegert_phase",
            metric="b1_mae_abs",
            value=b1_mae,
            threshold=float(cfg.get("threshold_b1_mae_abs", 0.06)),
            unit="ratio",
        ),
        _metric_row(
            domain="B1",
            model="b1_bloch_siegert",
            case="b1_bloch_siegert_phase",
            metric="spurious_rate",
            value=spurious_rate,
            threshold=float(cfg.get("threshold_spurious_rate", 0.1)),
            unit="ratio",
        ),
    ]
    case_row = _case_row(
        domain="B1",
        model="b1_bloch_siegert",
        case="b1_bloch_siegert_phase",
        seed=seed,
        n_samples=n_samples,
        primary_metric="b1_mae_abs",
        metrics=metrics,
    )
    return case_row, metrics


def _validate_r2star_mono(
    cfg: dict[str, Any], *, seed: int, default_n_samples: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from qmrpy.models import T2StarMonoR2

    rng = np.random.default_rng(seed)
    n_samples = int(cfg.get("n_samples", default_n_samples))

    te_ms = np.asarray(_as_float_list(cfg.get("te_ms", [4.0, 8.0, 12.0, 16.0, 20.0]), name="core.r2star_mono.te_ms"))
    t2_lo, t2_hi = _as_pair(cfg.get("t2star_range_ms", [12.0, 50.0]), name="core.r2star_mono.t2star_range_ms")
    s0 = float(cfg.get("s0", 1000.0))
    noise_model = str(cfg.get("noise_model", "gaussian"))
    noise_sigma = float(cfg.get("noise_sigma", 0.0))

    model = T2StarMonoR2(te_ms=te_ms)
    t2_true = rng.uniform(t2_lo, t2_hi, size=n_samples)
    t2_hat = np.empty(n_samples, dtype=np.float64)
    r2_hat = np.empty(n_samples, dtype=np.float64)

    for i in range(n_samples):
        signal = model.forward(s0=s0, t2star_ms=float(t2_true[i]))
        signal_noisy = _add_noise(signal, model=noise_model, sigma=noise_sigma, rng=rng)
        out = model.fit(signal_noisy)
        t2_hat[i] = float(out["params"]["t2star_ms"])
        r2_hat[i] = float(out["params"]["r2star_hz"])

    t2_rel_mae = float(np.mean(np.abs(t2_hat - t2_true) / t2_true))
    r2_true = 1000.0 / t2_true
    r2_rel_mae = float(np.mean(np.abs(r2_hat - r2_true) / r2_true))

    metrics = [
        _metric_row(
            domain="T2*",
            model="r2star_mono",
            case="r2star_mono_magnitude",
            metric="t2star_rel_mae",
            value=t2_rel_mae,
            threshold=float(cfg.get("threshold_t2star_rel_mae", 0.08)),
            unit="ratio",
        ),
        _metric_row(
            domain="T2*",
            model="r2star_mono",
            case="r2star_mono_magnitude",
            metric="r2star_rel_mae",
            value=r2_rel_mae,
            threshold=float(cfg.get("threshold_r2star_rel_mae", 0.08)),
            unit="ratio",
        ),
    ]
    case_row = _case_row(
        domain="T2*",
        model="r2star_mono",
        case="r2star_mono_magnitude",
        seed=seed,
        n_samples=n_samples,
        primary_metric="t2star_rel_mae",
        metrics=metrics,
    )
    return case_row, metrics


def _validate_r2star_complex(
    cfg: dict[str, Any], *, seed: int, default_n_samples: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from qmrpy.models import T2StarComplexR2

    rng = np.random.default_rng(seed)
    n_samples = int(cfg.get("n_samples", default_n_samples))

    te_ms = np.asarray(
        _as_float_list(cfg.get("te_ms", [4.0, 8.0, 12.0, 16.0, 20.0, 24.0]), name="core.r2star_complex.te_ms")
    )
    t2_lo, t2_hi = _as_pair(cfg.get("t2star_range_ms", [12.0, 50.0]), name="core.r2star_complex.t2star_range_ms")
    df_lo, df_hi = _as_pair(cfg.get("delta_f_range_hz", [-30.0, 30.0]), name="core.r2star_complex.delta_f_range_hz")
    s0 = float(cfg.get("s0", 600.0))
    noise_sigma = float(cfg.get("noise_sigma", 0.0))

    model = T2StarComplexR2(te_ms=te_ms)
    t2_true = rng.uniform(t2_lo, t2_hi, size=n_samples)
    df_true = rng.uniform(df_lo, df_hi, size=n_samples)

    t2_hat = np.empty(n_samples, dtype=np.float64)
    df_hat = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        signal = model.forward(
            s0=s0,
            t2star_ms=float(t2_true[i]),
            delta_f_hz=float(df_true[i]),
            phi0_rad=float(rng.uniform(-np.pi, np.pi)),
        )
        if noise_sigma > 0:
            noise = rng.normal(0.0, noise_sigma, size=signal.shape) + 1j * rng.normal(
                0.0, noise_sigma, size=signal.shape
            )
            signal = signal + noise
        out = model.fit(signal)
        t2_hat[i] = float(out["params"]["t2star_ms"])
        df_hat[i] = float(out["params"]["delta_f_hz"])

    t2_rel_mae = float(np.mean(np.abs(t2_hat - t2_true) / t2_true))
    df_mae_hz = float(np.mean(np.abs(df_hat - df_true)))

    metrics = [
        _metric_row(
            domain="T2*",
            model="r2star_complex",
            case="r2star_complex_phase",
            metric="t2star_rel_mae",
            value=t2_rel_mae,
            threshold=float(cfg.get("threshold_t2star_rel_mae", 0.10)),
            unit="ratio",
        ),
        _metric_row(
            domain="T2*",
            model="r2star_complex",
            case="r2star_complex_phase",
            metric="delta_f_mae_hz",
            value=df_mae_hz,
            threshold=float(cfg.get("threshold_delta_f_mae_hz", 2.0)),
            unit="Hz",
        ),
    ]
    case_row = _case_row(
        domain="T2*",
        model="r2star_complex",
        case="r2star_complex_phase",
        seed=seed,
        n_samples=n_samples,
        primary_metric="t2star_rel_mae",
        metrics=metrics,
    )
    return case_row, metrics


def _validate_despot1_hifi(
    cfg: dict[str, Any], *, seed: int, default_n_samples: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from qmrpy.models import T1DESPOT1HIFI, T1InversionRecovery

    rng = np.random.default_rng(seed)
    n_samples = int(cfg.get("n_samples", default_n_samples))
    fa = np.asarray(_as_float_list(cfg.get("flip_angle_deg", [3.0, 10.0, 18.0]), name="core.despot1_hifi.flip_angle_deg"))
    tr_ms = float(cfg.get("tr_ms", 18.0))
    t1_lo, t1_hi = _as_pair(cfg.get("t1_range_ms", [500.0, 2000.0]), name="core.despot1_hifi.t1_range_ms")
    b1_lo, b1_hi = _as_pair(cfg.get("b1_range", [0.85, 1.15]), name="core.despot1_hifi.b1_range")
    m0 = float(cfg.get("m0", 1200.0))
    ti_ms = np.asarray(_as_float_list(cfg.get("ti_ms", [200.0, 500.0, 900.0, 1500.0, 2500.0]), name="core.despot1_hifi.ti_ms"))
    noise_model = str(cfg.get("noise_model", "gaussian"))
    noise_sigma = float(cfg.get("noise_sigma", 2.0))

    model = T1DESPOT1HIFI(flip_angle_deg=fa, tr_ms=tr_ms)
    ir_model = T1InversionRecovery(ti_ms=ti_ms)

    t1_true = rng.uniform(t1_lo, t1_hi, size=n_samples)
    b1_true = rng.uniform(b1_lo, b1_hi, size=n_samples)

    t1_hat = np.empty(n_samples, dtype=np.float64)
    b1_hat = np.empty(n_samples, dtype=np.float64)
    m0_hat = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        sig_vfa = model.forward(m0=m0, t1_ms=float(t1_true[i]), b1=float(b1_true[i]))
        sig_ir = ir_model.forward(t1_ms=float(t1_true[i]), ra=m0, rb=-2.0 * m0, magnitude=True)
        sig_vfa = _add_noise(sig_vfa, model=noise_model, sigma=noise_sigma, rng=rng)
        sig_ir = _add_noise(sig_ir, model=noise_model, sigma=noise_sigma, rng=rng)
        out = model.fit(sig_vfa, estimate_b1=True, ir_signal=sig_ir, ti_ms=ti_ms)
        t1_hat[i] = float(out["params"]["t1_ms"])
        b1_hat[i] = float(out["params"]["b1"])
        m0_hat[i] = float(out["params"]["m0"])

    t1_rel_mae = float(np.mean(np.abs(t1_hat - t1_true) / t1_true))
    b1_mae = float(np.mean(np.abs(b1_hat - b1_true)))
    m0_rel_mae = float(np.mean(np.abs(m0_hat - m0) / max(m0, 1e-12)))

    metrics = [
        _metric_row(
            domain="T1",
            model="despot1_hifi",
            case="despot1_hifi_joint",
            metric="t1_rel_mae",
            value=t1_rel_mae,
            threshold=float(cfg.get("threshold_t1_rel_mae", 0.10)),
            unit="ratio",
        ),
        _metric_row(
            domain="T1",
            model="despot1_hifi",
            case="despot1_hifi_joint",
            metric="b1_mae_abs",
            value=b1_mae,
            threshold=float(cfg.get("threshold_b1_mae_abs", 0.08)),
            unit="ratio",
        ),
        _metric_row(
            domain="T1",
            model="despot1_hifi",
            case="despot1_hifi_joint",
            metric="m0_rel_mae",
            value=m0_rel_mae,
            threshold=float(cfg.get("threshold_m0_rel_mae", 0.2)),
            unit="ratio",
        ),
    ]
    case_row = _case_row(
        domain="T1",
        model="despot1_hifi",
        case="despot1_hifi_joint",
        seed=seed,
        n_samples=n_samples,
        primary_metric="t1_rel_mae",
        metrics=metrics,
    )
    return case_row, metrics


def _validate_mp2rage(
    cfg: dict[str, Any], *, seed: int, default_n_samples: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from qmrpy.models import T1MP2RAGE

    rng = np.random.default_rng(seed)
    n_samples = int(cfg.get("n_samples", default_n_samples))

    ti1_ms = float(cfg.get("ti1_ms", 700.0))
    ti2_ms = float(cfg.get("ti2_ms", 2500.0))
    alpha1_deg = float(cfg.get("alpha1_deg", 4.0))
    alpha2_deg = float(cfg.get("alpha2_deg", 5.0))
    t1_lo, t1_hi = _as_pair(cfg.get("t1_range_ms", [500.0, 2200.0]), name="core.mp2rage.t1_range_ms")
    b1_lo, b1_hi = _as_pair(cfg.get("b1_range", [0.9, 1.1]), name="core.mp2rage.b1_range")
    m0 = float(cfg.get("m0", 900.0))
    noise_model = str(cfg.get("noise_model", "gaussian"))
    noise_sigma = float(cfg.get("noise_sigma", 1.0))
    method = str(cfg.get("method", "nls"))

    model = T1MP2RAGE(ti1_ms=ti1_ms, ti2_ms=ti2_ms, alpha1_deg=alpha1_deg, alpha2_deg=alpha2_deg)
    t1_true = rng.uniform(t1_lo, t1_hi, size=n_samples)
    b1_true = rng.uniform(b1_lo, b1_hi, size=n_samples)

    t1_hat = np.empty(n_samples, dtype=np.float64)
    b1_hat = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        signal = model.forward(m0=m0, t1_ms=float(t1_true[i]), b1=float(b1_true[i]))
        signal = _add_noise(signal, model=noise_model, sigma=noise_sigma, rng=rng)
        out = model.fit(signal, method=method, estimate_b1=True)
        t1_hat[i] = float(out["params"]["t1_ms"])
        b1_hat[i] = float(out["params"]["b1"])

    t1_rel_mae = float(np.mean(np.abs(t1_hat - t1_true) / t1_true))
    b1_mae = float(np.mean(np.abs(b1_hat - b1_true)))

    metrics = [
        _metric_row(
            domain="T1",
            model="mp2rage",
            case="mp2rage_inv_pair",
            metric="t1_rel_mae",
            value=t1_rel_mae,
            threshold=float(cfg.get("threshold_t1_rel_mae", 0.12)),
            unit="ratio",
        ),
        _metric_row(
            domain="T1",
            model="mp2rage",
            case="mp2rage_inv_pair",
            metric="b1_mae_abs",
            value=b1_mae,
            threshold=float(cfg.get("threshold_b1_mae_abs", 0.08)),
            unit="ratio",
        ),
    ]
    case_row = _case_row(
        domain="T1",
        model="mp2rage",
        case="mp2rage_inv_pair",
        seed=seed,
        n_samples=n_samples,
        primary_metric="t1_rel_mae",
        metrics=metrics,
    )
    return case_row, metrics


def _validate_emc_t2(
    cfg: dict[str, Any], *, seed: int, default_n_samples: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from qmrpy.models import T2EMC

    rng = np.random.default_rng(seed)
    n_samples = int(cfg.get("n_samples", default_n_samples))

    n_te = int(cfg.get("n_te", 16))
    te_ms = float(cfg.get("te_ms", 10.0))
    t1_ms = float(cfg.get("t1_ms", 1000.0))
    alpha_deg = float(cfg.get("alpha_deg", 180.0))
    beta_deg = float(cfg.get("beta_deg", 180.0))
    m0 = float(cfg.get("m0", 1000.0))
    t2_lo, t2_hi = _as_pair(cfg.get("t2_range_ms", [40.0, 120.0]), name="core.emc_t2.t2_range_ms")
    b1_lo, b1_hi = _as_pair(cfg.get("b1_range", [0.85, 1.15]), name="core.emc_t2.b1_range")
    noise_model = str(cfg.get("noise_model", "gaussian"))
    noise_sigma = float(cfg.get("noise_sigma", 3.0))

    model = T2EMC(n_te=n_te, te_ms=te_ms, t1_ms=t1_ms, alpha_deg=alpha_deg, beta_deg=beta_deg)
    t2_true = rng.uniform(t2_lo, t2_hi, size=n_samples)
    b1_true = rng.uniform(b1_lo, b1_hi, size=n_samples)

    t2_hat = np.empty(n_samples, dtype=np.float64)
    b1_hat = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        signal = model.forward(m0=m0, t2_ms=float(t2_true[i]), b1=float(b1_true[i]))
        signal = _add_noise(signal, model=noise_model, sigma=noise_sigma, rng=rng)
        out = model.fit(signal, estimate_b1=True)
        t2_hat[i] = float(out["params"]["t2_ms"])
        b1_hat[i] = float(out["params"]["b1"])

    t2_rel_mae = float(np.mean(np.abs(t2_hat - t2_true) / t2_true))
    b1_mae = float(np.mean(np.abs(b1_hat - b1_true)))

    metrics = [
        _metric_row(
            domain="T2",
            model="emc_t2",
            case="emc_t2_joint_b1",
            metric="t2_rel_mae",
            value=t2_rel_mae,
            threshold=float(cfg.get("threshold_t2_rel_mae", 0.12)),
            unit="ratio",
        ),
        _metric_row(
            domain="T2",
            model="emc_t2",
            case="emc_t2_joint_b1",
            metric="b1_mae_abs",
            value=b1_mae,
            threshold=float(cfg.get("threshold_b1_mae_abs", 0.10)),
            unit="ratio",
        ),
    ]
    case_row = _case_row(
        domain="T2",
        model="emc_t2",
        case="emc_t2_joint_b1",
        seed=seed,
        n_samples=n_samples,
        primary_metric="t2_rel_mae",
        metrics=metrics,
    )
    return case_row, metrics


def _core_validation_rows(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    core_cfg = config.get("core", {})
    if not isinstance(core_cfg, dict):
        raise ValueError("[core] section is required in validation config")

    meta_cfg = config.get("meta", {})
    run_cfg = config.get("run", {})
    if not isinstance(meta_cfg, dict):
        meta_cfg = {}
    if not isinstance(run_cfg, dict):
        run_cfg = {}

    base_seed = int(meta_cfg.get("seed", 0))
    default_n_samples = int(run_cfg.get("n_samples", 48))

    validators: list[tuple[str, Any]] = [
        ("mono_t2", _validate_mono_t2),
        ("r2star_mono", _validate_r2star_mono),
        ("r2star_complex", _validate_r2star_complex),
        ("vfa_t1", _validate_vfa_t1),
        ("despot1_hifi", _validate_despot1_hifi),
        ("mp2rage", _validate_mp2rage),
        ("inversion_recovery", _validate_inversion_recovery),
        ("epg_t2", _validate_epg_t2),
        ("emc_t2", _validate_emc_t2),
        ("mwf", _validate_mwf),
        ("b1_dam", _validate_b1_dam),
        ("b1_bloch_siegert", _validate_b1_bloch_siegert),
        ("b0_dual_echo", _validate_b0_dual_echo),
        ("b0_multi_echo", _validate_b0_multi_echo),
        ("qsm", _validate_qsm),
        ("simulation", _validate_simulation),
    ]

    case_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    offset = 0
    for name, validator in validators:
        model_cfg = core_cfg.get(name, {})
        if not isinstance(model_cfg, dict):
            raise ValueError(f"[core.{name}] must be a table")
        if not bool(model_cfg.get("enabled", True)):
            continue

        seed = _seed_for(base_seed, offset)
        offset += 1

        case_row, metrics = validator(model_cfg, seed=seed, default_n_samples=default_n_samples)
        case_rows.append(case_row)
        metric_rows.extend(metrics)

    case_rows.sort(key=lambda r: (r["domain"], r["model"], r["case"]))
    metric_rows.sort(key=lambda r: (r["domain"], r["model"], r["case"], r["metric"]))
    return case_rows, metric_rows


def _decaes_parity_rows() -> list[dict[str, Any]]:
    from qmrpy.models.t2.decaes_t2 import T2DECAESMap

    rows: list[dict[str, Any]] = []
    signal = _load_csv_1d(TEST_DATA_DIR / "decaes_ref_signal.csv")

    # Case 1: fixed flip angle, no regularization
    t2times = _load_csv_1d(TEST_DATA_DIR / "decaes_ref_t2times.csv") * 1000.0
    echotimes = _load_csv_1d(TEST_DATA_DIR / "decaes_ref_echotimes.csv") * 1000.0
    alpha_ref = float(_load_csv_1d(TEST_DATA_DIR / "decaes_ref_alpha.csv")[0])
    dist_ref = _load_csv_1d(TEST_DATA_DIR / "decaes_ref_dist.csv")

    model = T2DECAESMap(
        n_te=len(echotimes),
        te_ms=float(echotimes[0]),
        n_t2=len(t2times),
        t2_range_ms=(float(t2times[0]), float(t2times[-1])),
        reg="none",
        set_flip_angle_deg=180.0,
        epg_backend="decaes",
    )
    out = model.fit(signal)
    dist_diff = out["params"]["distribution"] - dist_ref
    rows.append(
        {
            "case": "decaes_fixed_flip_no_reg",
            "reg": "none",
            "alpha_abs_deg": abs(float(out["params"]["alpha_deg"]) - alpha_ref),
            "mu_abs": None,
            "chi2factor_abs": None,
            "dist_max_abs": float(np.max(np.abs(dist_diff))),
            "dist_rms": float(np.sqrt(np.mean(dist_diff**2))),
            "n_te": len(echotimes),
            "n_t2": len(t2times),
        }
    )

    # Case 2: GCV + flip angle optimization
    alpha_ref = float(_load_csv_1d(TEST_DATA_DIR / "decaes_ref2_alpha.csv")[0])
    mu_ref = float(_load_csv_1d(TEST_DATA_DIR / "decaes_ref2_mu.csv")[0])
    dist_ref = _load_csv_1d(TEST_DATA_DIR / "decaes_ref2_dist.csv")
    model = T2DECAESMap(
        n_te=16,
        te_ms=10.0,
        n_t2=30,
        t2_range_ms=(10.0, 2000.0),
        reg="gcv",
        n_ref_angles=64,
        save_reg_param=True,
        epg_backend="decaes",
    )
    out = model.fit(signal)
    dist_diff = out["params"]["distribution"] - dist_ref
    rows.append(
        {
            "case": "decaes_opt_alpha_gcv",
            "reg": "gcv",
            "alpha_abs_deg": abs(float(out["params"]["alpha_deg"]) - alpha_ref),
            "mu_abs": abs(float(out["mu"]) - mu_ref),
            "chi2factor_abs": None,
            "dist_max_abs": float(np.max(np.abs(dist_diff))),
            "dist_rms": float(np.sqrt(np.mean(dist_diff**2))),
            "n_te": 16,
            "n_t2": 30,
        }
    )

    def _reg_case(reg: str, *, chi2_factor: float | None = None, noise_level: float | None = None) -> None:
        alpha_ref = float(_load_csv_1d(TEST_DATA_DIR / f"decaes_ref_{reg}_alpha.csv")[0])
        mu_ref = float(_load_csv_1d(TEST_DATA_DIR / f"decaes_ref_{reg}_mu.csv")[0])
        chi2_ref = float(_load_csv_1d(TEST_DATA_DIR / f"decaes_ref_{reg}_chi2factor.csv")[0])
        dist_ref = _load_csv_1d(TEST_DATA_DIR / f"decaes_ref_{reg}_dist.csv")
        model = T2DECAESMap(
            n_te=16,
            te_ms=10.0,
            n_t2=30,
            t2_range_ms=(10.0, 2000.0),
            reg=reg,
            n_ref_angles=64,
            save_reg_param=True,
            epg_backend="decaes",
            chi2_factor=chi2_factor,
            noise_level=noise_level,
        )
        out = model.fit(signal)
        dist_diff = out["params"]["distribution"] - dist_ref
        rows.append(
            {
                "case": f"decaes_opt_alpha_{reg}",
                "reg": reg,
                "alpha_abs_deg": abs(float(out["params"]["alpha_deg"]) - alpha_ref),
                "mu_abs": abs(float(out["mu"]) - mu_ref),
                "chi2factor_abs": abs(float(out["chi2factor"]) - chi2_ref),
                "dist_max_abs": float(np.max(np.abs(dist_diff))),
                "dist_rms": float(np.sqrt(np.mean(dist_diff**2))),
                "n_te": 16,
                "n_t2": 30,
            }
        )

    _reg_case("lcurve")
    _reg_case("chi2", chi2_factor=1.02)
    _reg_case("mdp", noise_level=1e-3)
    return rows


def _collect_qmrlab_mwf_reports() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    roots = [
        ROOT_DIR / "output" / "reports" / "qmrlab_parity",
        ROOT_DIR / "output" / "reports" / "qmrlab_parity_sweeps",
    ]
    report_paths: list[Path] = []
    for root in roots:
        if root.exists():
            report_paths.extend(sorted(root.glob("**/report.json")))

    for path in report_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        diff = payload.get("diff", {})
        case = payload.get("case", {})
        qmrpy = payload.get("qmrpy", {})
        rows.append(
            {
                "report_path": str(path),
                "mwf_abs_diff_pct": abs(float(diff.get("mwf_percent", float("nan")))),
                "t2mw_abs_diff_ms": abs(float(diff.get("t2mw_ms", float("nan")))),
                "t2iew_abs_diff_ms": abs(float(diff.get("t2iew_ms", float("nan")))),
                "noise_model": case.get("noise_model"),
                "noise_sigma": case.get("noise_sigma"),
                "cutoff_ms": case.get("cutoff_ms"),
                "regularization_alpha": qmrpy.get("regularization_alpha"),
            }
        )

    rows.sort(
        key=lambda r: (
            r.get("mwf_abs_diff_pct", float("inf")),
            r.get("t2mw_abs_diff_ms", float("inf")),
        )
    )
    return rows


def _suite_summary_from_case_rows(case_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not case_rows:
        return {
            "n_cases": 0,
            "n_pass": 0,
            "n_fail": 0,
            "pass_rate": float("nan"),
            "domains": [],
        }
    n_cases = len(case_rows)
    n_pass = int(sum(int(row.get("pass", 0)) for row in case_rows))
    domains = sorted({str(row.get("domain", "")) for row in case_rows})
    return {
        "n_cases": n_cases,
        "n_pass": n_pass,
        "n_fail": int(n_cases - n_pass),
        "pass_rate": float(n_pass / n_cases),
        "domains": domains,
    }


def _write_summary_markdown(
    out_dir: Path,
    *,
    core_case_rows: list[dict[str, Any]] | None,
    decaes_rows: list[dict[str, Any]] | None,
    qmrlab_rows: list[dict[str, Any]] | None,
) -> Path:
    lines: list[str] = []
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines.append("# Validation Summary")
    lines.append(f"Generated: {ts}")
    lines.append("")

    if core_case_rows is not None:
        lines.append("## Core validation (external dependency free)")
        core_summary = _suite_summary_from_case_rows(core_case_rows)
        lines.append(f"- cases: {core_summary['n_cases']}")
        lines.append(f"- pass: {core_summary['n_pass']}")
        lines.append(f"- fail: {core_summary['n_fail']}")
        lines.append(f"- domains: {', '.join(core_summary['domains'])}")
        core_md = out_dir / "core_validation.md"
        _write_markdown_table(core_md, core_case_rows, CORE_CASE_COLUMNS)
        lines.append(f"- Table: {_safe_relpath(core_md)}")
        lines.append("")

    if decaes_rows is not None:
        lines.append("## DECAES.jl parity (qmrpy vs reference)")
        decaes_md = out_dir / "decaes_parity.md"
        _write_markdown_table(decaes_md, decaes_rows, DECAES_COLUMNS)
        lines.append(f"- Table: {_safe_relpath(decaes_md)}")
        lines.append("")

    if qmrlab_rows is not None:
        lines.append("## qMRLab MWF parity (existing report.json only)")
        qmrlab_md = out_dir / "qmrlab_mwf_parity.md"
        _write_markdown_table(qmrlab_md, qmrlab_rows, QMRLAB_COLUMNS)
        lines.append(f"- Table: {_safe_relpath(qmrlab_md)}")
        lines.append("")

    summary_path = out_dir / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Summarize validation/parity results into CSV/Markdown/JSON tables."
    )
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Validation config TOML path")
    p.add_argument(
        "--suite",
        type=str,
        choices=["all", "core", "decaes", "qmrlab"],
        default="all",
        help="Validation suite selection",
    )
    p.add_argument(
        "--formats",
        type=str,
        default="csv,markdown,json",
        help="Comma-separated output formats: csv,markdown,json",
    )

    # Backward-compatible switches
    p.add_argument("--no-decaes", action="store_true", help="Skip DECAES parity computation.")
    p.add_argument("--no-qmrlab", action="store_true", help="Skip qMRLab report collection.")

    args = p.parse_args(argv)

    formats = _parse_formats(args.formats)
    write_csv = "csv" in formats
    write_md = "markdown" in formats
    write_json = "json" in formats

    run_core = args.suite in {"all", "core"}
    run_decaes = args.suite in {"all", "decaes"} and not args.no_decaes
    run_qmrlab = args.suite in {"all", "qmrlab"} and not args.no_qmrlab

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    written_files: list[Path] = []

    core_case_rows: list[dict[str, Any]] | None = None
    core_metric_rows: list[dict[str, Any]] | None = None
    validation_config: dict[str, Any] | None = None
    if run_core:
        validation_config = _load_validation_config(args.config)
        core_case_rows, core_metric_rows = _core_validation_rows(validation_config)

        if write_csv:
            core_case_csv = out_dir / "core_validation.csv"
            core_metrics_csv = out_dir / "core_validation_metrics.csv"
            _write_csv(core_case_csv, core_case_rows, CORE_CASE_COLUMNS)
            _write_csv(core_metrics_csv, core_metric_rows, CORE_METRIC_COLUMNS)
            written_files.extend([core_case_csv, core_metrics_csv])

        if write_json:
            core_json = out_dir / "core_validation.json"
            _write_json(
                core_json,
                {
                    "config_path": str(args.config),
                    "meta": validation_config.get("meta", {}) if validation_config else {},
                    "cases": core_case_rows,
                    "metrics": core_metric_rows,
                },
            )
            written_files.append(core_json)

        if write_md:
            core_md = out_dir / "core_validation.md"
            _write_markdown_table(core_md, core_case_rows, CORE_CASE_COLUMNS)
            written_files.append(core_md)

    decaes_rows: list[dict[str, Any]] | None = None
    if run_decaes:
        decaes_rows = _decaes_parity_rows()
        if write_csv:
            decaes_csv = out_dir / "decaes_parity.csv"
            _write_csv(decaes_csv, decaes_rows, DECAES_COLUMNS)
            written_files.append(decaes_csv)
        if write_json:
            decaes_json = out_dir / "decaes_parity.json"
            _write_json(decaes_json, {"rows": decaes_rows})
            written_files.append(decaes_json)
        if write_md:
            decaes_md = out_dir / "decaes_parity.md"
            _write_markdown_table(decaes_md, decaes_rows, DECAES_COLUMNS)
            written_files.append(decaes_md)

    qmrlab_rows: list[dict[str, Any]] | None = None
    if run_qmrlab:
        qmrlab_rows = _collect_qmrlab_mwf_reports()
        if not qmrlab_rows:
            qmrlab_rows = None
        else:
            if write_csv:
                qmrlab_csv = out_dir / "qmrlab_mwf_parity.csv"
                _write_csv(qmrlab_csv, qmrlab_rows, QMRLAB_COLUMNS)
                written_files.append(qmrlab_csv)
            if write_json:
                qmrlab_json = out_dir / "qmrlab_mwf_parity.json"
                _write_json(qmrlab_json, {"rows": qmrlab_rows})
                written_files.append(qmrlab_json)
            if write_md:
                qmrlab_md = out_dir / "qmrlab_mwf_parity.md"
                _write_markdown_table(qmrlab_md, qmrlab_rows, QMRLAB_COLUMNS)
                written_files.append(qmrlab_md)

    summary_payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "out_dir": str(out_dir),
        "suite": str(args.suite),
        "formats": sorted(formats),
        "suites": {},
    }

    if core_case_rows is not None:
        summary_payload["suites"]["core"] = _suite_summary_from_case_rows(core_case_rows)
    if decaes_rows is not None:
        summary_payload["suites"]["decaes"] = {"n_cases": len(decaes_rows)}
    if qmrlab_rows is not None:
        summary_payload["suites"]["qmrlab"] = {"n_reports": len(qmrlab_rows)}

    if write_md:
        summary_md = _write_summary_markdown(
            out_dir,
            core_case_rows=core_case_rows,
            decaes_rows=decaes_rows,
            qmrlab_rows=qmrlab_rows,
        )
        written_files.append(summary_md)

    if write_json:
        summary_json = out_dir / "summary.json"
        _write_json(summary_json, summary_payload)
        written_files.append(summary_json)

    for path in written_files:
        print(f"wrote: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
