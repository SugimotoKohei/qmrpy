from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from qmrpy.models import B0MultiEcho, B1BlochSiegert, T2EPG, T2StarComplexR2, T1VFA


def _fit_complex_df_fixed0(signal: np.ndarray, te_ms: np.ndarray) -> float:
    """Fit complex mono-exponential decay with delta_f fixed to 0 Hz."""
    y = np.asarray(signal, dtype=np.complex128)
    y_abs = np.abs(y)
    x0 = np.array([max(float(y_abs[0]), 1e-6), 30.0, float(np.angle(y[0]))], dtype=np.float64)

    def residuals(params: np.ndarray) -> np.ndarray:
        s0 = float(params[0])
        t2star_ms = float(params[1])
        phi0_rad = float(params[2])
        pred = s0 * np.exp(-te_ms / t2star_ms) * np.exp(1j * phi0_rad)
        return np.concatenate([(pred.real - y.real), (pred.imag - y.imag)])

    lower = np.array([0.0, 1e-6, -np.inf], dtype=np.float64)
    upper = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    out = least_squares(residuals, x0=x0, bounds=(lower, upper), max_nfev=300)
    return float(out.x[1])


def run_benchmark(*, seed: int = 20260211, n_samples: int = 300) -> dict[str, Any]:
    rng = np.random.default_rng(seed)

    flip_angle_deg = np.array([4.0, 12.0, 25.0], dtype=np.float64)
    te_ms_t2star = np.array([4.0, 8.0, 12.0, 16.0, 20.0, 24.0], dtype=np.float64)
    te_s_t2star = te_ms_t2star / 1000.0

    b1_model = B1BlochSiegert(k_bs_rad_per_b1sq=1.0)
    b0_model = B0MultiEcho(te_ms=te_ms_t2star)
    r2star_model = T2StarComplexR2(te_ms=te_ms_t2star)

    t1_err_unc: list[float] = []
    t1_err_cor: list[float] = []
    t2_err_unc: list[float] = []
    t2_err_cor: list[float] = []
    t2star_err_unc: list[float] = []
    t2star_err_cor: list[float] = []
    b1_abs_err: list[float] = []
    b0_sq_err: list[float] = []

    for _ in range(n_samples):
        m0 = float(rng.uniform(800.0, 1400.0))
        b1_true = float(rng.uniform(0.8, 1.2))
        t1_true_ms = float(rng.uniform(700.0, 1900.0))
        t2_true_ms = float(rng.uniform(30.0, 110.0))
        t2star_true_ms = float(rng.uniform(12.0, 70.0))
        b0_true_hz = float(rng.uniform(-120.0, 120.0))
        phase0_true_rad = float(rng.uniform(-np.pi, np.pi))

        bs_signal = b1_model.forward(b1=b1_true, phase0_rad=phase0_true_rad)
        bs_signal_noisy = bs_signal + rng.normal(0.0, 0.008, size=2)
        b1_hat = float(b1_model.fit(bs_signal_noisy)["params"]["b1_raw"])
        b1_abs_err.append(abs(b1_hat - b1_true))

        t1_signal = T1VFA(flip_angle_deg=flip_angle_deg, tr_ms=18.0, b1=b1_true).forward(m0=m0, t1_ms=t1_true_ms)
        t1_signal_noisy = t1_signal + rng.normal(0.0, 2.0, size=t1_signal.shape)
        t1_out_unc = T1VFA(flip_angle_deg=flip_angle_deg, tr_ms=18.0, b1=1.0).fit(t1_signal_noisy)
        t1_out_cor = T1VFA(flip_angle_deg=flip_angle_deg, tr_ms=18.0, b1=b1_hat).fit(t1_signal_noisy)
        t1_err_unc.append(abs(float(t1_out_unc["params"]["t1_ms"]) - t1_true_ms) / t1_true_ms)
        t1_err_cor.append(abs(float(t1_out_cor["params"]["t1_ms"]) - t1_true_ms) / t1_true_ms)

        t2_signal = T2EPG(
            n_te=16,
            te_ms=10.0,
            t1_ms=1000.0,
            alpha_deg=180.0,
            beta_deg=180.0,
            b1=b1_true,
        ).forward(m0=m0, t2_ms=t2_true_ms)
        t2_signal_noisy = t2_signal + rng.normal(0.0, 2.0, size=t2_signal.shape)
        t2_out_unc = T2EPG(
            n_te=16,
            te_ms=10.0,
            t1_ms=1000.0,
            alpha_deg=180.0,
            beta_deg=180.0,
            b1=1.0,
        ).fit(t2_signal_noisy)
        t2_out_cor = T2EPG(
            n_te=16,
            te_ms=10.0,
            t1_ms=1000.0,
            alpha_deg=180.0,
            beta_deg=180.0,
            b1=b1_hat,
        ).fit(t2_signal_noisy)
        t2_err_unc.append(abs(float(t2_out_unc["params"]["t2_ms"]) - t2_true_ms) / t2_true_ms)
        t2_err_cor.append(abs(float(t2_out_cor["params"]["t2_ms"]) - t2_true_ms) / t2_true_ms)

        b0_phase = np.angle(np.exp(1j * (phase0_true_rad + 2.0 * np.pi * b0_true_hz * te_s_t2star)))
        b0_phase_noisy = b0_phase + rng.normal(0.0, 0.03, size=te_ms_t2star.shape)
        b0_hat = float(b0_model.fit(b0_phase_noisy)["params"]["b0_hz"])
        b0_sq_err.append((b0_hat - b0_true_hz) ** 2)

        t2star_signal = r2star_model.forward(
            s0=m0,
            t2star_ms=t2star_true_ms,
            delta_f_hz=b0_true_hz,
            phi0_rad=phase0_true_rad,
        )
        t2star_signal_noisy = t2star_signal + (
            rng.normal(0.0, 8.0, size=t2star_signal.shape)
            + 1j * rng.normal(0.0, 8.0, size=t2star_signal.shape)
        )

        t2star_unc_ms = _fit_complex_df_fixed0(t2star_signal_noisy, te_ms_t2star)
        t2star_signal_demod = t2star_signal_noisy * np.exp(-1j * 2.0 * np.pi * b0_hat * te_s_t2star)
        t2star_cor_ms = _fit_complex_df_fixed0(t2star_signal_demod, te_ms_t2star)
        t2star_err_unc.append(abs(t2star_unc_ms - t2star_true_ms) / t2star_true_ms)
        t2star_err_cor.append(abs(t2star_cor_ms - t2star_true_ms) / t2star_true_ms)

    arr_t1_unc = np.asarray(t1_err_unc, dtype=np.float64)
    arr_t1_cor = np.asarray(t1_err_cor, dtype=np.float64)
    arr_t2_unc = np.asarray(t2_err_unc, dtype=np.float64)
    arr_t2_cor = np.asarray(t2_err_cor, dtype=np.float64)
    arr_t2star_unc = np.asarray(t2star_err_unc, dtype=np.float64)
    arr_t2star_cor = np.asarray(t2star_err_cor, dtype=np.float64)
    arr_b1_abs = np.asarray(b1_abs_err, dtype=np.float64)
    arr_b0_sq = np.asarray(b0_sq_err, dtype=np.float64)

    metrics: dict[str, float] = {
        "t1_rel_median_uncorrected": float(np.median(arr_t1_unc)),
        "t1_rel_median_corrected": float(np.median(arr_t1_cor)),
        "t2_rel_median_uncorrected": float(np.median(arr_t2_unc)),
        "t2_rel_median_corrected": float(np.median(arr_t2_cor)),
        "t2star_rel_median_uncorrected": float(np.median(arr_t2star_unc)),
        "t2star_rel_median_corrected": float(np.median(arr_t2star_cor)),
        "b1_mae": float(np.mean(arr_b1_abs)),
        "b0_rmse_hz": float(np.sqrt(np.mean(arr_b0_sq))),
        "t1_improved_fraction": float(np.mean(arr_t1_cor < arr_t1_unc)),
        "t2_improved_fraction": float(np.mean(arr_t2_cor < arr_t2_unc)),
        "t2star_improved_fraction": float(np.mean(arr_t2star_cor < arr_t2star_unc)),
    }

    thresholds: dict[str, float] = {
        "t1_rel_median_corrected_max": 0.05,
        "t2_rel_median_corrected_max": 0.07,
        "t2star_rel_median_corrected_max": 0.08,
        "b1_mae_max": 0.05,
        "b0_rmse_hz_max": 2.0,
    }

    passes = {
        "t1_rel_median_corrected": metrics["t1_rel_median_corrected"] <= thresholds["t1_rel_median_corrected_max"],
        "t2_rel_median_corrected": metrics["t2_rel_median_corrected"] <= thresholds["t2_rel_median_corrected_max"],
        "t2star_rel_median_corrected": (
            metrics["t2star_rel_median_corrected"] <= thresholds["t2star_rel_median_corrected_max"]
        ),
        "b1_mae": metrics["b1_mae"] <= thresholds["b1_mae_max"],
        "b0_rmse_hz": metrics["b0_rmse_hz"] <= thresholds["b0_rmse_hz_max"],
    }

    return {
        "seed": int(seed),
        "n_samples": int(n_samples),
        "protocol": {
            "flip_angle_deg": flip_angle_deg.tolist(),
            "te_ms_t2star": te_ms_t2star.tolist(),
            "noise": {
                "bloch_siegert_phase_std_rad": 0.008,
                "vfa_signal_std": 2.0,
                "epg_signal_std": 2.0,
                "b0_phase_std_rad": 0.03,
                "t2star_complex_std": 8.0,
            },
        },
        "metrics": metrics,
        "thresholds": thresholds,
        "passes": passes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="B1/B0 補正なし・ありでの T1/T2/T2* 合成比較レポートを生成する"
    )
    parser.add_argument("--seed", type=int, default=20260211)
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("output/reports/b0_b1_correction_report.json"),
        help="JSON出力パス",
    )
    args = parser.parse_args()

    report = run_benchmark(seed=args.seed, n_samples=args.n_samples)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
