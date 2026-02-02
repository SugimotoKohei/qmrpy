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


def _decaes_parity_rows() -> list[dict[str, Any]]:
    from qmrpy.models.t2.decaes_t2 import DECAEST2Map

    rows: list[dict[str, Any]] = []
    signal = _load_csv_1d(TEST_DATA_DIR / "decaes_ref_signal.csv")

    # Case 1: fixed flip angle, no regularization
    t2times = _load_csv_1d(TEST_DATA_DIR / "decaes_ref_t2times.csv") * 1000.0
    echotimes = _load_csv_1d(TEST_DATA_DIR / "decaes_ref_echotimes.csv") * 1000.0
    alpha_ref = float(_load_csv_1d(TEST_DATA_DIR / "decaes_ref_alpha.csv")[0])
    dist_ref = _load_csv_1d(TEST_DATA_DIR / "decaes_ref_dist.csv")

    model = DECAEST2Map(
        n_te=len(echotimes),
        te_ms=float(echotimes[0]),
        n_t2=len(t2times),
        t2_range_ms=(float(t2times[0]), float(t2times[-1])),
        reg="none",
        set_flip_angle_deg=180.0,
        epg_backend="decaes",
    )
    out = model.fit(signal)
    dist_diff = out["distribution"] - dist_ref
    rows.append(
        {
            "case": "decaes_fixed_flip_no_reg",
            "reg": "none",
            "alpha_abs_deg": abs(float(out["alpha_deg"]) - alpha_ref),
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
    model = DECAEST2Map(
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
    dist_diff = out["distribution"] - dist_ref
    rows.append(
        {
            "case": "decaes_opt_alpha_gcv",
            "reg": "gcv",
            "alpha_abs_deg": abs(float(out["alpha_deg"]) - alpha_ref),
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
        model = DECAEST2Map(
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
        dist_diff = out["distribution"] - dist_ref
        rows.append(
            {
                "case": f"decaes_opt_alpha_{reg}",
                "reg": reg,
                "alpha_abs_deg": abs(float(out["alpha_deg"]) - alpha_ref),
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
    rows.sort(key=lambda r: (r.get("mwf_abs_diff_pct", float("inf")), r.get("t2mw_abs_diff_ms", float("inf"))))
    return rows


def _write_summary_markdown(
    out_dir: Path,
    *,
    decaes_rows: list[dict[str, Any]] | None,
    qmrlab_rows: list[dict[str, Any]] | None,
) -> Path:
    lines: list[str] = []
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines.append(f"# Parity Summary")
    lines.append(f"Generated: {ts}")
    lines.append("")

    if decaes_rows is not None:
        lines.append("## DECAES.jl parity (qmrpy vs reference)")
        decaes_cols = [
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
        md_path = out_dir / "decaes_parity.md"
        _write_markdown_table(md_path, decaes_rows, decaes_cols)
        lines.append(f"- Table: {_safe_relpath(md_path)}")
        lines.append("")

    if qmrlab_rows is not None:
        lines.append("## qMRLab MWF parity (existing report.json only)")
        qmrlab_cols = [
            "report_path",
            "mwf_abs_diff_pct",
            "t2mw_abs_diff_ms",
            "t2iew_abs_diff_ms",
            "noise_model",
            "noise_sigma",
            "cutoff_ms",
            "regularization_alpha",
        ]
        md_path = out_dir / "qmrlab_mwf_parity.md"
        _write_markdown_table(md_path, qmrlab_rows, qmrlab_cols)
        lines.append(f"- Table: {_safe_relpath(md_path)}")
        lines.append("")

    summary_path = out_dir / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Summarize parity results into CSV/Markdown tables.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--no-decaes", action="store_true", help="Skip DECAES parity computation.")
    p.add_argument("--no-qmrlab", action="store_true", help="Skip qMRLab report collection.")
    args = p.parse_args(argv)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    decaes_rows: list[dict[str, Any]] | None = None
    if not args.no_decaes:
        decaes_rows = _decaes_parity_rows()
        decaes_cols = [
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
        _write_csv(out_dir / "decaes_parity.csv", decaes_rows, decaes_cols)

    qmrlab_rows: list[dict[str, Any]] | None = None
    if not args.no_qmrlab:
        qmrlab_rows = _collect_qmrlab_mwf_reports()
        qmrlab_cols = [
            "report_path",
            "mwf_abs_diff_pct",
            "t2mw_abs_diff_ms",
            "t2iew_abs_diff_ms",
            "noise_model",
            "noise_sigma",
            "cutoff_ms",
            "regularization_alpha",
        ]
        if qmrlab_rows:
            _write_csv(out_dir / "qmrlab_mwf_parity.csv", qmrlab_rows, qmrlab_cols)
        else:
            qmrlab_rows = None

    summary_path = _write_summary_markdown(out_dir, decaes_rows=decaes_rows, qmrlab_rows=qmrlab_rows)
    print(f"wrote: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
