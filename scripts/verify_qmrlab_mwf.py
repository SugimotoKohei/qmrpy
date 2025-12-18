#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QmrlabMwfCase:
    mwf_percent: float
    t2mw_ms: float
    t2iew_ms: float
    cutoff_ms: float
    noise_model: str = "none"
    noise_sigma: float = 0.0
    seed: int = 0


def _mat_struct_to_dict(obj: Any) -> Any:
    import numpy as np

    # scipy.io.loadmat with struct_as_record=False yields mat_struct objects
    if hasattr(obj, "_fieldnames"):
        return {name: _mat_struct_to_dict(getattr(obj, name)) for name in obj._fieldnames}

    # scipy.io.loadmat loads MATLAB structs as numpy.void with dtype.names
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.size == 1:
        return _mat_struct_to_dict(obj.item())

    if isinstance(obj, np.void) and obj.dtype.names:
        out: dict[str, Any] = {}
        for name in obj.dtype.names:
            out[name] = _mat_struct_to_dict(obj[name])
        return out

    if isinstance(obj, np.ndarray):
        if obj.size == 1:
            return _mat_struct_to_dict(obj.reshape(-1)[0])
        return obj

    return obj


def _run_octave_generate_and_fit(
    *,
    qmrlab_path: Path,
    out_mat: Path,
    case: QmrlabMwfCase,
    script_path: Path,
    qmrlab_sigma: float | None = None,
) -> None:
    sigma_value = case.noise_sigma if qmrlab_sigma is None else float(qmrlab_sigma)
    cmd = [
        "octave",
        "--no-gui",
        "--quiet",
        "--eval",
        (
            "warning('off','all'); "
            f"qMRLab_path='{qmrlab_path.resolve()}'; "
            f"out_mat='{out_mat.resolve()}'; "
            f"MWF_percent={case.mwf_percent}; "
            f"T2MW_ms={case.t2mw_ms}; "
            f"T2IEW_ms={case.t2iew_ms}; "
            f"Cutoff_ms={case.cutoff_ms}; "
            f"NoiseModel='{case.noise_model}'; "
            f"NoiseSigma={case.noise_sigma}; "
            f"Seed={case.seed}; "
            f"QmrlabSigma={sigma_value}; "
            f"source('{script_path.resolve()}');"
        ),
    ]
    subprocess.check_call(cmd)


def run_case(
    *,
    qmrlab_path: Path,
    out_dir: Path,
    case: QmrlabMwfCase,
    regularization_alpha: float,
    t2_basis_n: int = 120,
    t2_basis_max_ms: float = 400.0,
    upper_cutoff_iew_ms: float = 200.0,
    qmrlab_sigma: float | None = None,
) -> Path:
    """Run one qMRLab-vs-qmrpy MWF comparison and write report.json; returns report path."""
    if not qmrlab_path.exists():
        raise FileNotFoundError(f"qMRLab not found at: {qmrlab_path}")

    if subprocess.call(["bash", "-lc", "command -v octave >/dev/null 2>&1"]) != 0:
        raise RuntimeError("octave が見つかりません。Octave をインストールしてから再実行してください。")

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = out_dir / f"{ts}__mwf_fixed_vector"
    out_dir.mkdir(parents=True, exist_ok=True)

    octave_script = Path("scripts/octave/mwf_generate_and_fit.m")
    out_mat = out_dir / "qmrlab_out.mat"
    _run_octave_generate_and_fit(
        qmrlab_path=qmrlab_path,
        out_mat=out_mat,
        case=case,
        script_path=octave_script,
        qmrlab_sigma=qmrlab_sigma,
    )

    import numpy as np
    import scipy.io

    from qmrpy.models.t2 import MultiComponentT2

    mat = scipy.io.loadmat(out_mat, squeeze_me=True, struct_as_record=False)
    echo_times_ms = np.asarray(mat["EchoTimes_ms"], dtype=float).reshape(-1)
    signal = np.asarray(mat["Signal"], dtype=float).reshape(-1)
    fit_results = _mat_struct_to_dict(mat["FitResults"])

    lower_basis = float(1.5 * echo_times_ms[0])
    basis = MultiComponentT2.default_t2_basis_ms(t2_min_ms=lower_basis, t2_max_ms=float(t2_basis_max_ms), n=int(t2_basis_n))
    model = MultiComponentT2(te_ms=echo_times_ms, t2_basis_ms=basis)
    py = model.fit(
        signal,
        regularization_alpha=float(regularization_alpha),
        lower_cutoff_mw_ms=lower_basis,
        cutoff_ms=float(case.cutoff_ms),
        upper_cutoff_iew_ms=float(upper_cutoff_iew_ms),
        use_weighted_geometric_mean=True,
    )

    qmrlab_mwf_percent = float(np.asarray(fit_results.get("MWF")).reshape(-1)[0])
    qmrlab_t2mw_ms = float(np.asarray(fit_results.get("T2MW")).reshape(-1)[0])
    qmrlab_t2iew_ms = float(np.asarray(fit_results.get("T2IEW")).reshape(-1)[0])

    report = {
        "case": {
            "mwf_percent": case.mwf_percent,
            "t2mw_ms": case.t2mw_ms,
            "t2iew_ms": case.t2iew_ms,
            "cutoff_ms": case.cutoff_ms,
            "noise_model": str(case.noise_model),
            "noise_sigma": float(case.noise_sigma),
            "seed": int(case.seed),
        },
        "protocol": {"echo_times_ms": echo_times_ms.tolist()},
        "qmrlab": {"mwf_percent": qmrlab_mwf_percent, "t2mw_ms": qmrlab_t2mw_ms, "t2iew_ms": qmrlab_t2iew_ms},
        "qmrpy": {
            "mwf_percent": float(100.0 * py["mwf"]),
            "t2mw_ms": float(py["t2mw_ms"]),
            "t2iew_ms": float(py["t2iew_ms"]),
            "gmt2_ms": float(py["gmt2_ms"]),
            "resid_l2": float(py["resid_l2"]),
            "regularization_alpha": float(regularization_alpha),
            "basis": {"t2_min_ms": lower_basis, "t2_max_ms": float(t2_basis_max_ms), "n": int(t2_basis_n)},
            "upper_cutoff_iew_ms": float(upper_cutoff_iew_ms),
        },
        "diff": {
            "mwf_percent": float(100.0 * py["mwf"] - qmrlab_mwf_percent),
            "t2mw_ms": float(py["t2mw_ms"] - qmrlab_t2mw_ms),
            "t2iew_ms": float(py["t2iew_ms"] - qmrlab_t2iew_ms),
        },
        "artifacts": {"qmrlab_out_mat": str(out_mat)},
    }

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="qMRLab(mwf) vs qmrpy(MWF) fixed-vector comparison (Octave required).")
    p.add_argument("--qmrlab-path", type=Path, default=Path("qMRLab"), help="Path to qMRLab checkout (default: ./qMRLab)")
    p.add_argument("--mwf-percent", type=float, default=15.0, help="Ground-truth MWF in percent (qMRLab convention).")
    p.add_argument("--t2mw-ms", type=float, default=20.0, help="Ground-truth T2MW [ms].")
    p.add_argument("--t2iew-ms", type=float, default=80.0, help="Ground-truth T2IEW [ms].")
    p.add_argument("--cutoff-ms", type=float, default=40.0, help="Cutoff [ms].")
    p.add_argument("--noise-model", type=str, default="none", help="none|gaussian|rician (default: none)")
    p.add_argument("--noise-sigma", type=float, default=0.0, help="Noise sigma (default: 0.0)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed used in Octave generation (default: 0)")
    p.add_argument(
        "--qmrlab-sigma",
        type=float,
        default=None,
        help="Sigma used in qMRLab fitting (default: same as noise-sigma)",
    )
    p.add_argument(
        "--regularization-alpha",
        type=float,
        default=0.0,
        help="qmrpy NNLS Tikhonov regularization alpha (default: 0.0).",
    )
    p.add_argument("--t2-basis-n", type=int, default=120, help="qMRLab-like basis points (default: 120)")
    p.add_argument("--t2-basis-max-ms", type=float, default=400.0, help="qMRLab-like basis max [ms] (default: 400)")
    p.add_argument("--upper-cutoff-iew-ms", type=float, default=200.0, help="Upper cutoff for IEW [ms] (default: 200)")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/reports/qmrlab_parity"),
        help="Output directory under output/ (default: output/reports/qmrlab_parity).",
    )
    args = p.parse_args(argv)

    case = QmrlabMwfCase(
        mwf_percent=float(args.mwf_percent),
        t2mw_ms=float(args.t2mw_ms),
        t2iew_ms=float(args.t2iew_ms),
        cutoff_ms=float(args.cutoff_ms),
        noise_model=str(args.noise_model),
        noise_sigma=float(args.noise_sigma),
        seed=int(args.seed),
    )

    try:
        report_path = run_case(
            qmrlab_path=args.qmrlab_path,
            out_dir=args.out_dir,
            case=case,
            regularization_alpha=float(args.regularization_alpha),
            t2_basis_n=int(args.t2_basis_n),
            t2_basis_max_ms=float(args.t2_basis_max_ms),
            upper_cutoff_iew_ms=float(args.upper_cutoff_iew_ms),
            qmrlab_sigma=(None if args.qmrlab_sigma is None else float(args.qmrlab_sigma)),
        )
    except Exception as e:  # noqa: BLE001
        print(str(e), file=sys.stderr)
        return 2

    print("=== qMRLab vs qmrpy (MWF fixed vector) ===")
    print(f"output: {report_path}")
    payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
    qm = payload["qmrlab"]
    py = payload["qmrpy"]
    diff = payload["diff"]
    print(f"qMRLab: MWF={qm['mwf_percent']:.4f}% T2MW={qm['t2mw_ms']:.2f}ms T2IEW={qm['t2iew_ms']:.2f}ms")
    print(f"qmrpy : MWF={py['mwf_percent']:.4f}% T2MW={py['t2mw_ms']:.2f}ms T2IEW={py['t2iew_ms']:.2f}ms")
    print(f"diff  : dMWF={diff['mwf_percent']:+.4f}% dT2MW={diff['t2mw_ms']:+.2f}ms dT2IEW={diff['t2iew_ms']:+.2f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
