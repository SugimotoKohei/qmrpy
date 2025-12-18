#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SweepGrid:
    regularization_alpha: list[float]
    t2_basis_n: list[int]
    t2_basis_max_ms: list[float]
    cutoff_ms: list[float]
    noise_model: list[str]
    noise_sigma: list[float]
    seed: int


def _maybe_plot(df: Any, *, out_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mpl_cache"))
    try:
        from plotnine import aes, geom_hline, geom_line, geom_point, ggplot, labs, theme_bw
        from plotnine import ggsave
    except Exception:  # noqa: BLE001
        return

    fig = (
        ggplot(df, aes(x="regularization_alpha", y="abs_dmwf_percent", color="noise_model"))
        + geom_line()
        + geom_point()
        + geom_hline(yintercept=0.5, linetype="dashed", alpha=0.4)
        + theme_bw()
        + labs(
            title="|dMWF| vs regularization_alpha (qMRLab vs qmrpy)",
            x="regularization_alpha",
            y="|dMWF| (%)",
        )
    )
    ggsave(fig, filename=str(out_dir / "plot__abs_dmwf_vs_alpha.png"), verbose=False, dpi=150)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Sweep qMRLab(mwf) vs qmrpy comparison parameters.")
    p.add_argument("--qmrlab-path", type=Path, default=Path("qMRLab"))
    p.add_argument("--out-dir", type=Path, default=Path("output/reports/qmrlab_parity_sweeps"))

    p.add_argument("--mwf-percent", type=float, default=15.0)
    p.add_argument("--t2mw-ms", type=float, default=20.0)
    p.add_argument("--t2iew-ms", type=float, default=80.0)

    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--cutoff-ms", type=str, default="30,40,50")
    p.add_argument("--alpha", type=str, default="0,1e-6,1e-4,1e-3,1e-2")
    p.add_argument("--t2-basis-n", type=str, default="120")
    p.add_argument("--t2-basis-max-ms", type=str, default="300,400")

    p.add_argument("--noise-model", type=str, default="none,gaussian,rician")
    # Note: qMRLab's mwf.equation returns a normalized signal with S(TE=0)~1,
    # so noise sigma should typically be small (e.g., 0.001..0.01) for reasonable SNR.
    p.add_argument("--noise-sigma", type=str, default="0,0.001,0.002,0.005,0.01")
    p.add_argument("--plot", action="store_true", help="Also write a quick diagnostic plot (requires plotnine).")
    args = p.parse_args(argv)

    def _parse_floats(s: str) -> list[float]:
        out: list[float] = []
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            out.append(float(part))
        return out

    def _parse_ints(s: str) -> list[int]:
        out: list[int] = []
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            out.append(int(part))
        return out

    grid = SweepGrid(
        regularization_alpha=_parse_floats(args.alpha),
        t2_basis_n=_parse_ints(args.t2_basis_n),
        t2_basis_max_ms=_parse_floats(args.t2_basis_max_ms),
        cutoff_ms=_parse_floats(args.cutoff_ms),
        noise_model=[x.strip() for x in args.noise_model.split(",") if x.strip()],
        noise_sigma=_parse_floats(args.noise_sigma),
        seed=int(args.seed),
    )

    # Make sibling scripts importable when executed via `uv run scripts/...`.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from verify_qmrlab_mwf import QmrlabMwfCase, run_case  # type: ignore

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = args.out_dir / f"{ts}__mwf_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []

    combos = list(
        itertools.product(
            grid.regularization_alpha,
            grid.t2_basis_n,
            grid.t2_basis_max_ms,
            grid.cutoff_ms,
            grid.noise_model,
            grid.noise_sigma,
        )
    )

    for alpha, n_basis, t2_max, cutoff, noise_model, noise_sigma in combos:
        case = QmrlabMwfCase(
            mwf_percent=float(args.mwf_percent),
            t2mw_ms=float(args.t2mw_ms),
            t2iew_ms=float(args.t2iew_ms),
            cutoff_ms=float(cutoff),
            noise_model=str(noise_model),
            noise_sigma=float(noise_sigma),
            seed=int(grid.seed),
        )

        report_path = run_case(
            qmrlab_path=args.qmrlab_path,
            out_dir=out_dir / "cases",
            case=case,
            regularization_alpha=float(alpha),
            t2_basis_n=int(n_basis),
            t2_basis_max_ms=float(t2_max),
            upper_cutoff_iew_ms=200.0,
        )
        payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
        diff = payload["diff"]
        records.append(
            {
                "report_path": str(report_path),
                "regularization_alpha": float(alpha),
                "t2_basis_n": int(n_basis),
                "t2_basis_max_ms": float(t2_max),
                "cutoff_ms": float(cutoff),
                "noise_model": str(noise_model),
                "noise_sigma": float(noise_sigma),
                "dmwf_percent": float(diff["mwf_percent"]),
                "dt2mw_ms": float(diff["t2mw_ms"]),
                "dt2iew_ms": float(diff["t2iew_ms"]),
            }
        )

    import pandas as pd

    df = pd.DataFrame.from_records(records)
    df["abs_dmwf_percent"] = df["dmwf_percent"].abs()
    df["abs_dt2mw_ms"] = df["dt2mw_ms"].abs()
    df["abs_dt2iew_ms"] = df["dt2iew_ms"].abs()
    df = df.sort_values(["abs_dmwf_percent", "abs_dt2mw_ms"], ascending=True)

    df.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "grid.json").write_text(json.dumps(grid.__dict__, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    best = df.head(10).to_dict(orient="records")
    (out_dir / "top10.json").write_text(json.dumps(best, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if bool(args.plot):
        _maybe_plot(df, out_dir=out_dir)

    print(f"wrote: {out_dir / 'summary.csv'}")
    print(f"wrote: {out_dir / 'top10.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
