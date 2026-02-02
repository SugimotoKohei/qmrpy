#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).parent.parent
DEFAULT_OUT_DIR = ROOT_DIR / "output" / "reports" / "test_matrix"


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
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rows() -> list[dict[str, Any]]:
    return [
        {
            "category": "T1",
            "target": "VFA T1",
            "verification": "forward/fit/shape",
            "tests": "tests/test_vfa_t1.py",
        },
        {
            "category": "T1",
            "target": "Inversion Recovery",
            "verification": "forward/fit",
            "tests": "tests/test_inversion_recovery.py",
        },
        {
            "category": "T2",
            "target": "MonoT2",
            "verification": "forward/fit/linear",
            "tests": "tests/test_mono_t2.py; tests/test_import.py",
        },
        {
            "category": "T2",
            "target": "DECAES T2 map",
            "verification": "numeric/parity",
            "tests": "tests/test_decaes_t2.py",
        },
        {
            "category": "T2",
            "target": "DECAES parity",
            "verification": "reference data match",
            "tests": "tests/test_decaes_parity.py",
        },
        {
            "category": "T2",
            "target": "DECAES T2 part",
            "verification": "summary metrics",
            "tests": "tests/test_decaes_t2part.py",
        },
        {
            "category": "T2",
            "target": "MWF (MultiComponentT2)",
            "verification": "keys/consistency",
            "tests": "tests/test_mwf.py",
        },
        {
            "category": "B1",
            "target": "AFI",
            "verification": "forward/fit",
            "tests": "tests/test_b1_afi.py",
        },
        {
            "category": "B1",
            "target": "DAM",
            "verification": "forward/fit",
            "tests": "tests/test_b1_dam.py",
        },
        {
            "category": "Noise",
            "target": "MPPCA",
            "verification": "shape/errors",
            "tests": "tests/test_mppca.py",
        },
        {
            "category": "QSM",
            "target": "Split Bregman",
            "verification": "pipeline/numeric",
            "tests": "tests/test_qsm_split_bregman.py",
        },
        {
            "category": "QSM",
            "target": "Pipeline",
            "verification": "integration",
            "tests": "tests/test_qsm_pipeline.py",
        },
        {
            "category": "Sim",
            "target": "Phantoms",
            "verification": "generation",
            "tests": "tests/test_phantoms.py",
        },
        {
            "category": "Sim",
            "target": "Simulation",
            "verification": "consistency",
            "tests": "tests/test_simulation.py",
        },
        {
            "category": "Sim",
            "target": "Sequence templates",
            "verification": "generation",
            "tests": "tests/test_sim_templates.py",
        },
        {
            "category": "Regression",
            "target": "Fixed vectors",
            "verification": "reproducibility",
            "tests": "tests/test_fixed_vectors.py",
        },
        {
            "category": "Smoke",
            "target": "Import",
            "verification": "import",
            "tests": "tests/test_import.py",
        },
    ]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Summarize test coverage into Markdown/CSV tables.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = p.parse_args(argv)

    rows = _rows()
    columns = ["category", "target", "verification", "tests"]
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(out_dir / "test_matrix.csv", rows, columns)
    _write_markdown_table(out_dir / "test_matrix.md", rows, columns)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    summary = out_dir / "summary.md"
    summary.write_text(
        "\n".join(
            [
                "# Test Matrix",
                f"Generated: {ts}",
                "",
                f"- Table: {str((out_dir / 'test_matrix.md').relative_to(ROOT_DIR))}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
