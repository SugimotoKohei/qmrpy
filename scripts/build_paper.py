#!/usr/bin/env python3
"""Build a submission PDF from a Markdown manuscript via pandoc.

Default source is ``paper_biorxiv.md`` (the bioRxiv preprint manuscript);
``paper.md`` (the JOSS-style paper) can be built with ``--source paper.md``.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = ROOT_DIR / "paper_biorxiv.md"
DEFAULT_OUT_DIR = ROOT_DIR / "output" / "paper_build"
PDF_ENGINES = ("xelatex", "pdflatex", "lualatex")


def _resolve_engine(requested: str | None) -> str:
    candidates = (requested,) if requested else PDF_ENGINES
    for engine in candidates:
        if engine and shutil.which(engine):
            return engine
    raise SystemExit("No LaTeX PDF engine found. Install one of: " + ", ".join(PDF_ENGINES))


def build(source: Path, out_pdf: Path, *, bibliography: Path, engine: str) -> None:
    pandoc = shutil.which("pandoc")
    if pandoc is None:
        raise SystemExit("pandoc not found. Install pandoc to build the manuscript PDF.")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        pandoc,
        str(source),
        "-o",
        str(out_pdf),
        "--from=markdown",
        "--citeproc",
        f"--bibliography={bibliography}",
        f"--pdf-engine={engine}",
        "--resource-path",
        str(ROOT_DIR),
        "-V",
        "linkcolor=blue",
    ]
    result = subprocess.run(cmd, cwd=ROOT_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(f"pandoc failed with exit code {result.returncode}")
    if result.stderr.strip():
        sys.stderr.write(result.stderr)
    print(f"wrote: {out_pdf.relative_to(ROOT_DIR)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--bibliography", type=Path, default=ROOT_DIR / "paper.bib")
    parser.add_argument("--pdf-engine", default=None, choices=[*PDF_ENGINES, None])
    args = parser.parse_args()

    source = args.source if args.source.is_absolute() else ROOT_DIR / args.source
    if not source.exists():
        raise SystemExit(f"manuscript not found: {source}")

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT_DIR / args.out_dir
    out_pdf = out_dir / f"{source.stem}.pdf"
    build(
        source,
        out_pdf,
        bibliography=args.bibliography,
        engine=_resolve_engine(args.pdf_engine),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
