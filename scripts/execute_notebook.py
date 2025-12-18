from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    ap = argparse.ArgumentParser(description="Execute a Jupyter notebook (.ipynb) in-place or to an output path")
    ap.add_argument("notebook", type=Path)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--kernel", type=str, default="python3")
    args = ap.parse_args()

    nb_path: Path = args.notebook
    out_path: Path = args.output if args.output is not None else nb_path

    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(nb, timeout=int(args.timeout), kernel_name=str(args.kernel))
    client.execute()
    nbformat.write(nb, out_path)


if __name__ == "__main__":
    main()
