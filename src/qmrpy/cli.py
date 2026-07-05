from __future__ import annotations

import argparse
from pathlib import Path


FIT_MODELS = {
    "t2-mono": {
        "param": "t2_ms",
        "requires": ("te_ms",),
    },
    "t1rho": {
        "param": "t1rho_ms",
        "requires": ("tsl_ms",),
    },
    "mtr": {
        "param": "mtr",
        "requires": (),
    },
}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qmrpy", description="qmrpy command line tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    info = subparsers.add_parser("info", help="Show qmrpy version and available CLI models")
    info.set_defaults(func=_cmd_info)

    fit = subparsers.add_parser("fit", help="Fit a supported model to a NIfTI image")
    fit.add_argument("model", choices=sorted(FIT_MODELS))
    fit.add_argument("--input", required=True, type=Path, help="Input NIfTI path")
    fit.add_argument(
        "--output", required=True, type=Path, help="Output NIfTI path for the selected map"
    )
    fit.add_argument("--param", default=None, help="Parameter map to save")
    fit.add_argument("--mask", default=None, help='Mask mode; use "otsu" for automatic masking')
    fit.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs; -1 uses all CPUs")
    fit.add_argument("--verbose", action="store_true", help="Show model progress")
    fit.add_argument("--te-ms", default=None, help="Comma-separated echo times for t2-mono")
    fit.add_argument("--tsl-ms", default=None, help="Comma-separated spin-lock times for t1rho")
    fit.set_defaults(func=_cmd_fit)

    validate = subparsers.add_parser("validate", help="Run the validation suite")
    validate.add_argument("--suite", default="core", choices=["core", "decaes", "qmrlab", "all"])
    validate.add_argument("--config", type=Path, default=None)
    validate.add_argument("--out-dir", type=Path, default=None)
    validate.set_defaults(func=_cmd_validate)
    return parser


def _cmd_info(_: argparse.Namespace) -> int:
    from qmrpy import __version__

    print(f"qmrpy {__version__}")
    print("CLI models:")
    for name, spec in sorted(FIT_MODELS.items()):
        print(f"  {name} -> default map: {spec['param']}")
    return 0


def _cmd_fit(args: argparse.Namespace) -> int:
    from qmrpy.io import load_nifti, save_nifti_map
    from qmrpy.models import MTR, T1Rho, T2Mono

    data, affine, header = load_nifti(args.input)
    model_name = str(args.model)
    if model_name == "t2-mono":
        model = T2Mono(te_ms=_parse_float_list(args.te_ms, name="--te-ms"))
        result = model.fit_image(data, mask=args.mask, n_jobs=args.n_jobs, verbose=args.verbose)
    elif model_name == "t1rho":
        model = T1Rho(tsl_ms=_parse_float_list(args.tsl_ms, name="--tsl-ms"))
        result = model.fit_image(data, mask=args.mask, n_jobs=args.n_jobs, verbose=args.verbose)
    elif model_name == "mtr":
        model = MTR()
        result = model.fit_image(data, mask=args.mask, n_jobs=args.n_jobs, verbose=args.verbose)
    else:  # pragma: no cover - argparse choices prevent this
        raise ValueError(f"unsupported model: {model_name}")

    param = args.param or FIT_MODELS[model_name]["param"]
    save_nifti_map(args.output, result, param, affine=affine, header=header, dtype="float32")
    print(f"wrote: {args.output}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    from scripts import summarize_parity

    cli_args = ["--suite", str(args.suite)]
    if args.config is not None:
        cli_args.extend(["--config", str(args.config)])
    if args.out_dir is not None:
        cli_args.extend(["--out-dir", str(args.out_dir)])
    return int(summarize_parity.main(cli_args))


def _parse_float_list(value: str | None, *, name: str) -> list[float]:
    if value is None:
        raise SystemExit(f"{name} is required")
    try:
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise SystemExit(f"{name} must be a comma-separated list of numbers") from exc
    if not values:
        raise SystemExit(f"{name} must not be empty")
    return values


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
