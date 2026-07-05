from __future__ import annotations

from pathlib import Path


def test_cli_help_and_info(capsys):
    import pytest

    from qmrpy.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0

    rc = main(["info"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "qmrpy" in captured.out
    assert "t2-mono" in captured.out


def test_cli_fit_t2_mono_nifti_roundtrip(tmp_path: Path):
    import pytest

    np = pytest.importorskip("numpy")
    pytest.importorskip("nibabel")

    from qmrpy.cli import main
    from qmrpy.io import load_nifti, save_nifti
    from qmrpy.models import T2Mono

    te_ms = np.array([10.0, 20.0, 40.0, 80.0], dtype=float)
    model = T2Mono(te_ms=te_ms)
    signal = model.forward(m0=1000.0, t2_ms=80.0)
    image = np.stack([signal, signal], axis=0).reshape(2, 1, 1, -1)
    affine = np.diag([2.0, 2.0, 3.0, 1.0])
    input_path = tmp_path / "input.nii.gz"
    output_path = tmp_path / "t2map.nii.gz"
    save_nifti(input_path, image.astype(np.float32), affine=affine)

    rc = main(
        [
            "fit",
            "t2-mono",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--te-ms",
            "10,20,40,80",
        ]
    )

    out, out_affine, _ = load_nifti(output_path)
    assert rc == 0
    assert out.shape == image.shape[:-1]
    assert np.allclose(out, 80.0, rtol=1e-4)
    assert np.allclose(out_affine, affine)


def test_cli_validate_core_smoke(tmp_path: Path):
    from qmrpy.cli import main

    rc = main(["validate", "--suite", "core", "--out-dir", str(tmp_path / "validation")])

    assert rc == 0
    assert (tmp_path / "validation" / "core_validation.csv").exists()
