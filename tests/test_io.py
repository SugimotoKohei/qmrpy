from __future__ import annotations

import builtins
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from qmrpy.io import (
    load_bids_relaxometry,
    load_dicom_series,
    load_nifti,
    save_nifti,
    save_nifti_map,
)


pytest.importorskip("nibabel")
pytest.importorskip("pydicom")


def test_nifti_roundtrip_preserves_affine_and_header(tmp_path: Path) -> None:
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    affine = np.array(
        [
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 2.5, 0.0, -3.0],
            [0.0, 0.0, 4.0, 5.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    path = tmp_path / "map.nii.gz"

    save_nifti(path, data, affine=affine)
    loaded, loaded_affine, header = load_nifti(path)

    assert np.array_equal(loaded, data)
    assert np.allclose(loaded_affine, affine)
    assert tuple(header.get_data_shape()) == data.shape


def test_save_nifti_map_accepts_fit_result_like_mapping(tmp_path: Path) -> None:
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    affine = np.eye(4)
    path = tmp_path / "t2_map.nii"

    save_nifti_map(path, {"params": {"t2_ms": data}}, "t2_ms", affine=affine)
    loaded, loaded_affine, _ = load_nifti(path)

    assert np.array_equal(loaded, data)
    assert np.allclose(loaded_affine, affine)


def test_dicom_series_loads_sorted_4d_echo_stack(tmp_path: Path) -> None:
    for echo_number, echo_time_ms in [(1, 10.0), (2, 20.0)]:
        for slice_index, z_mm in enumerate([0.0, 3.0], start=1):
            pixel = np.full((3, 4), echo_number * 100 + slice_index, dtype=np.uint16)
            _write_synthetic_dicom(
                tmp_path / f"echo-{echo_number}_slice-{slice_index}.dcm",
                pixel,
                echo_number=echo_number,
                echo_time_ms=echo_time_ms,
                z_mm=z_mm,
                instance_number=(echo_number - 1) * 10 + slice_index,
            )

    data, metadata = load_dicom_series(tmp_path)

    assert data.shape == (2, 3, 4, 2)
    assert np.all(data[0, :, :, 0] == 101)
    assert np.all(data[1, :, :, 0] == 102)
    assert np.all(data[0, :, :, 1] == 201)
    assert metadata["echo_time_ms"] == [10.0, 20.0]
    assert metadata["repetition_time_ms"] == 1000.0
    assert metadata["flip_angle_deg"] == 30.0
    assert metadata["inversion_time_ms"] == 450.0
    assert metadata["voxel_spacing_mm"] == (1.5, 2.0, 3.0)


def test_bids_relaxometry_loads_sidecars_and_stacks_last_axis(tmp_path: Path) -> None:
    affine = np.diag([1.2, 1.3, 2.5, 1.0])
    for echo_index, echo_time_s in [(1, 0.01), (2, 0.02)]:
        path = tmp_path / f"sub-01_echo-{echo_index}_T2w.nii.gz"
        save_nifti(path, np.full((2, 3, 4), echo_index, dtype=np.float32), affine=affine)
        sidecar = {
            "EchoTime": echo_time_s,
            "RepetitionTime": 1.5,
            "FlipAngle": 90.0,
            "InversionTime": 0.8,
        }
        path.with_name(path.name.removesuffix(".nii.gz") + ".json").write_text(
            json.dumps(sidecar),
            encoding="utf-8",
        )

    data, metadata = load_bids_relaxometry(tmp_path, suffix="T2w")

    assert data.shape == (2, 3, 4, 2)
    assert np.all(data[..., 0] == 1)
    assert np.all(data[..., 1] == 2)
    assert metadata["echo_time_ms"] == [10.0, 20.0]
    assert metadata["repetition_time_ms"] == [1500.0, 1500.0]
    assert metadata["flip_angle_deg"] == [90.0, 90.0]
    assert metadata["inversion_time_ms"] == [800.0, 800.0]
    assert metadata["bids_entities"][0]["echo"] == "1"
    assert metadata["bids_entities"][0]["suffix"] == "T2w"


def test_nifti_import_error_is_actionable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "nibabel":
            raise ModuleNotFoundError("No module named 'nibabel'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match=r"qmrpy\[io\]"):
        load_nifti(tmp_path / "missing.nii.gz")


def _write_synthetic_dicom(
    path: Path,
    pixel: np.ndarray,
    *,
    echo_number: int,
    echo_time_ms: float,
    z_mm: float,
    instance_number: int,
) -> None:
    pytest.importorskip("pydicom")
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.PatientName = "qmrpy^Synthetic"
    ds.PatientID = "QMRPY"
    ds.Modality = "MR"
    ds.SeriesInstanceUID = generate_uid()
    ds.StudyInstanceUID = generate_uid()
    ds.FrameOfReferenceUID = generate_uid()
    ds.Rows, ds.Columns = pixel.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelSpacing = [1.5, 2.0]
    ds.SliceThickness = 3.0
    ds.ImagePositionPatient = [0.0, 0.0, z_mm]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.InstanceNumber = instance_number
    ds.EchoNumbers = echo_number
    ds.EchoTime = echo_time_ms
    ds.RepetitionTime = 1000.0
    ds.FlipAngle = 30.0
    ds.InversionTime = 450.0
    ds.PixelData = pixel.tobytes()
    ds.save_as(path)
