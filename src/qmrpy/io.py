"""I/O utilities for qmrpy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import ArrayLike, NDArray
else:
    ArrayLike = Any
    NDArray = Any

__all__ = [
    "load_bids_relaxometry",
    "load_dicom_series",
    "load_nifti",
    "load_tiff",
    "save_nifti",
    "save_nifti_map",
    "save_tiff",
]


def _import_nibabel() -> Any:
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(
            "NIfTI/BIDS I/O requires the optional dependency 'nibabel'. "
            "Install it with `pip install 'qmrpy[io]'` or `uv add 'qmrpy[io]'`."
        ) from exc
    return nib


def _import_pydicom() -> Any:
    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(
            "DICOM I/O requires the optional dependency 'pydicom'. "
            "Install it with `pip install 'qmrpy[io]'` or `uv add 'qmrpy[io]'`."
        ) from exc
    return pydicom


def save_tiff(
    path: str | Path,
    data: ArrayLike,
    *,
    dtype: str | None = None,
) -> None:
    """Save array as uncompressed TIFF.

    Parameters
    ----------
    path : str or Path
        Output file path.
    data : array-like
        Image data to save. Can be 2D (grayscale), 3D (multi-page), or 4D.
    dtype : str, optional
        Output dtype (e.g., 'float32', 'uint16'). If None, uses input dtype.

    Examples
    --------
    >>> import numpy as np
    >>> from qmrpy.io import save_tiff
    >>> t2_map = np.random.rand(256, 256).astype(np.float32)
    >>> save_tiff("t2_map.tiff", t2_map)
    """
    import numpy as np
    from PIL import Image

    arr = np.asarray(data)
    if dtype is not None:
        arr = arr.astype(dtype)

    if arr.ndim == 2:
        # Single 2D image
        img = Image.fromarray(arr)
        img.save(path, compression=None)
    elif arr.ndim == 3:
        # Multi-page TIFF (stack of 2D images)
        images = [Image.fromarray(arr[i]) for i in range(arr.shape[0])]
        images[0].save(
            path,
            save_all=True,
            append_images=images[1:],
            compression=None,
        )
    elif arr.ndim == 4:
        # 4D: flatten first two dims into pages
        n_pages = arr.shape[0] * arr.shape[1]
        flat = arr.reshape((n_pages,) + arr.shape[2:])
        images = [Image.fromarray(flat[i]) for i in range(n_pages)]
        images[0].save(
            path,
            save_all=True,
            append_images=images[1:],
            compression=None,
        )
    else:
        raise ValueError(f"data must be 2D, 3D, or 4D, got ndim={arr.ndim}")


def load_tiff(path: str | Path) -> NDArray[np.floating[Any]]:
    """Load TIFF as numpy array.

    Parameters
    ----------
    path : str or Path
        Input file path.

    Returns
    -------
    NDArray
        Image data. Multi-page TIFFs are returned as 3D array.
    """
    import numpy as np
    from PIL import Image

    img = Image.open(path)

    # Check for multi-page
    frames = []
    try:
        while True:
            frames.append(np.asarray(img))
            img.seek(img.tell() + 1)
    except EOFError:
        pass

    if len(frames) == 1:
        return frames[0]
    return np.stack(frames, axis=0)


def load_nifti(path: str | Path) -> tuple[NDArray[Any], NDArray[Any], Any]:
    """Load a NIfTI image while preserving spatial metadata.

    Parameters
    ----------
    path : str or Path
        Input ``.nii`` or ``.nii.gz`` file path.

    Returns
    -------
    data : NDArray
        Image array. Nibabel scaling is applied by the array proxy.
    affine : NDArray
        4x4 voxel-to-world affine matrix.
    header : nibabel.Nifti1Header
        Copy of the NIfTI header.
    """
    import numpy as np

    nib = _import_nibabel()
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    return data, np.asarray(image.affine, dtype=np.float64), image.header.copy()


def save_nifti(
    path: str | Path,
    data: ArrayLike,
    *,
    affine: ArrayLike | None = None,
    header: Any | None = None,
    dtype: str | None = None,
) -> None:
    """Save an array as NIfTI.

    Parameters
    ----------
    path : str or Path
        Output ``.nii`` or ``.nii.gz`` file path.
    data : array-like
        Image data to save.
    affine : array-like, optional
        4x4 voxel-to-world affine matrix. If omitted, an identity affine is used.
    header : nibabel.Nifti1Header, optional
        Header to copy into the output image.
    dtype : str, optional
        Output dtype. If omitted, the input dtype is preserved.
    """
    import numpy as np

    nib = _import_nibabel()
    arr = np.asarray(data)
    if dtype is not None:
        arr = arr.astype(dtype)
    out_affine = np.eye(4, dtype=np.float64) if affine is None else np.asarray(affine, dtype=np.float64)
    out_header = header.copy() if hasattr(header, "copy") else header
    image = nib.Nifti1Image(arr, out_affine, header=out_header)
    nib.save(image, str(path))


def save_nifti_map(
    path: str | Path,
    result: Any,
    param_name: str,
    *,
    affine: ArrayLike | None = None,
    header: Any | None = None,
    dtype: str | None = None,
) -> None:
    """Save one parameter map from a qmrpy fit result as NIfTI.

    Parameters
    ----------
    path : str or Path
        Output NIfTI path.
    result : Mapping
        ``FitResult`` or dictionary-like object containing parameter maps.
    param_name : str
        Parameter map key, for example ``"t2_ms"``.
    affine : array-like, optional
        Spatial affine inherited from the source image.
    header : nibabel.Nifti1Header, optional
        Header inherited from the source image.
    dtype : str, optional
        Output dtype.
    """
    if param_name in result:
        data = result[param_name]
    elif "params" in result and param_name in result["params"]:
        data = result["params"][param_name]
    else:
        raise KeyError(f"result does not contain parameter map {param_name!r}")
    save_nifti(path, data, affine=affine, header=header, dtype=dtype)


def load_dicom_series(path: str | Path) -> tuple[NDArray[Any], dict[str, Any]]:
    """Load a DICOM series into a 3D or 4D array.

    Parameters
    ----------
    path : str or Path
        Directory containing DICOM files. Files are read recursively.

    Returns
    -------
    data : NDArray
        3D array with shape ``(z, y, x)`` for a single contrast, or 4D array
        with shape ``(z, y, x, n_volumes)`` for multi-echo or multi-contrast data.
    metadata : dict
        Acquisition parameters and spatial metadata. Echo time, repetition time,
        flip angle, and inversion time are reported as ``*_ms`` or ``*_deg`` keys.
    """
    import numpy as np

    pydicom = _import_pydicom()
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(root)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"No DICOM files found under {root}")

    records: list[dict[str, Any]] = []
    for file_path in files:
        try:
            ds = pydicom.dcmread(str(file_path), force=True)
        except Exception:
            continue
        if "PixelData" not in ds:
            continue
        pixel = ds.pixel_array.astype(np.float64)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        pixel = pixel * slope + intercept
        records.append(
            {
                "dataset": ds,
                "file": file_path,
                "pixel": pixel,
                "volume_key": _dicom_volume_key(ds),
                "z": _dicom_z_position(ds),
                "instance": int(getattr(ds, "InstanceNumber", len(records))),
            }
        )

    if not records:
        raise ValueError(f"No readable DICOM image slices found under {root}")

    volume_keys = sorted({record["volume_key"] for record in records})
    volumes = []
    datasets_by_volume = []
    for volume_key in volume_keys:
        volume_records = [record for record in records if record["volume_key"] == volume_key]
        volume_records.sort(key=lambda item: (item["z"], item["instance"]))
        shapes = {record["pixel"].shape for record in volume_records}
        if len(shapes) != 1:
            raise ValueError("All DICOM slices in a volume must have the same in-plane shape")
        volumes.append(np.stack([record["pixel"] for record in volume_records], axis=0))
        datasets_by_volume.append(volume_records[0]["dataset"])

    volume_shapes = {volume.shape for volume in volumes}
    if len(volume_shapes) != 1:
        raise ValueError("All DICOM volumes must have the same shape")
    if len(volumes) == 1:
        data = volumes[0]
    else:
        data = np.stack(volumes, axis=-1)

    metadata = _dicom_metadata(datasets_by_volume, records)
    metadata["files"] = [str(record["file"]) for record in sorted(records, key=lambda x: x["file"])]
    return data, metadata


def load_bids_relaxometry(
    path: str | Path,
    *,
    suffix: str | None = None,
) -> tuple[NDArray[Any], dict[str, Any]]:
    """Load minimal qMRI-BIDS relaxometry inputs from NIfTI and JSON sidecars.

    Parameters
    ----------
    path : str or Path
        A NIfTI file or a directory containing BIDS-like ``.nii``/``.nii.gz``
        files and JSON sidecars.
    suffix : str, optional
        Restrict directory loading to files whose BIDS suffix matches this value,
        for example ``"T1w"`` or ``"MEGRE"``.

    Returns
    -------
    data : NDArray
        One image array, or multiple 3D volumes stacked as ``(..., n_volumes)``.
    metadata : dict
        Normalized acquisition parameters plus raw sidecar metadata.
    """
    import numpy as np

    input_path = Path(path)
    files = _bids_nifti_files(input_path, suffix=suffix)
    if not files:
        raise ValueError(f"No BIDS NIfTI files found for {input_path}")

    arrays = []
    affines = []
    headers = []
    sidecars = []
    for file_path in files:
        data, affine, header = load_nifti(file_path)
        arrays.append(np.asarray(data))
        affines.append(affine)
        headers.append(header)
        sidecars.append(_load_bids_sidecar(file_path))

    data = _stack_bids_arrays(arrays)
    metadata = _bids_metadata(files, sidecars, affines, headers)
    return data, metadata


def _dicom_volume_key(ds: Any) -> tuple[Any, ...]:
    echo_number = getattr(ds, "EchoNumbers", None)
    temporal_position = getattr(ds, "TemporalPositionIdentifier", None)
    echo_time = getattr(ds, "EchoTime", None)
    inversion_time = getattr(ds, "InversionTime", None)
    flip_angle = getattr(ds, "FlipAngle", None)
    repetition_time = getattr(ds, "RepetitionTime", None)
    return (
        _maybe_float_or_none(echo_number),
        _maybe_float_or_none(temporal_position),
        _maybe_float_or_none(echo_time),
        _maybe_float_or_none(inversion_time),
        _maybe_float_or_none(flip_angle),
        _maybe_float_or_none(repetition_time),
    )


def _dicom_z_position(ds: Any) -> float:
    position = getattr(ds, "ImagePositionPatient", None)
    if position is not None and len(position) >= 3:
        return float(position[2])
    if hasattr(ds, "SliceLocation"):
        return float(ds.SliceLocation)
    return float(getattr(ds, "InstanceNumber", 0))


def _dicom_metadata(datasets: list[Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    first = datasets[0]
    z_positions = sorted({_dicom_z_position(record["dataset"]) for record in records})
    if len(z_positions) > 1:
        slice_spacing = float(np.median(np.diff(z_positions)))
    else:
        slice_spacing = float(getattr(first, "SpacingBetweenSlices", getattr(first, "SliceThickness", 1.0)))
    pixel_spacing = getattr(first, "PixelSpacing", [1.0, 1.0])
    voxel_spacing = (
        float(pixel_spacing[0]),
        float(pixel_spacing[1]),
        abs(slice_spacing),
    )
    return {
        "echo_time_ms": _dicom_values(datasets, "EchoTime"),
        "repetition_time_ms": _dicom_values(datasets, "RepetitionTime"),
        "flip_angle_deg": _dicom_values(datasets, "FlipAngle"),
        "inversion_time_ms": _dicom_values(datasets, "InversionTime"),
        "voxel_spacing_mm": voxel_spacing,
        "slice_positions": z_positions,
        "affine": np.diag([voxel_spacing[1], voxel_spacing[0], voxel_spacing[2], 1.0]),
    }


def _dicom_values(datasets: list[Any], name: str) -> float | list[float] | None:
    values = [_maybe_float_or_none(getattr(ds, name, None)) for ds in datasets]
    values = [value for value in values if value is not None]
    if not values:
        return None
    if len(values) == 1 or all(value == values[0] for value in values):
        return values[0]
    return values


def _maybe_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bids_nifti_files(path: Path, *, suffix: str | None) -> list[Path]:
    if path.is_file():
        if _is_nifti_path(path):
            return [path]
        raise ValueError(f"Expected a NIfTI file, got {path}")
    if not path.exists():
        raise FileNotFoundError(path)
    files = sorted(p for p in path.rglob("*") if p.is_file() and _is_nifti_path(p))
    if suffix is not None:
        files = [p for p in files if _bids_suffix(p) == suffix]
    return files


def _is_nifti_path(path: Path) -> bool:
    return path.name.endswith(".nii") or path.name.endswith(".nii.gz")


def _bids_suffix(path: Path) -> str:
    name = path.name.removesuffix(".nii.gz").removesuffix(".nii")
    return name.split("_")[-1]


def _bids_sidecar_path(path: Path) -> Path:
    if path.name.endswith(".nii.gz"):
        return path.with_name(path.name.removesuffix(".nii.gz") + ".json")
    return path.with_suffix(".json")


def _load_bids_sidecar(path: Path) -> dict[str, Any]:
    sidecar = _bids_sidecar_path(path)
    if not sidecar.exists():
        return {}
    with sidecar.open("r", encoding="utf-8") as f:
        content = json.load(f)
    if not isinstance(content, dict):
        raise ValueError(f"BIDS sidecar must contain a JSON object: {sidecar}")
    return content


def _stack_bids_arrays(arrays: list[NDArray[Any]]) -> NDArray[Any]:
    import numpy as np

    if len(arrays) == 1:
        return arrays[0]
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError("All BIDS NIfTI inputs must have the same shape")
    if arrays[0].ndim == 3:
        return np.stack(arrays, axis=-1)
    if arrays[0].ndim == 4:
        return np.concatenate(arrays, axis=-1)
    raise ValueError("BIDS relaxometry inputs must be 3D or 4D NIfTI images")


def _bids_metadata(
    files: list[Path],
    sidecars: list[dict[str, Any]],
    affines: list[NDArray[Any]],
    headers: list[Any],
) -> dict[str, Any]:
    return {
        "files": [str(path) for path in files],
        "sidecars": sidecars,
        "affine": affines[0],
        "headers": headers,
        "echo_time_ms": _bids_seconds_values(sidecars, "EchoTime"),
        "repetition_time_ms": _bids_seconds_values(sidecars, "RepetitionTime"),
        "flip_angle_deg": _bids_raw_values(sidecars, "FlipAngle"),
        "inversion_time_ms": _bids_seconds_values(sidecars, "InversionTime"),
        "bids_entities": [_parse_bids_entities(path) for path in files],
    }


def _bids_seconds_values(sidecars: list[dict[str, Any]], key: str) -> float | list[float] | None:
    values = _bids_raw_values(sidecars, key)
    if values is None:
        return None
    if isinstance(values, list):
        return [float(value) * 1000.0 for value in values]
    return float(values) * 1000.0


def _bids_raw_values(sidecars: list[dict[str, Any]], key: str) -> float | list[float] | None:
    values = [sidecar[key] for sidecar in sidecars if key in sidecar]
    if not values:
        return None
    values = [float(value) for value in values]
    if len(values) == 1:
        return values[0]
    return values


def _parse_bids_entities(path: Path) -> dict[str, str]:
    name = path.name.removesuffix(".nii.gz").removesuffix(".nii")
    entities: dict[str, str] = {}
    for part in name.split("_"):
        if "-" in part:
            key, value = part.split("-", 1)
            entities[key] = value
        else:
            entities["suffix"] = part
    return entities
