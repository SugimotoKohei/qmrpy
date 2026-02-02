"""I/O utilities for qmrpy."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import ArrayLike, NDArray
else:
    ArrayLike = Any
    NDArray = Any


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
