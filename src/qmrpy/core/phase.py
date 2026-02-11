from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import ArrayLike, NDArray
else:
    ArrayLike = Any
    NDArray = Any


def as_phase(values: ArrayLike) -> NDArray[np.float64]:
    import numpy as np

    arr = np.asarray(values)
    if np.iscomplexobj(arr):
        return np.angle(arr).astype(np.float64)
    return np.asarray(arr, dtype=np.float64)


def wrap_phase(phi_rad: NDArray[np.float64]) -> NDArray[np.float64]:
    import numpy as np

    return np.angle(np.exp(1j * phi_rad)).astype(np.float64)
