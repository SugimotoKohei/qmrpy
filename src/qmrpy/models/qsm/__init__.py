from typing import Any

from qmrpy.core.fit_protocols import ResultAdapterBase

from .gradient_mask import calc_gradient_mask_from_magnitude
from .pipeline import QSMSplitBregman as _QsmSplitBregman
from .sharp import background_removal_sharp
from .split_bregman import calc_chi_l2, qsm_split_bregman
from .unwrap import unwrap_phase_laplacian


class QSMSplitBregman(ResultAdapterBase):
    _IMPL_CLS = _QsmSplitBregman
    _PARAM_KEYS = ("chi_l2", "chi_l2_pcg", "chi_sb", "nfm")

    def fit(
        self,
        phase: Any,
        mask: Any,
        *,
        magnitude: Any | None = None,
        image_resolution_mm: Any | None = None,
    ) -> dict[str, Any]:
        out = self._impl.fit(
            phase,
            mask,
            magnitude=magnitude,
            image_resolution_mm=image_resolution_mm,
        )
        return self._nest(out)


__all__ = [
    "qsm_split_bregman",
    "calc_chi_l2",
    "unwrap_phase_laplacian",
    "background_removal_sharp",
    "calc_gradient_mask_from_magnitude",
    "QSMSplitBregman",
]
