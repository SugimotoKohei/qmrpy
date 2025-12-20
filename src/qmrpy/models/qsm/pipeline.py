from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .gradient_mask import calc_gradient_mask_from_magnitude
from .sharp import background_removal_sharp
from .split_bregman import calc_chi_l2, qsm_split_bregman
from .unwrap import unwrap_phase_laplacian


@dataclass(frozen=True, slots=True)
class QsmSplitBregman:
    """High-level QSM pipeline (qMRLab qsm_sb, simplified).

    Steps:
      1) Laplacian phase unwrapping
      2) SHARP background removal
      3) Optional magnitude weighting
      4) Split-Bregman or L2 reconstruction
    """

    direction: str = "forward"
    sharp_filter: bool = True
    sharp_mode: str = "once"
    pad_size: tuple[int, int, int] = (9, 9, 9)
    magn_weighting: bool = False

    l1_regularized: bool = True
    l2_regularized: bool = False
    no_regularization: bool = False

    lambda_l1: float = 9.210553177e-4
    lambda_l2: float = 0.0316228

    def fit(
        self,
        phase_gre: NDArray[np.float64],
        mask: NDArray[np.float64],
        *,
        magn_gre: NDArray[np.float64] | None = None,
        image_resolution_mm: NDArray[np.float64] | None = None,
    ) -> dict[str, Any]:
        phase = np.asarray(phase_gre, dtype=np.float64)
        mask = np.asarray(mask, dtype=np.float64)
        if phase.shape != mask.shape:
            raise ValueError("phase_gre and mask must have same shape")

        if image_resolution_mm is None:
            image_resolution_mm = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        else:
            image_resolution_mm = np.asarray(image_resolution_mm, dtype=np.float64)

        # 1) Laplacian unwrap
        phase_lunwrap = unwrap_phase_laplacian(phase)

        # 2) SHARP background removal
        if self.sharp_filter:
            phase_lunwrap_pad = np.pad(
                phase_lunwrap,
                ((self.pad_size[0], self.pad_size[0]), (self.pad_size[1], self.pad_size[1]), (self.pad_size[2], self.pad_size[2])),
                mode="constant",
            )
            mask_pad = np.pad(
                mask,
                ((self.pad_size[0], self.pad_size[0]), (self.pad_size[1], self.pad_size[1]), (self.pad_size[2], self.pad_size[2])),
                mode="constant",
            )
            nfm_sharp, mask_sharp = background_removal_sharp(
                phase_lunwrap_pad,
                mask_pad,
                filter_mode=self.sharp_mode,
            )
        else:
            nfm_sharp = phase_lunwrap
            mask_sharp = mask

        # 3) Magnitude weighting (optional)
        magn_weight = None
        if self.magn_weighting and magn_gre is not None:
            magn_weight = calc_gradient_mask_from_magnitude(
                magn=np.asarray(magn_gre, dtype=np.float64),
                mask_sharp=mask_sharp,
                pad_size=self.pad_size,
                direction=self.direction,
            )

        out: dict[str, Any] = {
            "unwrapped_phase": phase_lunwrap,
            "mask_out": mask_sharp,
        }

        # 4) Reconstruction
        if self.no_regularization:
            # no-regularization: just return SHARP-processed phase
            out["nfm"] = nfm_sharp
            return out

        if self.l2_regularized:
            chi_l2, chi_l2_pcg = calc_chi_l2(
                nfm_sharp,
                lambda_l2=self.lambda_l2,
                direction=self.direction,
                image_resolution_mm=image_resolution_mm,
                mask=mask_sharp,
                padding_size=self.pad_size,
                magn_weight=magn_weight,
            )
            out["chiL2"] = chi_l2
            if chi_l2_pcg is not None:
                out["chiL2pcg"] = chi_l2_pcg

        if self.l1_regularized:
            chi_sb = qsm_split_bregman(
                nfm_sharp,
                mask_sharp,
                lambda_l1=self.lambda_l1,
                lambda_l2=self.lambda_l2,
                direction=self.direction,
                image_resolution_mm=image_resolution_mm,
                pad_size=self.pad_size,
                precon_mag_weight=self.magn_weighting,
                magn_weight=magn_weight,
            )
            out["chiSB"] = chi_sb

        return out
