from qmrpy.core.fit_protocols import ResultAdapterBase

from .estatics import T2StarESTATICS as _R2StarEstatics
from .r2star import T2StarComplexR2 as _R2StarComplex
from .r2star import T2StarMonoR2 as _R2StarMono


class T2StarMonoR2(ResultAdapterBase):
    _IMPL_CLS = _R2StarMono
    _PARAM_KEYS = ("s0", "t2star_ms", "r2star_hz", "offset")


class T2StarComplexR2(ResultAdapterBase):
    _IMPL_CLS = _R2StarComplex
    _PARAM_KEYS = ("s0", "t2star_ms", "r2star_hz", "delta_f_hz", "phi0_rad")


class T2StarESTATICS(ResultAdapterBase):
    _IMPL_CLS = _R2StarEstatics
    _PARAM_KEYS = ("s0", "s0_per_contrast", "t2star_ms", "r2star_hz")


__all__ = ["T2StarComplexR2", "T2StarESTATICS", "T2StarMonoR2"]
