from qmrpy.core.fit_protocols import ResultAdapterBase

from .mt import MTR as _MTR
from .mt import MTsat as _MTsat


class MTR(ResultAdapterBase):
    _IMPL_CLS = _MTR
    _PARAM_KEYS = ("mtr", "mtr_percent")


class MTsat(ResultAdapterBase):
    _IMPL_CLS = _MTsat
    _PARAM_KEYS = ("mtsat", "mtsat_percent")


__all__ = ["MTR", "MTsat"]
