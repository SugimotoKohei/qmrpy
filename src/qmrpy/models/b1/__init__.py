from qmrpy.core.fit_protocols import ResultAdapterBase

from .afi import B1AFI as _B1Afi
from .bloch_siegert import B1BlochSiegert as _B1BlochSiegert
from .dam import B1DAM as _B1Dam


class B1AFI(ResultAdapterBase):
    _IMPL_CLS = _B1Afi
    _PARAM_KEYS = ("b1_raw",)


class B1DAM(ResultAdapterBase):
    _IMPL_CLS = _B1Dam
    _PARAM_KEYS = ("b1_raw",)


class B1BlochSiegert(ResultAdapterBase):
    _IMPL_CLS = _B1BlochSiegert
    _PARAM_KEYS = ("b1_raw",)


__all__ = ["B1AFI", "B1BlochSiegert", "B1DAM"]
