from qmrpy.core.fit_protocols import ResultAdapterBase

from .dual_echo import B0DualEcho as _B0DualEcho
from .multi_echo import B0MultiEcho as _B0MultiEcho


class B0DualEcho(ResultAdapterBase):
    _IMPL_CLS = _B0DualEcho
    _PARAM_KEYS = ("b0_hz", "phase0_rad")


class B0MultiEcho(ResultAdapterBase):
    _IMPL_CLS = _B0MultiEcho
    _PARAM_KEYS = ("b0_hz", "phase0_rad")


__all__ = ["B0DualEcho", "B0MultiEcho"]
