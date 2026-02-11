from qmrpy.core.fit_protocols import ResultAdapterBase

from .despot1_hifi import T1DESPOT1HIFI as _T1DESPOT1HIFI
from .inversion_recovery import T1InversionRecovery as _T1InversionRecovery
from .mp2rage import T1MP2RAGE as _T1MP2RAGE
from .vfa_t1 import T1VFA as _T1VFA


class T1VFA(ResultAdapterBase):
    _IMPL_CLS = _T1VFA
    _PARAM_KEYS = ("m0", "t1_ms", "b1")

    def fit_linear(self, signal, **kwargs):  # type: ignore[no-untyped-def]
        return self._nest(self._impl.fit_linear(signal, **kwargs))


class T1InversionRecovery(ResultAdapterBase):
    _IMPL_CLS = _T1InversionRecovery
    _PARAM_KEYS = ("t1_ms", "ra", "rb")


class T1DESPOT1HIFI(ResultAdapterBase):
    _IMPL_CLS = _T1DESPOT1HIFI
    _PARAM_KEYS = ("m0", "t1_ms", "b1")


class T1MP2RAGE(ResultAdapterBase):
    _IMPL_CLS = _T1MP2RAGE
    _PARAM_KEYS = ("m0", "t1_ms", "b1")


__all__ = ["T1DESPOT1HIFI", "T1InversionRecovery", "T1MP2RAGE", "T1VFA"]
