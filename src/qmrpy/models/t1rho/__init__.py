from qmrpy.core.fit_protocols import ResultAdapterBase

from .t1rho import T1Rho as _T1Rho


class T1Rho(ResultAdapterBase):
    _IMPL_CLS = _T1Rho
    _PARAM_KEYS = ("m0", "t1rho_ms")


__all__ = ["T1Rho"]
