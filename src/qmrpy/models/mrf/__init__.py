from qmrpy.core.fit_protocols import ResultAdapterBase

from .mrf import MRFDictionary as _MRFDictionary


class MRFDictionary(ResultAdapterBase):
    _IMPL_CLS = _MRFDictionary
    _PARAM_KEYS = ("m0", "t1_ms", "t2_ms")
    _RMSE_KEYS = ()
    _N_POINTS_KEYS = ()


__all__ = ["MRFDictionary"]
