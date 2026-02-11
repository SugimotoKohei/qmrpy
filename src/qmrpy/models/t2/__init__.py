from typing import Any

from qmrpy.core.fit_protocols import ResultAdapterBase

from .decaes_t2 import T2DECAESMap as _DECAEST2Map
from .decaes_t2part import T2DECAESPart as _DECAEST2Part
from .emc_t2 import T2EMC as _EMCT2
from .epg_t2 import T2EPG as _EPGT2
from .mono_t2 import T2Mono as _MonoT2
from .mwf import T2MultiComponent as _MultiComponentT2


class T2Mono(ResultAdapterBase):
    _IMPL_CLS = _MonoT2
    _PARAM_KEYS = ("m0", "t2_ms", "offset", "b1")


class T2EPG(ResultAdapterBase):
    _IMPL_CLS = _EPGT2
    _PARAM_KEYS = ("m0", "t2_ms", "offset", "b1")


class T2EMC(ResultAdapterBase):
    _IMPL_CLS = _EMCT2
    _PARAM_KEYS = ("m0", "t2_ms", "b1")


class T2DECAESMap(ResultAdapterBase):
    _IMPL_CLS = _DECAEST2Map
    _PARAM_KEYS = (
        "echotimes_ms",
        "t2times_ms",
        "alpha_deg",
        "gdn",
        "ggm",
        "gva",
        "fnr",
        "snr",
        "distribution",
    )

    def fit_image(self, signal: Any, **kwargs: Any) -> dict[str, Any]:
        maps, distributions = self._impl.fit_image(signal, **kwargs)
        flat = dict(maps)
        flat["distribution"] = distributions
        return self._nest(flat)


class T2DECAESPart(ResultAdapterBase):
    _IMPL_CLS = _DECAEST2Part
    _PARAM_KEYS = ("sfr", "sgm", "mfr", "mgm")


class T2MultiComponent(ResultAdapterBase):
    _IMPL_CLS = _MultiComponentT2
    _PARAM_KEYS = (
        "weights",
        "t2_basis_ms",
        "mwf",
        "t2mw_ms",
        "t2iew_ms",
        "gmt2_ms",
        "mw_weight",
        "iew_weight",
        "total_weight",
    )


__all__ = [
    "T2DECAESMap",
    "T2DECAESPart",
    "T2EMC",
    "T2EPG",
    "T2Mono",
    "T2MultiComponent",
]
