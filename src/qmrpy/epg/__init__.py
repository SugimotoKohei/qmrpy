"""EPG (Extended Phase Graph) simulation module.

This module provides EPG-based MRI signal simulations for various pulse sequences.

Submodules
----------
core : Core EPG engine with state transition matrices
epg_se : Spin Echo sequences (CPMG, TSE/FSE)
epg_gre : Gradient Echo sequences (SPGR, bSSFP)

Examples
--------
>>> from qmrpy.epg import epg_se, epg_gre

# Spin Echo simulation
>>> signal_se = epg_se.cpmg(t2_ms=80, t1_ms=1000, te_ms=10, n_echoes=32)

# Gradient Echo simulation
>>> signal_gre = epg_gre.spgr(t1_ms=1000, tr_ms=10, fa_deg=15)
>>> signal_ssfp = epg_gre.bssfp(t1_ms=1000, t2_ms=80, tr_ms=5, fa_deg=45)

References
----------
.. [1] Hennig J (1988). Multiecho imaging sequences with low refocusing flip angles.
       J Magn Reson, 78(3):397-407.
.. [2] Weigel M (2015). Extended phase graphs: dephasing, RF pulses, and echoes -
       pure and simple. J Magn Reson Imaging, 41(2):266-295.
"""

from . import epg_gre, epg_se
from .core import EPGSimulator, epg_cpmg_decaes, epg_weigel, rf_rotation_matrix, relaxation_operator

__all__ = [
    "epg_se",
    "epg_gre",
    "EPGSimulator",
    "epg_cpmg_decaes",
    "epg_weigel",
    "rf_rotation_matrix",
    "relaxation_operator",
]
