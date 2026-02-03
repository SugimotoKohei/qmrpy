"""Tests for the EPG simulation module."""

import numpy as np
import pytest


class TestEPGCore:
    """Tests for epg/core.py."""

    def test_rf_rotation_matrix_180(self):
        """180 degree pulse should swap F+ and F-."""
        from qmrpy.epg.core import rf_rotation_matrix

        r = rf_rotation_matrix(180.0)
        assert r.shape == (3, 3)
        # 180 pulse: F+ -> F-, F- -> F+, Z -> -Z
        np.testing.assert_allclose(np.abs(r[0, 1]), 1.0, atol=1e-10)
        np.testing.assert_allclose(np.abs(r[1, 0]), 1.0, atol=1e-10)

    def test_rf_rotation_matrix_90(self):
        """90 degree pulse should rotate Z to transverse."""
        from qmrpy.epg.core import rf_rotation_matrix

        r = rf_rotation_matrix(90.0)
        assert r.shape == (3, 3)
        # Check that Z component contributes to F+/F-
        assert np.abs(r[0, 2]) > 0
        assert np.abs(r[1, 2]) > 0

    def test_relaxation_operator(self):
        """Test relaxation operator values."""
        from qmrpy.epg.core import relaxation_operator

        e = relaxation_operator(t_ms=100.0, t1_ms=1000.0, t2_ms=100.0)
        assert e.shape == (3,)
        # E2 should be exp(-100/100) = exp(-1)
        np.testing.assert_allclose(e[0], np.exp(-1), atol=1e-10)
        np.testing.assert_allclose(e[1], np.exp(-1), atol=1e-10)
        # E1 should be exp(-100/1000) = exp(-0.1)
        np.testing.assert_allclose(e[2], np.exp(-0.1), atol=1e-10)

    def test_epg_simulator_reset(self):
        """Test simulator initialization."""
        from qmrpy.epg.core import EPGSimulator

        sim = EPGSimulator(n_states=10, t1_ms=1000.0, t2_ms=100.0)
        sim.reset(m0=2.0)
        assert sim.states[0, 2] == 2.0  # Z0 = M0
        assert sim.states[0, 0] == 0.0  # F+ = 0
        assert sim.states[0, 1] == 0.0  # F- = 0


class TestEPGSpinEcho:
    """Tests for epg/epg_se.py."""

    def test_cpmg_shape(self):
        """Test CPMG output shape."""
        from qmrpy.epg import epg_se

        signal = epg_se.cpmg(t2_ms=80, t1_ms=1000, te_ms=10, n_echoes=32)
        assert signal.shape == (32,)

    def test_cpmg_decay(self):
        """CPMG signal should decay over time."""
        from qmrpy.epg import epg_se

        signal = epg_se.cpmg(t2_ms=80, t1_ms=1000, te_ms=10, n_echoes=32)
        # Signal should decrease
        assert signal[0] > signal[-1]
        # All values should be positive
        assert np.all(signal >= 0)

    def test_cpmg_ideal_180(self):
        """With ideal 180 pulses and long T1, should match exp decay."""
        from qmrpy.epg import epg_se

        t2_ms = 80.0
        te_ms = 10.0
        n_echoes = 10
        signal = epg_se.cpmg(
            t2_ms=t2_ms, t1_ms=10000, te_ms=te_ms, n_echoes=n_echoes, b1=1.0
        )
        # Normalize
        signal_norm = signal / signal[0]
        # Expected pure T2 decay
        te_array = te_ms * np.arange(1, n_echoes + 1)
        expected = np.exp(-te_array / t2_ms)
        expected_norm = expected / expected[0]
        # Should be close (not exact due to EPG stimulated echo effects)
        np.testing.assert_allclose(signal_norm, expected_norm, rtol=0.05)

    def test_cpmg_b1_effect(self):
        """B1 < 1 should change the decay curve."""
        from qmrpy.epg import epg_se

        signal_b1_1 = epg_se.cpmg(t2_ms=80, t1_ms=1000, te_ms=10, n_echoes=32, b1=1.0)
        signal_b1_08 = epg_se.cpmg(t2_ms=80, t1_ms=1000, te_ms=10, n_echoes=32, b1=0.8)
        # Signals should be different
        assert not np.allclose(signal_b1_1, signal_b1_08)

    def test_mese_alias(self):
        """MESE should be alias for CPMG."""
        from qmrpy.epg import epg_se

        signal_cpmg = epg_se.cpmg(t2_ms=80, t1_ms=1000, te_ms=10, n_echoes=32)
        signal_mese = epg_se.mese(t2_ms=80, t1_ms=1000, te_ms=10, n_echoes=32)
        np.testing.assert_array_equal(signal_cpmg, signal_mese)

    def test_tse_variable_angles(self):
        """TSE with variable angles should work."""
        from qmrpy.epg import epg_se

        angles = [180, 160, 140, 120, 100, 80, 60, 40]
        signal = epg_se.tse(t2_ms=80, t1_ms=1000, te_ms=10, etl=8, refocus_angles_deg=angles)
        assert signal.shape == (8,)
        assert np.all(signal >= 0)

    def test_tse_constant_equals_cpmg(self):
        """TSE with constant 180 should equal CPMG."""
        from qmrpy.epg import epg_se

        signal_cpmg = epg_se.cpmg(t2_ms=80, t1_ms=1000, te_ms=10, n_echoes=8)
        signal_tse = epg_se.tse(t2_ms=80, t1_ms=1000, te_ms=10, etl=8, refocus_angles_deg=None)
        np.testing.assert_allclose(signal_cpmg, signal_tse, rtol=1e-10)

    def test_decay_curve_native(self):
        """Test native backend decay curve."""
        from qmrpy.epg import epg_se

        dc = epg_se.decay_curve(t2_ms=80, t1_ms=1000, te_ms=10, etl=32, backend="native")
        assert dc.shape == (32,)
        # Should be normalized
        np.testing.assert_allclose(dc[0], 1.0, atol=1e-10)


class TestEPGGradientEcho:
    """Tests for epg/epg_gre.py."""

    def test_spgr_shape(self):
        """Test SPGR output shape."""
        from qmrpy.epg import epg_gre

        signal = epg_gre.spgr(t1_ms=1000, tr_ms=10, fa_deg=15, n_pulses=100)
        assert signal.shape == (100,)

    def test_spgr_steady_state(self):
        """SPGR should reach steady state."""
        from qmrpy.epg import epg_gre

        signal = epg_gre.spgr(t1_ms=1000, tr_ms=10, fa_deg=15, n_pulses=500)
        # Last values should be nearly constant
        np.testing.assert_allclose(signal[-10:], signal[-1], rtol=0.01)

    def test_spgr_steady_state_analytical(self):
        """Compare SPGR simulation to Ernst equation."""
        from qmrpy.epg import epg_gre

        t1_ms = 1000
        tr_ms = 10
        fa_deg = 15

        # Numerical simulation
        signal_sim = epg_gre.spgr(t1_ms=t1_ms, tr_ms=tr_ms, fa_deg=fa_deg, n_pulses=500)
        ss_sim = signal_sim[-1]

        # Analytical Ernst equation
        ss_analytical = epg_gre.spgr_steady_state(t1_ms=t1_ms, tr_ms=tr_ms, fa_deg=fa_deg)

        np.testing.assert_allclose(ss_sim, ss_analytical, rtol=0.01)

    def test_ernst_angle(self):
        """Test Ernst angle calculation."""
        from qmrpy.epg import epg_gre

        t1_ms = 1000
        tr_ms = 10
        ernst = epg_gre.ernst_angle(t1_ms=t1_ms, tr_ms=tr_ms)

        # Check that Ernst angle gives maximum signal
        angles = np.linspace(1, 90, 100)
        signals = [epg_gre.spgr_steady_state(t1_ms=t1_ms, tr_ms=tr_ms, fa_deg=a) for a in angles]
        max_angle = angles[np.argmax(signals)]

        np.testing.assert_allclose(ernst, max_angle, atol=1.0)

    def test_bssfp_shape(self):
        """Test bSSFP output shape."""
        from qmrpy.epg import epg_gre

        signal = epg_gre.bssfp(t1_ms=1000, t2_ms=80, tr_ms=5, fa_deg=45, n_pulses=100)
        assert signal.shape == (100,)
        # Should be complex
        assert signal.dtype == np.complex128

    def test_bssfp_vs_spgr(self):
        """bSSFP should give different signal than SPGR for same parameters."""
        from qmrpy.epg import epg_gre

        signal_spgr = epg_gre.spgr(t1_ms=1000, tr_ms=5, fa_deg=45, n_pulses=100)
        signal_bssfp = epg_gre.bssfp(t1_ms=1000, t2_ms=80, tr_ms=5, fa_deg=45, n_pulses=100)

        # Different sequences should give different signals
        assert not np.allclose(signal_spgr, np.abs(signal_bssfp))

    def test_ssfp_fid_shape(self):
        """Test SSFP-FID output shape."""
        from qmrpy.epg import epg_gre

        signal = epg_gre.ssfp_fid(t1_ms=1000, t2_ms=80, tr_ms=10, fa_deg=30, n_pulses=50)
        assert signal.shape == (50,)

    def test_ssfp_echo_shape(self):
        """Test SSFP-Echo output shape."""
        from qmrpy.epg import epg_gre

        signal = epg_gre.ssfp_echo(t1_ms=1000, t2_ms=80, tr_ms=10, fa_deg=30, n_pulses=50)
        assert signal.shape == (50,)


class TestEPGValidation:
    """Validation tests comparing with known results."""

    def test_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        from qmrpy.epg import epg_se, epg_gre

        with pytest.raises(ValueError):
            epg_se.cpmg(t2_ms=-80, t1_ms=1000, te_ms=10, n_echoes=32)

        with pytest.raises(ValueError):
            epg_se.cpmg(t2_ms=80, t1_ms=1000, te_ms=10, n_echoes=0)

        with pytest.raises(ValueError):
            epg_gre.spgr(t1_ms=-1000, tr_ms=10, fa_deg=15)

        with pytest.raises(ValueError):
            epg_gre.bssfp(t1_ms=1000, t2_ms=80, tr_ms=0, fa_deg=45)
