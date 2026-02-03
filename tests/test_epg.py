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

    def test_epg_cpmg_decaes_matches_reference(self):
        """Test that epg_cpmg_decaes matches the reference implementation."""
        from qmrpy.epg.core import epg_cpmg_decaes
        from qmrpy.models.t2.decaes_t2 import epg_decay_curve as decaes_ref

        t2_ms, t1_ms, te_ms, etl = 80.0, 1000.0, 10.0, 32
        
        # Reference
        ref = decaes_ref(etl=etl, alpha_deg=180.0, te_ms=te_ms,
                         t2_ms=t2_ms, t1_ms=t1_ms, beta_deg=180.0)
        # New implementation
        new = epg_cpmg_decaes(etl=etl, alpha_deg=180.0, te_ms=te_ms,
                              t2_ms=t2_ms, t1_ms=t1_ms, beta_deg=180.0)
        
        np.testing.assert_allclose(new, ref, atol=1e-12)


class TestEPGSpinEcho:
    """Tests for epg/epg_se.py."""

    def test_se_shape(self):
        """Test SE output shape."""
        from qmrpy.epg import epg_se

        # Single echo
        signal = epg_se.se(t2_ms=80, t1_ms=1000, te_ms=10)
        assert signal.shape == (1,)

        # Multi-echo
        signal = epg_se.se(t2_ms=80, t1_ms=1000, te_ms=10, n_echoes=32)
        assert signal.shape == (32,)

    def test_se_decay(self):
        """SE signal should decay over time."""
        from qmrpy.epg import epg_se

        signal = epg_se.se(t2_ms=80, t1_ms=1000, te_ms=10, n_echoes=32)
        # Signal should decrease
        assert signal[0] > signal[-1]
        # All values should be positive
        assert np.all(signal >= 0)

    def test_se_cpmg_vs_cp_b1_perfect(self):
        """With B1=1.0, CP and CPMG should give same results."""
        from qmrpy.epg import epg_se

        cpmg = epg_se.se(t2_ms=80, t1_ms=1000, te_ms=10, n_echoes=8, b1=1.0, cpmg=True)
        cp = epg_se.se(t2_ms=80, t1_ms=1000, te_ms=10, n_echoes=8, b1=1.0, cpmg=False)
        np.testing.assert_allclose(cpmg, cp, rtol=1e-10)

    def test_se_cpmg_vs_cp_b1_imperfect(self):
        """With B1<1.0, CP and CPMG should give different results."""
        from qmrpy.epg import epg_se

        cpmg = epg_se.se(t2_ms=80, t1_ms=1000, te_ms=10, n_echoes=8, b1=0.8, cpmg=True)
        cp = epg_se.se(t2_ms=80, t1_ms=1000, te_ms=10, n_echoes=8, b1=0.8, cpmg=False)
        # They should be different (especially at later echoes)
        assert np.max(np.abs(cpmg - cp)) > 0.1

    def test_tse_constant_equals_se(self):
        """TSE with constant 180 should equal SE."""
        from qmrpy.epg import epg_se

        signal_se = epg_se.se(t2_ms=80, t1_ms=1000, te_ms=10, n_echoes=8)
        signal_tse = epg_se.tse(t2_ms=80, t1_ms=1000, te_ms=10, etl=8)
        np.testing.assert_allclose(signal_se, signal_tse, rtol=1e-10)


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

    def test_spgr_matches_ernst(self):
        """SPGR simulation should match Ernst equation."""
        from qmrpy.epg import epg_gre

        t1_ms = 1000
        tr_ms = 10

        for fa_deg in [5, 10, 15, 20, 30, 45]:
            # Ernst equation
            e1 = np.exp(-tr_ms / t1_ms)
            alpha = np.deg2rad(fa_deg)
            ernst = np.sin(alpha) * (1 - e1) / (1 - e1 * np.cos(alpha))

            # EPG simulation
            signal = epg_gre.spgr(t1_ms=t1_ms, tr_ms=tr_ms, fa_deg=fa_deg, n_pulses=500)
            epg_ss = signal[-1]

            np.testing.assert_allclose(epg_ss, ernst, rtol=0.001,
                                       err_msg=f"Mismatch at FA={fa_deg}")

    def test_spgr_steady_state_analytical(self):
        """Compare SPGR simulation to analytical function."""
        from qmrpy.epg import epg_gre

        t1_ms, tr_ms, fa_deg = 1000, 10, 15

        signal_sim = epg_gre.spgr(t1_ms=t1_ms, tr_ms=tr_ms, fa_deg=fa_deg, n_pulses=500)
        ss_sim = signal_sim[-1]
        ss_analytical = epg_gre.spgr_steady_state(t1_ms=t1_ms, tr_ms=tr_ms, fa_deg=fa_deg)

        np.testing.assert_allclose(ss_sim, ss_analytical, rtol=0.001)

    def test_ernst_angle(self):
        """Test Ernst angle calculation."""
        from qmrpy.epg import epg_gre

        t1_ms = 1000
        tr_ms = 10
        ernst = epg_gre.ernst_angle(t1_ms=t1_ms, tr_ms=tr_ms)

        # Expected: arccos(exp(-TR/T1))
        expected = np.rad2deg(np.arccos(np.exp(-tr_ms / t1_ms)))
        np.testing.assert_allclose(ernst, expected, atol=1e-10)

    def test_bssfp_shape(self):
        """Test bSSFP output shape."""
        from qmrpy.epg import epg_gre

        signal = epg_gre.bssfp(t1_ms=1000, t2_ms=80, tr_ms=5, fa_deg=45, n_pulses=100)
        assert signal.shape == (100,)
        assert signal.dtype == np.complex128

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
    """Validation tests for invalid parameters."""

    def test_invalid_parameters_se(self):
        """Test that invalid parameters raise errors for SE."""
        from qmrpy.epg import epg_se

        with pytest.raises(ValueError):
            epg_se.se(t2_ms=-80, t1_ms=1000, te_ms=10, n_echoes=32)

        with pytest.raises(ValueError):
            epg_se.se(t2_ms=80, t1_ms=1000, te_ms=10, n_echoes=0)

        with pytest.raises(ValueError):
            epg_se.se(t2_ms=80, t1_ms=1000, te_ms=0, n_echoes=32)

    def test_invalid_parameters_gre(self):
        """Test that invalid parameters raise errors for GRE."""
        from qmrpy.epg import epg_gre

        with pytest.raises(ValueError):
            epg_gre.spgr(t1_ms=-1000, tr_ms=10, fa_deg=15)

        with pytest.raises(ValueError):
            epg_gre.spgr(t1_ms=1000, tr_ms=0, fa_deg=15)

        with pytest.raises(ValueError):
            epg_gre.bssfp(t1_ms=1000, t2_ms=80, tr_ms=0, fa_deg=45)


class TestEPGWeigel:
    """Cross-validation tests against Weigel reference implementation."""

    @pytest.fixture
    def weigel_reference(self):
        """Load Weigel reference data if available."""
        import csv
        import os

        ref_path = os.path.join(
            os.path.dirname(__file__), "epg_reference", "weigel_reference.csv"
        )
        if not os.path.exists(ref_path):
            pytest.skip("Weigel reference file not found")

        results = {}
        with open(ref_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                test = row["test"]
                echo = int(row["echo"])
                value = float(row["value"])
                if test not in results:
                    results[test] = {}
                results[test][echo] = value
        return results

    def test_se_weigel_b1_1_0(self, weigel_reference):
        """SE with B1=1.0 should match Weigel reference."""
        from qmrpy.epg import epg_se

        T2, T1, ESP, N = 80.0, 1000.0, 10.0, 32
        signal = epg_se.se(t2_ms=T2, t1_ms=T1, te_ms=ESP, n_echoes=N, b1=1.0)
        weigel = np.array([weigel_reference["cpmg_b1_1.0"][i] for i in range(1, N + 1)])
        np.testing.assert_allclose(signal, weigel, rtol=1e-10)

    def test_se_weigel_b1_0_9(self, weigel_reference):
        """SE with B1=0.9 should match Weigel reference."""
        from qmrpy.epg import epg_se

        T2, T1, ESP, N = 80.0, 1000.0, 10.0, 32
        signal = epg_se.se(t2_ms=T2, t1_ms=T1, te_ms=ESP, n_echoes=N, b1=0.9)
        weigel = np.array([weigel_reference["cpmg_b1_0.9"][i] for i in range(1, N + 1)])
        np.testing.assert_allclose(signal, weigel, rtol=1e-10)

    def test_se_weigel_b1_0_8(self, weigel_reference):
        """SE with B1=0.8 should match Weigel reference."""
        from qmrpy.epg import epg_se

        T2, T1, ESP, N = 80.0, 1000.0, 10.0, 32
        signal = epg_se.se(t2_ms=T2, t1_ms=T1, te_ms=ESP, n_echoes=N, b1=0.8)
        weigel = np.array([weigel_reference["cpmg_b1_0.8"][i] for i in range(1, N + 1)])
        np.testing.assert_allclose(signal, weigel, rtol=1e-10)
