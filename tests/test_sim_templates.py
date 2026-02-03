def test_sim_templates_build_sequences_and_protocols():
    import pytest

    pytest.importorskip("pypulseq")

    from qmrpy.sim import (
        build_cpmg_sequence,
        build_se_sequence,
        build_flash_sequence,
        mrzero_protocol_cpmg,
        mrzero_protocol_se,
        mrzero_protocol_flash,
    )

    seq_se = build_se_sequence()
    seq_cpmg = build_cpmg_sequence()
    seq_flash = build_flash_sequence()

    for seq in (seq_se, seq_cpmg, seq_flash):
        assert hasattr(seq, "write")

    proto_se = mrzero_protocol_se()
    proto_cpmg = mrzero_protocol_cpmg()
    proto_flash = mrzero_protocol_flash()

    for proto in (proto_se, proto_cpmg, proto_flash):
        assert proto.simulation_backend == "mrzero_bloch"
        assert "mrzero" in proto.model_protocol
        assert "model" in proto.model_protocol
        assert callable(proto.model_protocol["mrzero"]["data_factory"])
