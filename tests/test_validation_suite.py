from __future__ import annotations

import csv
from pathlib import Path


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _case_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row["domain"], row["model"], row["case"])


def test_validation_suite_core_output_schema(tmp_path: Path) -> None:
    from scripts import summarize_parity

    out_dir = tmp_path / "core_validation"
    rc = summarize_parity.main(
        [
            "--suite",
            "core",
            "--formats",
            "csv,json",
            "--out-dir",
            str(out_dir),
            "--config",
            str(Path("configs/exp/validation_core.toml")),
        ]
    )
    assert rc == 0

    core_case_csv = out_dir / "core_validation.csv"
    core_metric_csv = out_dir / "core_validation_metrics.csv"
    summary_json = out_dir / "summary.json"

    assert core_case_csv.exists()
    assert core_metric_csv.exists()
    assert summary_json.exists()

    case_rows = _read_csv_rows(core_case_csv)
    metric_rows = _read_csv_rows(core_metric_csv)

    assert case_rows
    assert metric_rows

    domains = {row["domain"] for row in case_rows}
    assert domains == {"T1", "T2", "T2*", "B1", "B0", "QSM", "Simulation"}

    models = {row["model"] for row in case_rows}
    assert {
        "mono_t2",
        "r2star_mono",
        "r2star_complex",
        "vfa_t1",
        "despot1_hifi",
        "mp2rage",
        "inversion_recovery",
        "epg_t2",
        "emc_t2",
        "mwf",
        "b1_dam",
        "b1_bloch_siegert",
        "b0_dual_echo",
        "b0_multi_echo",
        "qsm",
        "simulation",
    }.issubset(models)

    metric_pass_map: dict[tuple[str, str, str], list[int]] = {}
    for row in metric_rows:
        float(row["value"])
        float(row["threshold"])
        assert row["pass"] in {"0", "1"}
        metric_pass_map.setdefault(_case_key(row), []).append(int(row["pass"]))

    for row in case_rows:
        key = _case_key(row)
        assert key in metric_pass_map
        expected = int(all(metric_pass_map[key]))
        assert int(row["pass"]) == expected
        float(row["primary_value"])
        float(row["primary_threshold"])


def test_validation_suite_seed_reproducibility(tmp_path: Path) -> None:
    from scripts import summarize_parity

    out_dir_1 = tmp_path / "run_1"
    out_dir_2 = tmp_path / "run_2"
    args = [
        "--suite",
        "core",
        "--formats",
        "csv",
        "--config",
        str(Path("configs/exp/validation_core.toml")),
    ]

    rc1 = summarize_parity.main([*args, "--out-dir", str(out_dir_1)])
    rc2 = summarize_parity.main([*args, "--out-dir", str(out_dir_2)])
    assert rc1 == 0
    assert rc2 == 0

    rows_1 = _read_csv_rows(out_dir_1 / "core_validation.csv")
    rows_2 = _read_csv_rows(out_dir_2 / "core_validation.csv")
    map_1 = {_case_key(r): float(r["primary_value"]) for r in rows_1}
    map_2 = {_case_key(r): float(r["primary_value"]) for r in rows_2}

    assert map_1.keys() == map_2.keys()
    for key in map_1:
        assert abs(map_1[key] - map_2[key]) < 1e-12
