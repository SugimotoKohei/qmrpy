from __future__ import annotations

from typing import Any


def _import_mrzero() -> Any:
    try:
        import MRzeroCore as mr0  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional install
        raise ModuleNotFoundError(
            "MRzeroCore が見つかりません。MRzero を使うには MRzeroCore を別途インストールしてください。"
        ) from exc
    return mr0


def _normalize_sequence(seq_or_path: Any) -> Any:
    mr0 = _import_mrzero()
    if isinstance(seq_or_path, str):
        return mr0.Sequence.import_file(seq_or_path)
    return seq_or_path


def simulate_pdg(
    seq_or_path: Any,
    data: Any,
    *,
    max_states: int = 200,
    min_latent_signal: float = 1e-4,
    min_emitted_signal: float = 1e-4,
    print_progress: bool = True,
    return_graph: bool = False,
) -> Any:
    """MRzeroCore を用いた PDG/EPG シミュレーション。

    Parameters
    ----------
    seq_or_path : Sequence or str
        MRzeroCore の Sequence もしくは seq ファイルパス。
    data : SimData
        MRzeroCore の SimData。
    max_states : int, optional
        グラフ生成時の最大状態数。
    min_latent_signal : float, optional
        グラフ生成/実行のしきい値。
    min_emitted_signal : float, optional
        実行時の出力しきい値。
    print_progress : bool, optional
        進捗出力を行うか。
    return_graph : bool, optional
        True の場合は (signal, graph) を返す。
    """
    mr0 = _import_mrzero()
    seq = _normalize_sequence(seq_or_path)

    graph = mr0.compute_graph(
        seq,
        data,
        max_state_count=int(max_states),
        min_state_mag=float(min_latent_signal),
    )
    signal = mr0.execute_graph(
        graph,
        seq,
        data,
        min_emitted_signal=float(min_emitted_signal),
        min_latent_signal=float(min_latent_signal),
        print_progress=bool(print_progress),
    )
    if return_graph:
        return signal, graph
    return signal


def simulate_bloch(
    seq_or_path: Any,
    data: Any,
    *,
    spin_count: int = 5000,
    perfect_spoiling: bool = False,
    print_progress: bool = True,
    spin_dist: str = "rand",
    r2_seed: int | None = None,
) -> Any:
    """MRzeroCore を用いた Bloch（isochromat）シミュレーション。"""
    mr0 = _import_mrzero()
    seq = _normalize_sequence(seq_or_path)
    return mr0.isochromat_sim(
        seq,
        data,
        spin_count=int(spin_count),
        perfect_spoiling=bool(perfect_spoiling),
        print_progress=bool(print_progress),
        spin_dist=str(spin_dist),
        r2_seed=r2_seed,
    )
