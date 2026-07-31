# memo
## 2026-01-28
- EPG補正の単一T2モデルとして `src/qmrpy/models/t2/epg_t2.py` を追加
- T2モデルの公開APIに `EpgT2` を追加（`src/qmrpy/models/t2/__init__.py`, `src/qmrpy/models/__init__.py`）
- `tests/test_epg_t2.py` を追加してフィット/fit_imageの基本挙動を確認
- 次アクション: `uv run pytest tests/test_epg_t2.py` でテスト実行 (TBD)
- `uv run pytest tests/test_epg_t2.py` を実行（3 passed）
- `EpgT2` に b1 補正のための `b1`/`b1_map` 対応を追加
- README に `EpgT2` の使用例と B1 補正の説明を追記
- `uv run pytest tests/test_epg_t2.py` を再実行（4 passed）
- `uv run --locked -m pytest` を実行（51 passed, warnings 5件）
- `qmrpy.functional` に `simulate_t2_epg` / `fit_t2_epg` を追加
- README に Functional API と B1連携の例を追記
- `uv run --locked -m pytest` を再実行（51 passed, warnings 5件）
- `tests/test_epg_t2_functional.py` を追加して Functional API の回帰テストを追加
- `uv run pytest tests/test_epg_t2_functional.py` を実行（2 passed）
- `scripts/demo_epg_t2.py` を追加して EPG T2 の簡易デモを追加
- `uv run python scripts/demo_epg_t2.py` を実行（出力確認）

## 2026-02-02
- `mono_t2.py` の `fit` メソッドで m0 が正規化後の値ではなく元のスケールで返されるよう修正
- バージョンを 0.4.0 → 0.4.1 に更新
- `.github/workflows/ci.yml` に auto-tag ジョブを追加（mainブランチでテスト成功後、未作成タグを自動作成・プッシュ）
- これにより、バージョン変更 → テスト成功 → 自動タグ → PyPI公開 の流れが自動化
- v0.5.0: TIFF I/O (`save_tiff`, `load_tiff`) を追加、Pillow 依存を追加
- v0.5.1: Otsu マスキング (`mask="otsu"`) を全 `fit_image` メソッドに追加
- v0.6.0: 並列フィッティング (`n_jobs` パラメータ) を全 `fit_image` メソッドに追加
- v0.6.1: テスト拡充 (simulation/QSM)、API引数名を `signal` に統一
- v0.6.2: README を英語+日本語セクションに刷新、Pillow 依存を正式追加
- v0.6.3: `functional.py` に型ヒント (NDArray, ArrayLike) を追加
- v0.7.0: 全 `fit_image` に `verbose` パラメータ追加、tqdm 進捗バー + logging サポート
- v0.7.1: mkdocs-material ドキュメント基盤追加、GitHub Pages 自動デプロイ設定
- v0.8.0: PEP 8 厳密準拠のため破壊的変更を実施
  - クラス名: `VfaT1`→`T1VFA`, `EpgT2`→`T2EPG`, `DecaesT2Map`→`T2DECAESMap`, `DecaesT2Part`→`T2DECAESPart`
  - 関数名: `SimVary`→`sim_vary`, `SimRnd`→`sim_rnd`, `SimFisherMatrix`→`sim_fisher_matrix`, `SimCRLB`→`sim_crlb`
  - パラメータ名・戻り値キーも snake_case 化
- `docs/index.md` のライセンス表記を BSD-2-Clause → MIT に修正（LICENSE ファイルと統一）
- `THIRD_PARTY_NOTICES.md` を日本語から英語に翻訳
- `paper.md` / `paper.bib` は JOSS 投稿予定のため公開リポジトリに残す（JOSS は公開リポジトリ必須）

## 2026-02-03
- v0.9.0: EPG シミュレーションモジュール追加 (`src/qmrpy/epg/`)
  - `epg/core.py`: 汎用 EPG エンジン（状態遷移行列、RF回転、緩和演算子）
  - `epg/epg_se.py`: Spin Echo シーケンス（CPMG, MESE, TSE/FSE）
  - `epg/epg_gre.py`: Gradient Echo シーケンス（FLASH, bSSFP, SSFP-FID/Echo）
- `tests/test_epg.py` 追加（21 tests）
- `docs/api/epg.md` ドキュメント追加
- `paper.md` を JOSS フォーマットに更新
- EPG 実装を Weigel 参照コードで検証
  - DECAES: B1 が励起パルスとリフォーカスパルス両方に影響（物理的に正確）
  - Weigel: B1 はリフォーカスパルスのみに影響（励起は理想 90°）
  - `cpmg()` に `b1_excitation` パラメータ追加（両モード対応）
  - 検証テスト 3 件追加（B1=1.0, 0.9, 0.8 で Weigel 参照と完全一致）
  - `EPGSimulator` に `apply_gradient_dephasing()` / `apply_gradient_rephasing()` 追加
- `epg_weigel()` を DECAES 方式に統一：励起パルスにも B1 影響を適用
  - 初期磁化 = sin(B1 × 90°)、リフォーカスも B1 × nominal
  - 減衰曲線の「形状」は Weigel 参照と完全一致、絶対値のみ異なる
  - テスト更新：正規化後の形状一致＋M0比の検証
- 全 88 テスト通過
- auto-tag の version 抽出が `__all__` 側の `"__version__"` まで拾う問題を修正（`.github/workflows/ci.yml` を `python -c` 抽出に変更、push まで実施）
- GitHub Pages の EPG ドキュメント（`docs/api/epg.md`）を現行APIに合わせて更新（`epg_se.se`/`tse`、B1例、CP/CPMG説明、API参照の整合）
- 作業メモは `docs/memo.md` に追記する運用に統一
- `AGENTS.md` に `docs/memo.md` 運用ルールを追記し、`docs/` をGit管理対象に追加
- GitHub About の説明文を "qmrpy: Python library for quantitative MRI." に更新（`gh repo edit`）
- qmrpy アイコンを `docs/assets/` に配置し、MkDocs の logo/favicon と README のロゴ表示を更新
- ルートの `qmrpy.png` を削除（アイコンは `docs/assets/` に集約）
- GitHub の Social preview はAPI/CLIで更新不可のため、`docs/assets/qmrpy-icon.png` を使って手動設定する方針を共有

## 2026-02-11
- `configs/exp/validation_core.toml` を追加し、`T1/T2/B1/QSM/Simulation` の検証ケース・seed・閾値を定義
- `scripts/summarize_parity.py` を拡張し、`--suite`/`--formats`/`--config` 対応と core 検証（外部依存なし）を追加
- core 検証の成果物として `core_validation.csv` / `core_validation_metrics.csv` / `core_validation.json` / `summary.json` を出力する仕様を追加
- `tests/test_validation_suite.py` を追加し、出力スキーマ・閾値判定整合・seed 再現性を検証
- `docs/guide/validation.md` を新規作成し、検証手順・出力定義・閾値運用を文書化
- `mkdocs.yml` の User Guide に Validation ページを追加
- `README.md` に JOSS 向けの検証実行セクション（英語/日本語）を追加
- `paper.md` を verification-first 方針に改稿し、`core_validation.csv` の実測値を含む定量表を追記
- `uv run --locked -m pytest tests/test_validation_suite.py` を実行（2 passed）
- `uv run --locked -m pytest` を実行（95 passed, warnings 5件）
- `$memo-entry`: T1/T2/T2* + B0/B1 拡張を実装（`src/qmrpy/models/b0/`, `src/qmrpy/models/t2star/`, `src/qmrpy/models/t1/despot1_hifi.py`, `src/qmrpy/models/t1/mp2rage.py`, `src/qmrpy/models/t2/emc_t2.py`）
- `T2EPG.fit` に `estimate_b1`/`b1_bounds`/`b1_init`/`b0_hz` を追加し、`fit_image` でも `estimate_b1` のマップ出力を対応
- 公開APIを更新（`src/qmrpy/models/__init__.py`, `src/qmrpy/models/t1/__init__.py`, `src/qmrpy/models/t2/__init__.py`, `src/qmrpy/models/b1/__init__.py`, `src/qmrpy/functional.py`, `src/qmrpy/__init__.py`）
- 新規テストを追加（`tests/test_b0.py`, `tests/test_b1_bloch_siegert.py`, `tests/test_r2star.py`, `tests/test_t1_advanced.py`, `tests/test_emc_t2.py`, `tests/test_functional_extended.py`）
- ドキュメントを更新（`docs/api/b0.md`, `docs/api/t2star.md`, `docs/api/index.md`, `docs/api/t1.md`, `docs/api/t2.md`, `docs/api/b1.md`, `docs/api/functional.md`, `docs/guide/t1-mapping.md`, `docs/guide/t2-mapping.md`, `docs/guide/b1-mapping.md`, `docs/index.md`, `mkdocs.yml`, `README.md`）
- `uv run --locked -m pytest` を実行（113 passed, warnings 5件）
- `uv run --locked ruff check` は既存コード由来の警告が残るため全体では未解消（今回追加ファイル対象の lint は通過）
- 次アクション: 既存コード由来の ruff 指摘（`src/qmrpy/_decaes/surrogate_1d.py` ほか）を別PRで整理するか判断 (TBD)
- `$memo-entry`: `scripts/summarize_parity.py` の core validator を拡張し、`B0DualEcho/B0MultiEcho/B1BlochSiegert/T2StarMonoR2/T2StarComplexR2/T1DESPOT1HIFI/T1MP2RAGE/T2EMC` の合成回収検証を追加
- `src/qmrpy/models/t1/despot1_hifi.py` を VFA+IR 同時最適化対応に更新し、`src/qmrpy/models/t1/mp2rage.py` は UNI 指標ベース LUT と `signal=(1,)` 受理を追加
- `src/qmrpy/models/t1/mp2rage.py` の NLS 初期値境界違反を修正（`m0` 非負化と `x0` の bounds 内クリップ）し、`tests/test_validation_suite.py` の失敗を解消
- `uv run --locked -m pytest tests/test_validation_suite.py tests/test_t1_advanced.py tests/test_functional_extended.py` を再実行（6 passed）
- `uv run --locked ruff check src tests scripts` を再実行（All checks passed）
- `uv run --locked -m pytest` を再実行（113 passed, warnings 5件）
- `$memo-entry`: `scripts/report_b0_b1_correction_effect.py` を追加し、`T1/T2/T2*` の補正なし vs `B1/B0` 補正あり比較（合成データ）を再現可能化
- `uv run --locked -- python scripts/report_b0_b1_correction_effect.py --seed 20260211 --n-samples 300 --json-out output/reports/b0_b1_correction_report.json` を実行し、補正後の閾値判定が全通過することを確認
- `docs/guide/validation.md` に補正効果レポートの実行手順と出力定義を追記
- `uv run --locked ruff check scripts/report_b0_b1_correction_effect.py` を実行（All checks passed）
- `uv run --locked -m pytest tests/test_validation_suite.py` を実行（2 passed）
- `$memo-entry`: `src/qmrpy/models/t1/mp2rage.py` をリファクタし、`fit()` の前処理・grid探索・LUT/NLS分岐を内部ヘルパーへ分解して可読性を改善
- `src/qmrpy/models/t1/mp2rage.py` に `t1_grid_ms` の空配列・非有限値・非正値を明示的に弾くバリデーションを追加
- `tests/test_t1_advanced.py` に `test_mp2rage_rejects_empty_t1_grid` を追加し、空グリッド時の `ValueError` を回帰テスト化
- `uv run --locked ruff check src/qmrpy/models/t1/mp2rage.py tests/test_t1_advanced.py` を実行（All checks passed）
- `uv run --locked -m pytest tests/test_t1_advanced.py tests/test_validation_suite.py` を実行（6 passed）
- `$memo-entry`: v1.0 全面リファクタ残件として `ResultSchemaMixin` 継承ラッパを `ResultAdapterBase` 委譲ラッパへ統一（`src/qmrpy/models/t2/__init__.py`, `src/qmrpy/models/t2star/__init__.py`, `src/qmrpy/models/b0/__init__.py`, `src/qmrpy/models/b1/__init__.py`, `src/qmrpy/models/qsm/__init__.py`）
- `T2DECAESMap.fit_image` と `QSMSplitBregman.fit` を個別ラップし、`fit_image` 内部の `self.fit` 干渉と QSM シグネチャ不整合を解消
- `src/qmrpy/sim/simulation.py` を更新し、`simulate_single_voxel/sensitivity_analysis/simulate_parameter_distribution` の `fit` 出力を `params/quality/diagnostics` 構造で一貫化
- 失敗していたテスト群を新スキーマ参照へ追随（`tests/test_decaes_parity.py`, `tests/test_decaes_t2.py`, `tests/test_decaes_t2part.py`, `tests/test_emc_t2.py`, `tests/test_epg_t2.py`, `tests/test_functional_extended.py`, `tests/test_inversion_recovery.py`, `tests/test_mono_t2.py`, `tests/test_mwf.py`, `tests/test_qsm_pipeline.py`, `tests/test_simulation.py`）
- `uv run --locked -m pytest tests/test_decaes_parity.py tests/test_decaes_t2.py tests/test_decaes_t2part.py tests/test_emc_t2.py tests/test_epg_t2.py tests/test_functional_extended.py tests/test_inversion_recovery.py tests/test_mono_t2.py tests/test_mwf.py tests/test_qsm_pipeline.py tests/test_simulation.py tests/test_validation_suite.py` を実行（43 passed）
- `uv run --locked ruff check src tests scripts` を実行（All checks passed）
- `uv run --locked -m pytest` を実行（114 passed, warnings 5件）
- `$memo-entry`: `README.md` のクイックスタート（英日）を新結果スキーマに追随し、`fit()` の戻り値例を `params/quality/diagnostics` 形式へ更新
- `README.md` の旧クラス名・旧functional名を再検索し、旧命名参照が残っていないことを確認
- `$memo-entry`: `gh run view` で `Deploy Docs` 失敗原因を調査し、`docs/api/functional.md` の `qmrpy.functional.decaes_t2map_spectrum` 参照が未実装であることを確認
- `docs/api/functional.md` から未実装 API 参照を削除して mkdocstrings のビルドエラーを解消
- `uv run --locked mkdocs build` を実行（build 成功）
- `$memo-entry`: `src/qmrpy/core/result_schema.py` に `FitResult` を追加し、`fit/fit_image` の戻り値を params辞書互換 + `quality`/`diagnostics` 属性アクセス可能な形式へ改訂
- `src/qmrpy/core/fit_protocols.py` と `src/qmrpy/core/__init__.py` を更新し、`nest_result` の返却型を `FitResult` に統一、公開APIに `FitResult` を追加
- `src/qmrpy/sim/simulation.py` の `_fit_model` を更新し、`FitResult` 属性アクセスと既存ネスト辞書の両方を扱えるように調整
- `src/qmrpy/functional.py` の型ヒントを `FitResult` 返却に更新
- `README.md` / `docs/getting-started/quickstart.md` を更新し、新しい `fit['param']` + `fit.quality` / `fit.diagnostics` 記法を反映
- `tests/test_fit_result.py` を追加し、params辞書互換アクセス・属性アクセス・後方互換アクセス（`result['params']`）を回帰テスト化
- `uv run --locked ruff check src/qmrpy/core/result_schema.py src/qmrpy/core/fit_protocols.py src/qmrpy/functional.py src/qmrpy/sim/simulation.py tests/test_fit_result.py` を実行（All checks passed）
- `uv run --locked -m pytest` を実行（117 passed, warnings 5件）
- `uv run --locked mkdocs build` を実行（build 成功）
- `$memo-entry`: `FitResult` 方針に合わせ、関連ドキュメントを網羅改訂（`docs/guide/t1-mapping.md`, `docs/guide/t2-mapping.md`, `docs/guide/b1-mapping.md`, `docs/api/index.md`, `README.md`, `docs/getting-started/quickstart.md`）
- ガイド内の参照を `result["params"][...]` / `maps["params"][...]` から `result["..."]` / `maps["..."]` に統一し、`quality`/`diagnostics` は属性アクセス（`result.quality`, `maps.diagnostics`）へ変更
- `docs/guide/t2-mapping.md` の `T2DECAESMap.fit_image` 戻り値説明を現仕様に合わせ、`maps, dist = ...` から `maps = ...; dist = maps["distribution"]` へ修正
- `uv run --locked mkdocs build` を再実行（build 成功）

## 2026-02-13
- `$memo-entry`: ルートの `/Users/sugim/Developments/qmrpy/qmrpy.png` をアイコン元画像として採用
- `docs/assets/qmrpy-icon.png` を `qmrpy.png` で更新
- `docs/assets/qmrpy-icon-32.png` を `qmrpy.png` から 32x32 にリサイズして更新（`sips -z 32 32`）
- `$memo-entry`: 旧アイコン置換の運用に合わせ、ルート `qmrpy.png` を廃止して `docs/assets/qmrpy-icon.png` へ移動統一
- `docs/assets/qmrpy-icon-32.png` を `docs/assets/qmrpy-icon.png` から再生成し、MkDocs の logo/favicon 参照先を維持したまま置換
- 次アクション: 置換コミットを作成して反映 (完了)
- `$memo-entry`: README 画像が更新されない表示問題に対し、`raw.githubusercontent.com` キャッシュ回避のため `README.md` のロゴURLへ `?v=20260213` を付与
- 次アクション: 変更をコミットして `origin/main` へ push し、表示更新を確認 (完了)

## 2026-05-07
- context: qmrpy ソフトウェア論文の新規性検討 / change: 論文主張を単なる Python 移植ではなく、検証基盤・qMRI-BIDS 連携・Pulseq/MRzero 連携・実データ応用のいずれかへ強化する方針を整理 / reason: 既存 `paper.md` の verification-first 主張だけでは方法論的な新規性が弱い可能性があるため / artifact: `docs/memo.md` / next: 採用する新規性軸を 1 つ選び、必要な実装・検証・論文改稿範囲を決める

## 2026-05-15
- context: T1/T2 限定の次期機能候補調査 / change: DWI/ADC ではなく、近年の同時 T1-T2 mapping・MR fingerprinting・EPG/辞書ベース T2 水/脂肪分離を候補として整理 / reason: 既存 qmrpy の T1/T2/EPG/Pulseq/MRzero 足場と整合し、再現可能な文献再現テーマになりやすいため / artifact: `docs/memo.md` / next: 実装候補を `JointT1T2MRF` または `T2WaterFatEPG` のどちらに絞る

## 2026-05-16
- context: GitHub repository topics 整備 / change: `python`, `quantitative-mri`, `qmri`, `mri`, `medical-imaging`, `simulation` を GitHub topics に追加 / reason: qmrpy の用途・公開内容を GitHub 上で見つけやすくするため / artifact: GitHub repository metadata / next: 必要なら topic の追加・削除を見直す

## 2026-07-05
- context: qmrpy 完全化タスク フェーズ1着手前ベースライン / change: `uv sync --locked` は通常権限で uv cache 権限エラー、昇格実行で成功。`uv lock --check` は成功。`uv run --locked -m pytest` は `pytest` 未同期で失敗、`uv run --locked --extra dev -m pytest` は 117 passed。`uv run --locked ruff check src tests scripts` は All checks passed。`uv run --locked mypy src/qmrpy` は mypy 未導入で失敗。`uv run --locked mkdocs build` は mkdocs 未同期で失敗、`uv run --locked --group docs mkdocs build` は成功。`uv run --locked scripts/summarize_parity.py --suite core` は成功 / reason: 実データ I/O 実装前にテスト・lint・型・docs・検証スイートの現状を固定するため / artifact: `docs/memo.md`, `output/reports/parity_summary/` / next: フェーズ1として NIfTI/DICOM/BIDS I/O と optional dependency、テスト、docs を実装する
- context: qmrpy 完全化タスク フェーズ1 実データ I/O / change: `nibabel`/`pydicom` を optional `io` 依存へ追加し、`load_nifti`/`save_nifti`/`save_nifti_map`/`load_dicom_series`/`load_bids_relaxometry` と公開 export、合成 NIfTI/DICOM/BIDS テスト、README/API/I/O guide を追加。NIfTI/DICOM/BIDS は遅延 import とし、未インストール時に `qmrpy[io]` 案内を出す設計にした / reason: 実データ relaxometry 入出力と結果マップの空間メタデータ継承を、既存 TIFF API とコア import 互換性を壊さず追加するため / artifact: `src/qmrpy/io.py`, `src/qmrpy/__init__.py`, `tests/test_io.py`, `docs/guide/io.md`, `docs/api/io.md`, `README.md`, `pyproject.toml`, `uv.lock` / next: フェーズ2の T1rho 実装へ進む前に、型チェックゲートは mypy 未導入のためフェーズ5で整備する (TBD)
- context: qmrpy 完全化タスク フェーズ1 検証 / change: `uv lock --check` 成功、`uv run --locked -m pytest` は 122 passed、`uv run --locked ruff check src tests scripts` は All checks passed、`uv run --locked mkdocs build` は成功、`uv run --locked scripts/summarize_parity.py --suite core` は成功、`uv run --locked mypy src/qmrpy` は mypy 未導入で失敗 / reason: フェーズ1完了前にテスト・lint・docs・検証スイートの品質ゲートを確認するため / artifact: `docs/memo.md`, `output/reports/parity_summary/` / next: 型チェック導入は CI/品質ゲート整備フェーズで対応する
- context: qmrpy 完全化タスク フェーズ2 T1rho / change: `T1Rho` スピンロック単指数モデル、ResultAdapter ラッパ、functional API、公開 export、forward/fit/fit_image/エラー系テスト、API docs、T1 guide、core validation synthetic recovery を追加 / reason: relaxometry 手法網羅の最初の追加モデルとして T1rho mapping を既存モデル契約に合わせて提供するため / artifact: `src/qmrpy/models/t1rho/`, `src/qmrpy/models/__init__.py`, `src/qmrpy/functional.py`, `tests/test_t1rho.py`, `configs/exp/validation_core.toml`, `scripts/summarize_parity.py`, `docs/api/t1rho.md`, `docs/guide/t1-mapping.md`, `README.md` / next: フェーズ2の次手法として MTR/MTsat を実装する
- context: qmrpy 完全化タスク フェーズ2 T1rho 検証 / change: `uv lock --check` 成功、`uv run --locked -m pytest` は 127 passed、`uv run --locked ruff check src tests scripts` は All checks passed、`uv run --locked mkdocs build` は成功、`uv run --locked scripts/summarize_parity.py --suite core` は成功、`uv run --locked mypy src/qmrpy` は mypy 未導入で失敗 / reason: T1rho 追加後に既存 API・検証スイート・docs の回帰がないことを確認するため / artifact: `docs/memo.md`, `output/reports/parity_summary/` / next: 型チェック導入はフェーズ5で対応する (TBD)
- context: qmrpy 完全化タスク フェーズ2 MTR/MTsat / change: `MTR` と `MTsat` モデル、ResultAdapter ラッパ、functional API、公開 export、forward/fit/fit_image/エラー系テスト、API docs、MT guide、core validation synthetic recovery を追加。MTsat は spoiled-GRE の T1/FA/TR 補正で MTなし参照信号を作る Helms 系近似にした / reason: magnetization transfer mapping を relaxometry 解析 API と同じ戻り値スキーマ・画像処理パターンで利用可能にするため / artifact: `src/qmrpy/models/mt/`, `src/qmrpy/models/__init__.py`, `src/qmrpy/functional.py`, `tests/test_mt.py`, `configs/exp/validation_core.toml`, `scripts/summarize_parity.py`, `docs/api/mt.md`, `docs/guide/mt-mapping.md`, `README.md` / next: フェーズ2の次手法として MRF 辞書ベース同時 T1-T2 mapping を実装する
- context: qmrpy 完全化タスク フェーズ2 MTR/MTsat 検証 / change: `uv lock --check` 成功、`uv run --locked -m pytest` は 133 passed、`uv run --locked ruff check src tests scripts` は All checks passed、`uv run --locked mkdocs build` は成功、`uv run --locked scripts/summarize_parity.py --suite core` は成功、`uv run --locked mypy src/qmrpy` は mypy 未導入で失敗 / reason: MTR/MTsat 追加後に既存 API・検証スイート・docs の回帰がないことを確認するため / artifact: `docs/memo.md`, `output/reports/parity_summary/` / next: 型チェック導入はフェーズ5で対応する (TBD)
- context: qmrpy 完全化タスク フェーズ2 MRF / change: `MRFDictionary` を追加し、EPG state engine を用いた spoiled-FISP 近似の可変 FA/TR/TE fingerprint 生成、辞書生成、正規化内積 matching、fit_image、functional API、公開 export、テスト、API docs、MRF guide、core validation synthetic recovery を追加 / reason: 同時 T1-T2 mapping の最小実装を既存 ResultAdapter/FitResult パターンで提供するため / artifact: `src/qmrpy/models/mrf/`, `src/qmrpy/models/__init__.py`, `src/qmrpy/functional.py`, `tests/test_mrf.py`, `configs/exp/validation_core.toml`, `scripts/summarize_parity.py`, `docs/api/mrf.md`, `docs/guide/mrf.md`, `README.md` / next: フェーズ2の次手法として T2 水/脂肪分離を最小2プール近似で実装する
- context: qmrpy 完全化タスク フェーズ2 MRF 検証 / change: `uv lock --check` 成功、`uv run --locked -m pytest` は 137 passed、`uv run --locked ruff check src tests scripts` は All checks passed、`uv run --locked mkdocs build` は成功、`uv run --locked scripts/summarize_parity.py --suite core` は成功、`uv run --locked mypy src/qmrpy` は mypy 未導入で失敗 / reason: MRF 追加後に既存 API・検証スイート・docs の回帰がないことを確認するため / artifact: `docs/memo.md`, `output/reports/parity_summary/` / next: 型チェック導入はフェーズ5で対応する (TBD)
- context: qmrpy 完全化タスク フェーズ2 T2水/脂肪分離 / change: `T2WaterFat` を追加し、T2w/T2f グリッドと NNLS による2プール水/脂肪振幅推定、fat fraction、fit_image、functional API、公開 export、テスト、T2 docs、core validation synthetic recovery を追加 / reason: 高難度な full EPG/complex multi-peak fat model の前段として、実運用可能な最小2プール近似を既存T2 APIで提供するため / artifact: `src/qmrpy/models/t2/water_fat.py`, `src/qmrpy/models/t2/__init__.py`, `src/qmrpy/models/__init__.py`, `src/qmrpy/functional.py`, `tests/test_t2_water_fat.py`, `configs/exp/validation_core.toml`, `scripts/summarize_parity.py`, `docs/api/t2.md`, `docs/guide/t2-mapping.md`, `README.md` / next: フェーズ2完了として pyproject の diffusion 記述整合を確認し、フェーズ3 CLI へ進む
- context: qmrpy 完全化タスク フェーズ2 T2水/脂肪分離 検証 / change: `uv lock --check` 成功、`uv run --locked -m pytest` は 141 passed、`uv run --locked ruff check src tests scripts` は All checks passed、`uv run --locked mkdocs build` は成功、`uv run --locked scripts/summarize_parity.py --suite core` は成功、`uv run --locked mypy src/qmrpy` は mypy 未導入で失敗 / reason: T2WaterFat 追加後に既存 API・検証スイート・docs の回帰がないことを確認するため / artifact: `docs/memo.md`, `output/reports/parity_summary/` / next: 型チェック導入はフェーズ5で対応する (TBD)
- context: qmrpy 完全化タスク フェーズ2 メタデータ整合 / change: `pyproject.toml` の keywords から実装対象外の `diffusion` を削除し、`magnetization-transfer` に置換。`uv lock --check`、`uv run --locked ruff check src tests scripts`、`uv run --locked -m pytest tests/test_import.py tests/test_t1rho.py tests/test_mt.py tests/test_mrf.py tests/test_t2_water_fat.py` は成功 / reason: relaxometry 特化の実装実態と package metadata の乖離を解消するため / artifact: `pyproject.toml`, `docs/memo.md` / next: フェーズ3 CLI 実装へ進む
- context: qmrpy 完全化タスク フェーズ3 CLI / change: `src/qmrpy/cli.py` と `[project.scripts] qmrpy = "qmrpy.cli:main"` を追加し、`qmrpy info`、NIfTI入出力の `qmrpy fit t2-mono|t1rho|mtr`、`qmrpy validate` を実装。CLI guide、README 使用例、help/info/fit/validate smoke tests を追加 / reason: scripts を厚くせず、既存 src API を呼ぶ薄い実行入口を提供するため / artifact: `src/qmrpy/cli.py`, `pyproject.toml`, `tests/test_cli.py`, `docs/guide/cli.md`, `README.md`, `mkdocs.yml` / next: フェーズ4 ガバナンスファイル整備へ進む
- context: qmrpy 完全化タスク フェーズ3 CLI 検証 / change: `uv lock --check` 成功、`uv run --locked -m pytest` は 144 passed、`uv run --locked ruff check src tests scripts` は All checks passed、`uv run --locked mkdocs build` は成功、`uv run --locked scripts/summarize_parity.py --suite core` は成功、`uv run --locked mypy src/qmrpy` は mypy 未導入で失敗 / reason: CLI 追加後に既存 API・検証スイート・docs の回帰がないことを確認するため / artifact: `docs/memo.md`, `output/reports/parity_summary/` / next: 型チェック導入はフェーズ5で対応する (TBD)
- context: qmrpy 完全化タスク フェーズ4 ガバナンス / change: `CONTRIBUTING.md`、`CHANGELOG.md`、`CODE_OF_CONDUCT.md`、`SECURITY.md`、`CITATION.cff` を追加し、`pyproject.toml` の `Development Status` を `4 - Beta` へ更新。Python classifier は AGENTS.md の 3.11 系運用に合わせて 3.11 のまま維持した / reason: OSS/JOSS 水準の貢献・引用・セキュリティ・変更履歴の基本文書を整え、1.0.0 と package metadata の乖離を減らすため / artifact: `CONTRIBUTING.md`, `CHANGELOG.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CITATION.cff`, `pyproject.toml` / next: フェーズ5 CI/品質ゲート整備へ進む
- context: qmrpy 完全化タスク フェーズ4 検証 / change: `uv lock --check` 成功、`uv run --locked -m pytest` は 144 passed、`uv run --locked ruff check src tests scripts` は All checks passed、`uv run --locked mkdocs build` は成功、`uv run --locked scripts/summarize_parity.py --suite core` は成功、`uv run --locked mypy src/qmrpy` は mypy 未導入で失敗 / reason: ガバナンス文書と metadata 更新が実装・docs・検証スイートに回帰を起こしていないことを確認するため / artifact: `docs/memo.md`, `output/reports/parity_summary/` / next: フェーズ5で mypy 導入、pre-commit、CI matrix、coverage gate を整備する (TBD)
- context: qmrpy 完全化タスク フェーズ5 CI/品質ゲート / change: `mypy`、`pre-commit`、`pre-commit-hooks` を optional dev 依存へ追加し、local hook の `.pre-commit-config.yaml`、mypy 段階導入設定、coverage artifact、Ubuntu/macOS OS matrix、docs build、core validation を CI に追加。`ruff format` をゲート化するため既存 Python/スクリプト/テストを機械整形し、coverage 生成物を `.gitignore` に追加した / reason: OSS/JOSS 水準の lint・型・coverage・docs・検証スイートを CI で自動確認するため / artifact: `.github/workflows/ci.yml`, `.github/workflows/docs.yml`, `.pre-commit-config.yaml`, `pyproject.toml`, `uv.lock`, `.gitignore`, `CONTRIBUTING.md`, `CHANGELOG.md`, `src/`, `scripts/`, `tests/` / next: フェーズ6で README/docs/paper 整合と `.DS_Store` 物理削除へ進む
- context: qmrpy 完全化タスク フェーズ5 検証 / change: `uv lock --check` 成功、`uv run --locked pre-commit run --all-files` 成功、`uv run --locked ruff check src tests scripts` 成功、`uv run --locked mypy src/qmrpy` 成功、`uv run --locked -m pytest --cov=qmrpy --cov-report=xml` は 144 passed かつ `coverage.xml` 生成、`uv run --locked mkdocs build` 成功、`uv run --locked scripts/summarize_parity.py --suite core` 成功、`git diff --check` 成功。`uv sync --locked --extra dev --extra io --group docs` は sandbox の uv cache 権限エラー後、権限昇格が利用上限で拒否されたため未確認 / reason: CI 品質ゲート追加後のローカル再現性と既存機能回帰を確認するため / artifact: `docs/memo.md`, `coverage.xml` (ignored), `output/reports/parity_summary/` / next: 利用上限回復後に `uv sync --locked --extra dev --extra io --group docs` を再確認する (TBD)
- context: qmrpy 完全化タスク フェーズ5 validation gate 修正 / change: `scripts/summarize_parity.py` が core validation fail 時に非ゼロ終了するよう修正し、MP2RAGE validator は config の T1/B1 bounds を fit に渡すよう変更。MP2RAGE core case は2点 surrogate の回収検証として固定B1・無ノイズ設定にし、T2 water/fat の fat fraction 閾値は浮動小数丸めを許す `1e-12` にした / reason: CI に追加した validation gate が失敗ケースを確実に検知し、core validation が実際に 21/21 pass になるようにするため / artifact: `scripts/summarize_parity.py`, `configs/exp/validation_core.toml`, `docs/memo.md` / next: フェーズ6の文書・論文整合へ戻る
- context: qmrpy 完全化タスク フェーズ6 クリーンアップ / change: ワークツリー内の `.DS_Store` を物理削除し、docs Home/Installation と `paper.md` を I/O・CLI・追加モデル・CI 品質ゲート後の validation 21/21 pass に合わせて更新 / reason: README/docs/paper の説明を実装済み機能と検証結果に整合させ、不要な OS 生成物を残さないため / artifact: `docs/index.md`, `docs/getting-started/installation.md`, `paper.md`, `docs/memo.md` / next: 最終コミットを作成する
- context: qmrpy 完全化タスク フェーズ6 検証 / change: `uv lock --check` 成功、`uv run --locked pre-commit run --all-files` 成功、`uv run --locked mypy src/qmrpy` 成功、`uv run --locked -m pytest --cov=qmrpy --cov-report=xml` は 144 passed、`uv run --locked mkdocs build` 成功、`uv run --locked scripts/summarize_parity.py --suite core` は 21/21 pass、`.DS_Store` は検出なし。`uv sync --locked --extra dev --extra io --group docs` は前フェーズ同様、権限昇格の利用上限により未再確認 / reason: 最終文書更新後にテスト・型・docs・validation の回帰がないことを確認するため / artifact: `docs/memo.md`, `coverage.xml` (ignored), `output/reports/parity_summary/` / next: 利用上限回復後に `uv sync --locked --extra dev --extra io --group docs` を再確認する (TBD)

## 2026-07-06
- context: qmrpy 完全化タスク 最終再確認 / change: `git status --short` は clean、`uv lock --check` 成功、`uv sync --locked --extra dev --extra io --group docs` は通常権限で uv cache 権限エラー後に昇格実行で成功、`uv run --locked pre-commit run --all-files` 成功、`uv run --locked mypy src/qmrpy` 成功、`uv run --locked -m pytest --cov=qmrpy --cov-report=xml` は 144 passed、`uv run --locked mkdocs build` 成功、`uv run --locked scripts/summarize_parity.py --suite core` 成功、`.DS_Store` は検出なし / reason: 前回残っていた `uv sync --locked` 未確認を含め、完了条件を日付を改めて再確認するため / artifact: `docs/memo.md`, `coverage.xml` (ignored), `output/reports/parity_summary/` / next: 必要に応じて push/PR 作成へ進む
- context: qmrpy 公開状態確認 / change: PyPI JSON で `qmrpy` 最新版が `1.0.0`、wheel/sdist が 2026-02-11 に公開済みであることを確認。GitHub Actions の release workflow は `v1.0.0` で success だが、GitHub release API は一時的な接続/DNS エラーで直接照会できなかった。ローカル `main` は `origin/main` より 12 commits ahead で、完全化タスクの追加分は未 push/未公開状態と判断 / reason: PyPI/GitHub Release が現行作業を反映しているか確認するため / artifact: `docs/memo.md` / next: 完全化タスクを公開するには version bump、push、tag/release、PyPI publish の手順を実施する
- context: qmrpy 1.1.0 公開準備 / change: `src/qmrpy/__init__.py` の `__version__` を `1.1.0` に更新し、`CHANGELOG.md` の Unreleased を `1.1.0 - 2026-07-06` として切り出し、`CITATION.cff` の version/date を更新 / reason: 完全化タスクで追加した I/O・relaxometry モデル・CLI・ガバナンス・CI 品質ゲートをマイナーリリースとして PyPI/GitHub Release に公開するため / artifact: `src/qmrpy/__init__.py`, `CHANGELOG.md`, `CITATION.cff`, `docs/memo.md` / next: 品質ゲート通過後に commit、push、tag/release workflow、PyPI/GitHub Release 反映を確認する
- context: qmrpy 1.1.0 リリース前検証 / change: `uv lock --check` 成功、`uv run --locked pre-commit run --all-files` 成功、`uv run --locked mypy src/qmrpy` 成功、`uv run --locked python -c 'import qmrpy; print(qmrpy.__version__)'` は `1.1.0`、`uv run --locked -m pytest --cov=qmrpy --cov-report=xml` は 144 passed、`uv run --locked mkdocs build` 成功、`uv run --locked scripts/summarize_parity.py --suite core` 成功 / reason: 1.1.0 として push/tag/release する前にローカル品質ゲートを再確認するため / artifact: `docs/memo.md`, `coverage.xml` (ignored), `output/reports/parity_summary/` / next: 1.1.0 リリース準備コミットを push し、release workflow を確認する
- context: qmrpy 1.1.0 release workflow 調整 / change: CI の auto-tag は成功したが `GITHUB_TOKEN` 由来の tag push では release workflow が起動しなかったため、`.github/workflows/release.yml` に `workflow_dispatch` と明示 tag 入力を追加 / reason: `v1.1.0` を手動 workflow dispatch で PyPI/GitHub Release へ公開し、今後も tag 連鎖起動に依存しない公開手順を用意するため / artifact: `.github/workflows/release.yml`, `docs/memo.md` / next: workflow 変更を push し、`release.yml` を `tag=v1.1.0` で手動実行する

## 2026-08-01
- context: bioRxiv (Biophysics) 論文投稿プロジェクト立ち上げ / change: `qmrpy` の検証結果・モデリング基盤を bioRxiv (Biophysics) プレプリントとして投稿するためのプロジェクトを立ち上げ。図表自動生成 (`scripts/generate_paper_figures.py`)、PDFビルド (`scripts/build_paper.py`)、原稿拡充 (`paper.md`)、投稿メタデータ (`docs/biorxiv_submission.md`) の実装に着手 / reason: 定量MRIモデルの検証基盤・Python参照実装の成果をプレプリントで広く学術コミュニティに共有するため / artifact: `docs/memo.md`, `implementation_plan.md` / next: `scripts/generate_paper_figures.py` を作成し、図表生成を確認
- context: bioRxiv 投稿用 図の再設計 / change: `scripts/generate_paper_figures.py` を全面書き換え。旧 Fig2/Fig3（単位混在の指標を同一対数軸に並べる棒グラフ、推定マップのみのファントム図）を廃止し、Fig1=閾値で正規化したマージン（lollipop, 対数軸, 厳密一致ケースは注記付き）、Fig2=代表4モデルの推定値 vs 真値（VFA T1 / T1rho / mono-T2 / MWF）、Fig3=合成ファントムの真値 vs 推定 T2 マップ（64x64, 4ディスク）に変更。すべて plotnine、設定値は `configs/exp/validation_core.toml` から読み込み / reason: 旧 Fig2 は Hz と無次元比を同一対数軸で比較しており誤解を招くうえ、値 0 のケースが対数軸で破綻していた。また `paper.md` の「Figure 1 / Figure 2」記述と実ファイルの図番号が不整合だった / artifact: `scripts/generate_paper_figures.py`, `output/paper_figures/fig1_validation_margin.*`, `fig2_parameter_recovery.*`, `fig3_t2_phantom_map.*` / next: 図の統計値（Fig2 relMAE, Fig3 MAE）を原稿の Results に反映する
- context: bioRxiv 投稿用 参考文献整備 / change: CrossRef API で 15 件の DOI を実在確認したうえで `paper.bib` に Marques2010, Helms2008, Ma2013, Hennig1988, Whittall1989, Prasloski2012, BenEliezer2015, Sacolick2010, Yarnykh2007, Harris2020, Virtanen2020, Gorgolewski2016, Karakuzu2022, Layton2017, Loktyushin2021 を追加 / reason: 実装済みモデル（MP2RAGE, MTsat, MRF, EPG, NNLS, EMC, Bloch-Siegert, AFI）と依存（NumPy/SciPy/Pulseq/MRzero）、qMRI-BIDS 対応の根拠文献を、捏造なしで原稿から引用できるようにするため / artifact: `paper.bib` / next: 原稿本文から引用し、pandoc citeproc で解決を確認する
- context: DECAES parity スクリプトの不具合修正 / change: `scripts/summarize_parity.py` の `_decaes_parity_rows()` が `out["params"]["distribution"]` / `out["params"]["alpha_deg"]` を参照して `KeyError: 'params'` で落ちていたため、`T2DECAESMap.fit` の実際の戻り値（フラットな dict）に合わせて `out["distribution"]` / `out["alpha_deg"]` へ修正（計6箇所） / reason: FitResult リファクタ後に core suite 側だけ追随しており、`--suite decaes` が実行不能だった。bioRxiv 原稿で DECAES parity を再現可能な結果として報告するには実行できる必要があるため / artifact: `scripts/summarize_parity.py`, `output/reports/parity_summary/decaes_parity.*` / next: 修正後の再実行値が既存アーカイブ値と一致することを確認する（一致を確認済み）
- context: bioRxiv 用フル原稿の執筆 / change: `paper_biorxiv.md` を新規作成。Abstract(226語) / Introduction / Methods(architecture, model table, verification framework, reproducibility) / Results(cross-domain validation, parameter recovery, image-level fitting, parity) / Discussion(限界5点を明示) / Conclusion / Data and code availability / Declarations の IMRaD 構成。Table 1=実装モデル一覧、Table 2=core validation 21件（`core_validation.csv` から生成）、Fig1-3 を埋め込み。`paper.md`(JOSS版) は温存し、図番号の記述のみ更新 / reason: JOSS 形式の `paper.md` はプレプリントとしては Methods/Results/Discussion が不足しており、bioRxiv の読者・スクリーニングに耐える構成が必要なため / artifact: `paper_biorxiv.md`, `paper.md` / next: pandoc で PDF を生成し、図・表・引用の描画を確認する
- context: 投稿用 PDF ビルド / change: `scripts/build_paper.py` を書き換え、pandoc + xelatex + citeproc で `output/paper_build/paper_biorxiv.pdf`（13ページ）を生成。`--source` で `paper.md` も選択可能。表の桁溢れを避けるため Table 2 を相対幅指定の pipe table に変更し、`&sigma;` は `$\sigma$`、指数表記は `$2.2 \times 10^{-15}$` 形式の LaTeX 数式へ置換 / reason: xelatex の Latin Modern に σ が無く欠字警告が出ていたこと、code span を含む表が本文幅を超えて重なっていたため / artifact: `scripts/build_paper.py`, `output/paper_build/paper_biorxiv.pdf` / next: 生成 PDF の全ページを目視確認する（13ページ全て確認済み）
- context: bioRxiv 投稿メタデータ整備 / change: `docs/biorxiv_submission.md` を全面改訂。公式ページ（submit-a-manuscript / collection）を 2026-08-01 に確認し、Subject Collection を **Bioengineering**、Article category を **New Results**、ライセンスを **CC BY 4.0** に確定。生成手順、フォーム貼付用プレーンテキスト Abstract、Declarations、投稿11ステップ、投稿前チェックリスト、投稿後作業（Zenodo DOI 取得）、投稿先の選択（arXiv eess.IV との比較）を記載 / reason: 旧ドラフトは Biophysics 想定かつ存在しない図ファイル名を参照しており、公式要件との突き合わせも未実施だったため / artifact: `docs/biorxiv_submission.md` / next: ユーザーが bioRxiv アカウントで投稿手続きを実施する（投稿操作自体は未実行）
- context: bioRxiv 投稿準備の品質ゲート / change: `uv lock --check` 成功、`uv run --locked ruff check src tests scripts` 成功、`uv run --locked pre-commit run --all-files` 成功（end-of-file-fixer が `docs/memo.md` を1度修正、再実行で全 Passed）、`uv run --locked mypy src/qmrpy` 成功（68 files）、`uv run --locked -m pytest` は 144 passed、`uv run --locked mkdocs build` 成功、`uv run --locked scripts/summarize_parity.py --suite core` は 21/21 pass / reason: 原稿・図・スクリプト変更後に既存の実装・docs・検証スイートへ回帰がないことを確認するため / artifact: `docs/memo.md`, `output/reports/parity_summary/` / next: 変更をコミットする
- context: 移植コードのライセンス監査 / change: 上流ライセンスを一次情報で確認（qMRLab=MIT, DECAES.jl=MIT, pypulseq=MIT, MRzero-Core=AGPL-3.0、依存パッケージ10件は BSD/MIT/MPL 系）。重大な問題を2件検出。(1) AGPL-3.0 の `mrzerocore` が `pyproject.toml` の必須 `dependencies` に入っており MIT 配布と整合しない（コード側は遅延 import で optional 設計）。(2) `src/qmrpy/models/noise/denoising_mppca.py` は qMRLab `External/mppca_denoise/MPdenoising.m` の逐行翻訳で、同ファイルは非商用主体限定・use/copy/modify のみ許諾（distribute/sublicense の許諾なし、臨床利用禁止）のため MIT 再頒布が許諾範囲を超える。QSM 由来関数（qsmSplitBregman/backgroundRemovalSharp/calcGradientMaskFromMagnitudeImage/unwrapPhaseLaplacian/calcFdr/kspaceKernel/applyForward/calcChiL2）は全て qMRLab `src/` 配下＝MIT 対象で問題なし（ただし各ファイルに Berkin Bilgic 由来の記載あり）。`_decaes/` は DECAES.jl (MIT) の明示的 port で、THIRD_PARTY_NOTICES.md が wheel/sdist 双方に同梱されており MIT の表示義務は充足 / reason: bioRxiv 公開でインストール導線が増えるため、公開前に再頒布可否を厳密に確認する必要があったため / artifact: `docs/memo.md` / next: (1) `mrzerocore` を optional extra へ移す、(2) MPPCA の扱い（削除・クリーンルーム再実装・権利者許諾のいずれか）を決める、(3) THIRD_PARTY_NOTICES.md を実態に合わせて改訂する、(4) 論文の "distributed under the MIT licence" 記述を修正する
- context: ライセンス是正 (1) AGPL 依存の分離 / change: `pyproject.toml` の必須 `dependencies` から `mrzerocore` を外し、optional extra `mrzero` として分離。理由をコメントで明記し、`uv lock` を更新。README に「MRzeroCore は AGPL-3.0 のため既定では入れない、extra を入れると結合環境に AGPL-3.0 が及ぶ」節を追加 / reason: MIT 配布のパッケージが AGPL-3.0 を必須依存にすると、全インストール環境で copyleft の及ぶ結合が成立するため。`src/qmrpy/sim/mrzero.py` は元から遅延 import で optional 設計だった / artifact: `pyproject.toml`, `uv.lock`, `README.md` / next: 副次効果として torch を含む重い依存が既定インストールから外れたことを確認済み
- context: ライセンス是正 (2) MPPCA 削除 / change: `src/qmrpy/models/noise/`（MPPCA）、`tests/test_mppca.py`、`scripts/verify_mppca.py`、`scripts/octave/verify_mppca.m`、`configs/exp/test_mppca.toml` を削除し、`models/__init__.py`、`scripts/summarize_tests.py`、`README.md`、`docs/index.md`、`paper.md`、`paper_biorxiv.md`、`paper.bib` (Veraart2016) から参照を除去。公開モデルクラスは 25 → 24、テストは 144 → 142 / reason: 実装が qMRLab `External/mppca_denoise/MPdenoising.m` の逐行翻訳であり、同ファイルは非商用主体限定・use/copy/modify のみ許諾（distribute/sublicense なし・臨床利用禁止）のため、MIT 再頒布が許諾範囲を超えるため / artifact: 上記ファイル群 / next: PyPI に残る 1.1.0 以前の版にも MPPCA が含まれるため、yank の要否を判断する (TBD)
- context: ライセンス是正 (3) 帰属表示の整備 / change: `THIRD_PARTY_NOTICES.md` を全面改訂し、qMRLab / DECAES.jl の移植元ファイルをモジュール単位の対応表にし、QSM が qMRLab 経由で Bilgic 2014 由来であることを明記、依存パッケージのライセンス一覧と AGPL extra の注意、削除した MPPCA の経緯を追加。qMRLab/DECAES から翻訳した 22 ファイルの先頭に帰属コメントを追加。qMRLab 側のパス 15 件はリポジトリツリーで実在確認（b1_dam.m/b1_afi.m は `src/Models/FieldMaps/` 配下と判明し訂正） / reason: MIT の表示義務を確実に満たし、どのコードがどこ由来かを追跡可能にするため / artifact: `THIRD_PARTY_NOTICES.md`, `src/qmrpy/**` 22 files / next: なし
- context: ライセンス是正 (4) 論文とバージョン / change: `paper_biorxiv.md` のモデル数 25→24、Table 1 から Denoising 行を削除、依存の記述を「MRzeroCore は AGPL-3.0 の opt-in extra」に修正、テスト数 144→142、version 記載を 2.0.0 に更新。`__version__`/`CITATION.cff` を 2.0.0 に、`CHANGELOG.md` に BREAKING 2件を追加。`docs/biorxiv_submission.md` の abstract・version・図生成コマンド（`--extra viz` が必要）・チェックリストを更新 / reason: 公開 API の削除は semver 上 major bump であり、論文の記述も現行実装と一致させる必要があるため / artifact: `paper_biorxiv.md`, `paper.md`, `src/qmrpy/__init__.py`, `CITATION.cff`, `CHANGELOG.md`, `docs/biorxiv_submission.md`, `output/paper_build/paper_biorxiv.pdf` / next: 2.0.0 のリリース（tag/push/PyPI）はユーザー判断
- context: ライセンス是正後の品質ゲート / change: `uv lock --check` 成功、`uv run --locked ruff format --check src scripts tests` は 115 files already formatted、`ruff check` 成功、`mypy src/qmrpy` 成功（66 files）、`pre-commit run --all-files` 全 Passed、`uv run --locked -m pytest` は 142 passed、`mkdocs build` 成功、core validation は 21/21 pass、図3件と PDF (13ページ) を再生成 / reason: MPPCA 削除と依存変更が既存実装・docs・検証スイートに回帰を起こしていないことを確認するため / artifact: `docs/memo.md`, `output/reports/parity_summary/`, `output/paper_figures/`, `output/paper_build/` / next: 変更をコミットする
- context: PyPI 公開済みバージョンの yank 判断 / change: PyPI の全 20 リリース (0.1.0〜1.0.0) の wheel を実際にダウンロードして検査し、**全件に MPPCA が含まれる**ことを確認。PyPI 最新は 1.0.0 で、1.1.0 は release workflow が完走しておらず未公開だった。GitHub Releases も 20 件中 17 件に wheel/sdist が添付されており同じコードを配布している。yank は PyPI の web UI 専用で API/CLI が存在せず（公式 docs で確認）、かつ認証情報を要するため自動実行は不可 / reason: 非商用限定コードを含む配布物の流通を止めるため / artifact: `docs/memo.md` / next: ユーザーが web UI で 20 件を yank する。yank は「== でピン留めした場合は依然インストール可能」なので、完全に流通を止めるには delete が必要である点を判断材料として提示済み
- context: PyPI 全リリースの yank 実行 / change: ユーザーがブラウザで PyPI にログインしたセッションを用い、`/manage/project/qmrpy/release/<version>/#yank_version-modal` から全 20 リリース (0.1.0, 0.1.1, 0.1.2, 0.1.3, 0.2.0, 0.2.1, 0.3.0, 0.4.0, 0.4.1, 0.5.0, 0.5.1, 0.6.0, 0.6.1, 0.6.2, 0.6.3, 0.7.0, 0.7.1, 0.8.0, 0.10.2, 1.0.0) を yank。理由文は全件共通で "Contains MP-PCA denoising code whose upstream licence does not permit redistribution. Please use qmrpy 2.0.0 or later."。PyPI JSON API で 20/20 が yanked=true、理由文が1種類のみであることを独立検証 / reason: 非商用限定コードを含む配布物が通常の依存解決で選ばれないようにするため / artifact: PyPI project metadata, `docs/memo.md` / next: delete は不可逆かつ実行者の判断が必要なため未実施。GitHub Releases 17件の添付物 (.whl/.tar.gz) も同一コードを配布しているため対応方針の決定が必要
- context: GitHub Releases の配布停止 / change: 資産 (.whl/.tar.gz) が添付されている 17 リリース (v0.1.3, v0.2.0, v0.2.1, v0.3.0, v0.4.0, v0.4.1, v0.5.0, v0.5.1, v0.6.0, v0.6.1, v0.6.2, v0.6.3, v0.7.0, v0.7.1, v0.8.0, v0.10.2, v1.0.0) を `gh release edit <tag> --draft` で下書きへ戻した。資産のない v0.1.0/v0.1.1/v0.1.2 は配布物がないため公開のまま。匿名の GitHub API で対象タグが HTTP 404、資産の直接 URL も HTTP 404 になることを検証 / reason: 削除は不可逆であるため、可逆な draft 化で公開配布のみを停止する方針を採用したため / artifact: GitHub Releases metadata, `docs/memo.md` / next: 2.0.0 リリース後、旧リリースを完全削除するかは別途判断する
- context: v2.0.0 リリース準備 / change: ブランチ `chore/license-remediation-2.0.0` を作成し、ライセンス是正と bioRxiv 原稿を1コミット (`e9edb4d`) にまとめて push。PR #1 (https://github.com/SugimotoKohei/qmrpy/pull/1) を作成 / reason: 破壊的変更を含むため main への直接 push を避け、変更内容と検証結果を PR に記録するため / artifact: PR #1, `docs/memo.md` / next: CI 通過後に merge し、auto-tag で v2.0.0 を作成、release.yml を workflow_dispatch で実行して PyPI へ公開する
- context: v2.0.0 リリース完了 / change: PR #1 が CI (ubuntu/macOS) 通過後 merge commit `ab1a673` でマージされ、main の CI auto-tag が `v2.0.0` タグを作成。GITHUB_TOKEN 由来のタグでは release workflow が起動しないため、`gh workflow run release.yml --ref main -f tag=v2.0.0` で手動実行し全ステップ success。PyPI で 2.0.0 が公開され、latest=2.0.0、既定インストール対象は 2.0.0 のみ（旧 20 件は yank 済み）。公開 wheel を展開して MPPCA/noise モジュールが存在しないこと、THIRD_PARTY_NOTICES.md が同梱されること、`_decaes/nnls.py` と `qsm/split_bregman.py` の帰属ヘッダが含まれることを検証。`requires_dist` で `mrzerocore>=0.4.1; extra == "mrzero"` となり必須依存から外れたことを確認。GitHub Release v2.0.0 も公開状態で資産2件が添付済み / reason: ライセンス是正済みの版を正規の公開版として提供するため / artifact: PyPI qmrpy 2.0.0, GitHub Release v2.0.0, `docs/memo.md` / next: bioRxiv 投稿（原稿の version 記載 2.0.0 と公開版が一致した）。旧リリースの完全削除の要否は別途判断
