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
- `qmrpy.functional` に `epg_t2_forward` / `epg_t2_fit` を追加
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
  - クラス名: `VfaT1`→`VFAT1`, `EpgT2`→`EPGT2`, `DecaesT2Map`→`DECAEST2Map`, `DecaesT2Part`→`DECAEST2Part`
  - 関数名: `SimVary`→`sim_vary`, `SimRnd`→`sim_rnd`, `SimFisherMatrix`→`sim_fisher_matrix`, `SimCRLB`→`sim_crlb`
  - パラメータ名・戻り値キーも snake_case 化
- `docs/index.md` のライセンス表記を BSD-2-Clause → MIT に修正（LICENSE ファイルと統一）
- `THIRD_PARTY_NOTICES.md` を日本語から英語に翻訳
- `paper.md` / `paper.bib` は JOSS 投稿予定のため公開リポジトリに残す（JOSS は公開リポジトリ必須）

## 2026-02-03
- v0.9.0: EPG シミュレーションモジュール追加 (`src/qmrpy/epg/`)
  - `epg/core.py`: 汎用 EPG エンジン（状態遷移行列、RF回転、緩和演算子）
  - `epg/epg_se.py`: Spin Echo シーケンス（CPMG, MESE, TSE/FSE）
  - `epg/epg_gre.py`: Gradient Echo シーケンス（SPGR, bSSFP, SSFP-FID/Echo）
- `tests/test_epg.py` 追加（21 tests）
- `docs/api/epg.md` ドキュメント追加
- `paper.md` を JOSS フォーマットに更新
