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
