# Contributing to qmrpy

qmrpy への貢献では、既存 API の互換性、再現可能な検証、テスト可能な小さい変更を優先します。

## 開発環境

Python 環境は uv で統一します。Python は 3.11 系を主対象にします。

```bash
uv sync --locked --extra dev --group docs
```

実データ I/O を扱う場合:

```bash
uv sync --locked --extra dev --extra io --group docs
```

## 検証コマンド

変更前後で必要な範囲を確認し、フェーズ単位または機能単位のコミット前には以下を通してください。

```bash
uv lock --check
uv run --locked -m pytest
uv run --locked ruff check src tests scripts
uv run --locked mkdocs build
uv run --locked scripts/summarize_parity.py --suite core
```

型チェックはフェーズ5で CI ゲートとして整備します。導入後は `uv run --locked mypy src/qmrpy` を標準確認に含めます。

## 実装方針

- `src/` は import 対象の本体コードのみ置きます。
- `scripts/` は薄い実行入口または検証スクリプトに限定します。
- 新規モデルは既存モデルと同じ `forward` / `fit` / `fit_image` パターンに合わせます。
- `fit` / `fit_image` の公開戻り値は `FitResult` スキーマを維持します。
- `mask="otsu"`、`n_jobs=-1`、`verbose` は画像 fitting で可能な限り揃えます。
- optional dependency は遅延 import にし、未インストール時は解決方法が分かる `ImportError` にします。

## ドキュメント

新しい公開 API を追加した場合は、以下も更新してください。

- `README.md`
- `docs/api/`
- 関連する `docs/guide/`
- `mkdocs.yml` の nav
- 必要に応じて `configs/exp/validation_core.toml` と `scripts/summarize_parity.py`

## 作業ログ

作業・判断・変更・次アクションが発生した場合は、`docs/memo.md` に日付見出しと箇条書きで追記します。既存の過去エントリは編集しません。

## コミットメッセージ

コミットメッセージは絵文字 + Conventional Commits 形式です。

```text
✨ feat(models): add VFA T1 fitting
🐛 fix(io): preserve NIfTI affine
📝 docs(api): update T2 guide
```

破壊的変更が避けられない場合は、本文またはフッタに `BREAKING CHANGE:` を明記してください。
