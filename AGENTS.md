<INSTRUCTIONS>
# AGENTS.md

## 基本方針（必須）

- すべての解答・提案・コメントは **日本語** で行うこと。
- このリポジトリは **qMRLab のPython移植実装を `qmrpy` パッケージとして公開（PyPI）**することを目的とする。
- 実装・検証・公開作業は、必ず `docs/studyplan.md` の計画・方針に従って進めること。
- 進捗は `docs/studynote.md` に逐次記録し、後から追跡できる形で残すこと（どのファイルを触ったかまで書く）。
- 重要な設計判断（API、I/O、単位、互換性、依存、公開方針）は `docs/decisions/` に根拠（採否と理由）を残すこと。
- コード・設定・ドキュメントの変更提案を行う際は、既存のディレクトリ構成・命名規則に整合するよう配慮すること。
- 再現性の原則（強制）：
  - **入力（コード・設定・手順・論文原稿）は Git 管理する**
  - **出力（実験結果・ログ・モデル・巨大データ）は原則 Git 管理しない**
  - ただし **論文に貼る最終成果（図表・原稿）は Git 管理する**

---

## 思考過程の可視化（Visualization-first：強制）

目的：実験の「結果」だけでなく、そこに至るまでの **観察→解釈→判断→次アクション** を可視化し、後から第三者（未来の自分）が「なぜそう進めたか」を追跡できる研究運用にする。

### 原則（強制）

- 実験は「数値の羅列」で終わらせず、必ず **可視化（図・表）＋短い解釈** をセットで残す。
- 可視化は “最終成果物” のためだけではない。むしろ **進行中の思考の外部記憶** として作る。
- 「良い可視化」とは、見栄えより **判断が変わる情報** を含むものを指す（例：分布、外れ値、分割別、時系列推移、比較差分）。

### 1 run あたりの最低要件（強制：done の定義）

各 run（`output/runs/<run_id>/`）は、最低限次を満たすこと：

- `metrics/`：主要指標（json/csv）の出力
- `figures/`：**最低 3 点**の図を出力（下記の「必須3点セット」）
- `run.json`：主要出力（metrics/figures/logs など）への相対パスを必ず含める
- `docs/studynote.md`：当該 run を参照し、**観察→解釈→次** を 1〜5 行で記録する

#### 必須3点セット（強制）

- (A) **データ確認図**：分布・欠損・外れ値・分割（train/valid/test）差などの「入力の健康診断」
- (B) **主要結果図**：Primary 指標の比較（条件差、ベースライン差、アブレーション差）
- (C) **失敗解析図**：誤差の内訳、残差、ケース別（層別）性能、キャリブレーション等の「どこで負けているか」

※ 図の種類はタスクに合わせて良いが、A/B/C の情報要件は満たすこと。

### 研究ログの書き方（強制：短くてよいが構造を固定）

`docs/studynote.md` には、run を実行したら最低限次を残す：

- 観察（Observation）：図・表から読み取れる事実（1〜2行）
- 解釈（Interpretation）：仮説の支持/反証、次に疑うべき要因（1〜2行）
- 次（Next）：次に回す実験、追加で切る可視化、確認するリスク（1行）

※ 「何となく良い/悪い」ではなく、「どの図の、どの差分で、判断がどう変わったか」を書く。

### 可視化の置き場（強制：層を分ける）

- 探索・途中経過：`output/runs/<run_id>/figures/`（Git 管理外）
- 論文に貼る最終版：`docs/manuscript/figures/`（Git 管理、昇格ルールに従う）
- 判断の根拠（採否と理由）：`docs/decisions/`（短くてよい）

### 可視化スタイル（規約）

- 可視化は原則 `plotnine` を使用し、図の生成は再実行可能（関数化・スクリプト化）に寄せる。
- 図ファイル名には「何の比較か」が分かる語を入れる（例：`metric_primary__baseline_vs_ablation.png`）。
- “図だけ生成して解釈が無い” 状態を禁止する。最低 1 行で良いので `studynote` に残す。

---

## 作業開始ゲート（Studyplan Gate：最初に必ずやる）

目的：研究開始前に「何を、どう検証し、何を探索と位置づけるか」を明文化し、後から説明できる研究運用にする。

### 原則（強制）

- リポジトリを立ち上げたら、**最初に AI（Codex/Gemini等）と共同して `docs/studyplan.md` を完成させる**。
- `docs/studyplan.md` が「完成（下記 Definition of Ready を満たす）」するまで、原則として以下を行わない：
  - `src/` の本体実装（最小限の雛形を除く）
  - `scripts/` の実験実装（最小限の入口雛形を除く）
  - `configs/exp/` の本格的な実験設定追加
  - 計算コストの大きい実験の実行
- 例外として、Studyplan を作るために必要な “初期足場” のみ許可する：
  - ディレクトリ作成（`docs/` など）
  - `AGENTS.md` / `README.md` / `.gitignore` の作成
  - `pyproject.toml` と `uv.lock` の整備（uvの標準設定の固定）
  - `docs/studyplan.md` のテンプレ投入（初版作成）
  - `docs/studynote.md` の作成（記録開始）

### Definition of Ready（Studyplan が「作業開始してよい」状態）

`docs/studyplan.md` が以下を満たしたら、`Status: active` とし、`Version: v0.1.0` を付けて作業開始する。

- 研究目的・スコープ（in/out）が明記されている
- Research Questions と仮説（少なくとも第一候補）が書かれている
- データ計画（取得・前処理・分割・除外基準）が書かれている
- 手法/モデルの候補と比較条件（ベースライン・アブレーション）が書かれている
- 評価指標（Primary/Secondary）と集計方法、seed/反復方針が書かれている
- 分析計画が **Confirmatory（検証）** と **Exploratory（探索）** に分離されている
- run の再現性仕様（`output/runs/<run_id>/run.json` と `config_snapshot/` 必須）が書かれている
- 論文計画（図表の昇格ルール、`figures_manifest.yaml` 運用）が書かれている
- リスクと対策、マイルストン（週単位目安）と完了条件（Exit criteria）が書かれている
- AI協働ルール（意思決定ログの残し方）が書かれている
- Changelog があり、更新ルール（差分＋理由）が明記されている

### AI協働の記録（強制）

- Studyplan の作成・更新で AI を使った場合、意思決定に影響した内容は必ず記録する：
  - 記録先：`docs/decisions/`（意思決定の要点、採否と理由）
  - 日々の作業ログ：`docs/studynote.md`（何をやったか・どのファイルを触ったか）
- Studyplan の更新は許可するが、必ず `docs/studyplan.md` の Changelog に「差分」と「理由」を残す。

---

## Python環境・ツールチェーン（uvで統一）

- Python の実行環境は **uv** で統一して管理すること。
  - Python のバージョンは **3.11 系**を使用すること。
  - Python の実行には `uv run` を使用すること。
- 依存パッケージやバージョンに言及する場合は、`pyproject.toml` および `uv.lock` と矛盾しない形で提案すること。
- **再現性確保のため、`uv.lock` は必ずバージョン管理に含めること。**

### uv運用ルール（規約）

- 依存追加：`uv add <pkg>`
- 依存削除：`uv remove <pkg>`
- 環境同期：`uv sync`
- ロック更新（明示的に行う）：`uv lock`
  - 更新が必要なら `uv lock --upgrade`（または `--upgrade-package <pkg>`）
- CI / 再現性チェック（ロックを勝手に変えない）：
  - `uv lock --check`
  - `uv sync --locked`
  - （必要なら）`uv run --locked -- <cmd>`

---

## Python可視化（標準：plotnine）

- Python による可視化は、**原則として `plotnine`（ggplot2 互換）を標準**として用いること。
- 例外的に `matplotlib` / `seaborn` 等を使う必要がある場合は、**その理由（plotnine で不足する点、必要な機能）と影響範囲**を明記すること（可能なら `docs/decisions/` に記録する）。
- 論文用図表（最終成果）を作る場合も、原則は plotnine で作成し、出力先や“昇格”ルール（`docs/manuscript/figures/` への固定、manifest 記録）に従うこと。

---

## ディレクトリ構成・リポジトリ運用

### トップレベル構成（必須）

- `src/qmrpy/`：配布対象のPythonパッケージ本体（import対象）
- `tests/`：自動テスト（pytest）
- `scripts/`：実験・検証の実行入口（薄く保つ）
- `configs/`：実験設定（再現性の入力、Git 管理）
- `docs/`：計画・記録・意思決定・論文（Git 管理）
- `notebooks/`：探索・可視化・説明（重い処理は避け、読み物/図作り中心）
- `.github/workflows/`：CI（pytest等）
- `output/`：run成果物（Git管理外）
- `data/`：データ（Git管理外）
- `qMRLab/`：upstream MATLAB実装のローカル参照用（**Git管理外**）

### ツリー表示（一覧・俯瞰用）

```text
.
├─ README.md
├─ AGENTS.md
├─ pyproject.toml
├─ uv.lock
├─ .gitignore
├─ .github/
│  └─ workflows/
├─ configs/
│  ├─ base/                  # 共通デフォルト（Git）
│  ├─ exp/                   # 実験エントリ（Git）
│  └─ local/                 # 個人/機密（Git外）
├─ data/
│  ├─ raw/                   # 元データ（Git外）
│  ├─ interim/               # 中間（Git外）
│  └─ processed/             # 最終入力（原則Git外／小さければGit可）
├─ output/
│  ├─ runs/
│  │  └─ <run_id>/
│  │     ├─ run.json         # 実行メタ情報（必須）
│  │     ├─ config_snapshot/ # 実行時設定スナップショット（必須）
│  │     ├─ env/             # 環境情報（任意）
│  │     ├─ metrics/         # 指標（json/csv）
│  │     ├─ figures/         # 図（原則Git外）
│  │     ├─ artifacts/       # モデル等（原則Git外）
│  │     └─ logs/            # ログ（原則Git外）
│  ├─ reports/               # 中間レポート（原則Git外）
│  └─ cache/                 # 捨ててよい（Git外）
├─ docs/
│  ├─ studyplan.md
│  ├─ studynote.md
│  ├─ decisions/
│  ├─ protocols/
│  └─ manuscript/            # 論文一式（Git）
│     ├─ figures/            # 論文に貼る最終図表（Git）
│     └─ figures_manifest.yaml  # 図の“出自”台帳（推奨）
├─ notebooks/                # jupyter (ipynb)
├─ scripts/                  # 実行入口（薄く保つ）
├─ tests/                    # pytest
└─ src/
   └─ qmrpy/                 # 配布対象パッケージ
```

### 新しいファイルの配置ルール（強制）

- 新しいファイル・スクリプト・ノートブックを提案する際は、上記いずれかのディレクトリに自然に収まる配置を提案すること。

---

## Git 運用（再現性の線引き）

- Git管理する（強制）：
  - `src/qmrpy/`, `tests/`, `scripts/`, `docs/`, `configs/base/`, `configs/exp/`, `notebooks/`, `.github/workflows/`
  - `pyproject.toml`, `uv.lock`, `README.md`, `AGENTS.md`
- 原則Git管理しない：
  - `output/`, `data/`, `configs/local/`, `qMRLab/`

---

## パッケージ公開（PyPI）運用（必須）

- 破壊的変更（API/入出力/単位）は `docs/decisions/` に必ず記録する。
- `main` 相当ブランチへのマージは CI（`.github/workflows/ci.yml`）で `pytest` が通ることを前提とする。
- リリース手順は `docs/protocols/release.md` を正とし、手順を増やしたら必ず更新する。

---

## コミットメッセージ規約（絵文字＋Conventional Commits）

- フォーマット：`<emoji> <type>(<scope>): <description>`
- `<type>` は Conventional Commits に準拠（例：`feat`, `fix`, `docs`, `refactor`, `test`, `chore`）。

---

## .gitignore（標準セット）

```gitignore
# outputs / data
output/
data/
configs/local/

# environments / caches
.venv/
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.ruff_cache/
.cache/

# OS/editor
.DS_Store
Thumbs.db
.vscode/
.idea/

# logs
*.log

# secrets (原則使わないが、万一置くなら絶対に管理しない)
.env
.env.*
```

---

## MRIシミュレーションに関する特別ルール

- 「MRI シミュレーション」一般について議論・提案を行うことは可能だが、ユーザーから明示的な指示があるまでは、`MRzero` を用いた具体的なシミュレーションや実装案には触れないこと。
</INSTRUCTIONS>
