# bioRxiv 投稿メタデータとチェックリスト

`qmrpy` のプレプリントを bioRxiv に投稿するための、入力値・アップロード対象・手順をまとめた文書。
公式要件は 2026-08-01 に <https://www.biorxiv.org/submit-a-manuscript> と
<https://www.biorxiv.org/collection> を確認した内容に基づく。

---

## 0. 投稿物の生成手順（投稿前に必ず再実行）

```bash
uv run --locked -- python scripts/summarize_parity.py --suite core --formats csv,markdown,json --config configs/exp/validation_core.toml --out-dir output/reports/parity_summary
```

```bash
uv run --locked --extra viz -- python scripts/generate_paper_figures.py
```

```bash
uv run --locked -- python scripts/build_paper.py
```

生成物:

- 原稿 PDF: `output/paper_build/paper_biorxiv.pdf`（本文・表・図をすべて含む単一 PDF）
- 図（個別ファイル）: `output/paper_figures/fig1_validation_margin.{png,pdf}`,
  `fig2_parameter_recovery.{png,pdf}`, `fig3_t2_phantom_map.{png,pdf}`
- 検証エビデンス: `output/reports/parity_summary/core_validation.{csv,md,json}`

bioRxiv は「本文と図表を含む単一 PDF のアップロード」を最も簡単な方法として案内している。
本プロジェクトはこの形式に合わせてあるため、**アップロードは `paper_biorxiv.pdf` 1 ファイルで完結する**。

---

## 1. 基本メタデータ

| 項目 | 入力値 |
|---|---|
| Article title | qmrpy: a verification-first Python reference implementation for quantitative MRI modelling and cross-domain validation |
| Running title | Verification-first qMRI modelling in Python |
| Subject Collection | **Bioengineering** |
| Article category | **New Results** |
| Corresponding author | Kohei Sugimoto |
| ORCID | 0000-0003-2702-5235 |
| Affiliation | Independent Researcher, Japan |
| Email | sugimotokouhei@gmail.com |
| License | **CC BY 4.0** |
| Manuscript version | qmrpy 2.0.0 に対応 |

補足:

- Subject Collection の公式候補は Bioengineering / Biophysics / Neuroscience /
  Scientific Communication and Education など全 27 種。本稿は「qMRI モデルの実装と検証基盤」
  であるため Bioengineering を採用する。
- Article category は New Results / Confirmatory Results / Contradictory Results の 3 択。
  既報の再現ではなく新規ツールの提示であるため New Results。
- License は CC BY / CC BY-NC / CC BY-ND / CC BY-NC-ND / CC0 / no reuse から選択。
  リポジトリが MIT ライセンスであることと整合させ CC BY 4.0 とする。

---

## 2. Abstract（投稿フォーム貼り付け用・プレーンテキスト・226 語）

```text
Quantitative MRI (qMRI) converts image contrast into physical tissue parameters such as T1, T2, T1rho, magnetization transfer indices, magnetic susceptibility and myelin water fraction. Reference implementations of these models are distributed across separate software ecosystems - most notably MATLAB (qMRLab) and Julia (DECAES) - which makes it difficult to check, in one place, whether several model families remain numerically consistent under a single reproducible setup. We present qmrpy, an open-source Python package that implements 24 qMRI model classes across nine domains behind a single API contract (forward, fit, fit_image) and a common result schema, and that ships the verification evidence together with the code. Verification is defined declaratively: a configuration file fixes each case, its random seed, its primary metric and its pass threshold, and a summary script emits machine-readable CSV, Markdown and JSON reports. In the dependency-free core suite, all 21 cases pass across B0, B1, MRF, MT, QSM, simulation, T1, T2 and T2* mapping; the tightest case still sits 20% below its threshold, and two cases recover the reference exactly. Parameter recovery over physiological ranges gives mean relative errors of 2.6% (T1), 0.37% (T1rho) and 0.77% (T2), and a mean absolute myelin water fraction error of 0.011. Against fixed DECAES reference vectors, T2 distributions agree to 2.2 x 10^-15 without regularization. qmrpy makes cross-domain qMRI model behaviour auditable, regression-tested and directly usable from Python.
```

原稿本文中の Abstract（LaTeX 数式込み）と内容は同一。フォーム側は数式が使えないため
`2.2 x 10^-15` の表記に置換してある。

---

## 3. Keywords

```text
quantitative MRI; relaxometry; myelin water imaging; magnetization transfer; reproducibility; open-source software; Python
```

---

## 4. Declarations（フォームの申告欄に対応）

- **Competing interests**: The author declares no competing interests.
- **Funding**: This work received no specific grant from any funding agency.
- **Ethics approval / consent**: Not applicable. 人・動物由来データを一切使用していない
  （全結果が合成信号とデジタルファントムによる）。
- **Data and code availability**: <https://github.com/SugimotoKohei/qmrpy>（MIT）、
  リリースは <https://pypi.org/project/qmrpy/>。本稿の結果は version 2.0.0 で生成。
- **Generative AI disclosure**: 原稿の草案作成に生成AIを使用し、本文・数値・引用はすべて著者が検証。
  原稿の Declarations セクションにも同内容を明記済み。
- **Prior publication**: 未発表。bioRxiv は投稿時点で未公表であることを要件としている。
- **Author consent**: 単著のため共著者同意の取得は不要。

---

## 5. 投稿手順

1. <https://www.biorxiv.org/> の SUBMIT からアカウント登録（無料）またはログインする。
2. 新規投稿を開始し、Subject Collection に **Bioengineering** を選ぶ。
3. Article category に **New Results** を選ぶ。
4. Title / Running title / Abstract / Keywords を本文書の §1–§3 からコピーして入力する。
5. 著者情報（氏名・ORCID・所属・メール）を入力し、corresponding author に設定する。
6. Manuscript file として `output/paper_build/paper_biorxiv.pdf` を 1 ファイルだけアップロードする
   （Supplemental material は今回なし）。
7. License 選択画面で **CC BY 4.0** を選ぶ。
8. Declarations（competing interests / funding / ethics / data availability）を §4 の内容で申告する。
9. B2J（bioRxiv から提携ジャーナルへの直接転送）を使うかどうかを選ぶ。今回は後で個別に投稿先を
   決めるため **使わない** 想定。
10. 自動生成された PDF プルーフを確認する。特に図 1–3 の解像度と表 1・表 2 の崩れがないことを見る。
11. 投稿を確定し、スクリーニング（数営業日）の結果を待つ。

> **注意**: bioRxiv は一度公開されると DOI が付与され、**取り下げができない**（公式に
> "cannot be removed" と明記）。手順 10 のプルーフ確認までは慎重に行うこと。

---

## 6. 投稿前チェックリスト

- [ ] `uv run --locked scripts/summarize_parity.py --suite core` が 21/21 pass で終了する
- [ ] `uv run --locked --extra viz scripts/generate_paper_figures.py` が図 1–3 を再生成できる
- [ ] `uv run --locked scripts/build_paper.py` が警告なしで PDF を生成する
- [ ] PDF 内の Table 2 の数値が `output/reports/parity_summary/core_validation.csv` と一致する
- [ ] PDF 内の引用がすべて解決されている（`[@key]` の生テキストが残っていない）
- [ ] 参考文献の DOI が実在する（CrossRef で確認済み: 2026-08-01）
- [ ] 著者名・ORCID・連絡先メールが正しい
- [ ] Abstract が投稿フォームの語数上限に収まる（現状 226 語）
- [ ] リポジトリが public で、README の記載が原稿の主張と矛盾しない
- [ ] 原稿に記載した version（2.0.0）が PyPI / GitHub の公開版と一致する
      （2026-08-01 時点で PyPI 最新は 1.1.0。投稿前に 2.0.0 をリリースするか、
      原稿の version 記載を公開済みバージョンに合わせる必要がある）
- [ ] `THIRD_PARTY_NOTICES.md` の記載が現行のコード構成と一致する

---

## 7. 投稿後の推奨作業（任意）

- GitHub リポジトリを Zenodo と連携させ、ソフトウェアアーカイブ DOI を取得する。
  取得後に `paper.bib` の `qmrpy_zenodo`（現在 `DOI to be assigned`）と `CITATION.cff` を更新する。
- bioRxiv DOI を `CITATION.cff` と `README.md` の引用案内に追記する。
- 査読誌へ投稿する場合は、`paper.md`（JOSS 形式）を JOSS 用に、`paper_biorxiv.md` を
  一般誌用の原稿として使い分ける。

---

## 8. 補足: 投稿先の選択について

本稿は臨床データを含まない qMRI ソフトウェア・検証手法の論文であるため、bioRxiv
（Bioengineering）のほかに arXiv の `eess.IV` / `physics.med-ph` も一般的な選択肢になる。
medRxiv は臨床・健康関連の研究が対象であり、本稿は合成データのみのため適合しない。
bioRxiv と arXiv の二重投稿は多くの場合許容されるが、投稿先ごとのポリシーを確認すること。
