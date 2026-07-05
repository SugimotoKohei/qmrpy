# Security Policy

## Supported Versions

現時点では、最新の `main` ブランチと最新リリースを対象に脆弱性報告を受け付けます。

## Reporting a Vulnerability

脆弱性を見つけた場合は、公開 issue に詳細な攻撃手順や未修正の exploit を投稿しないでください。GitHub Security Advisories を使うか、リポジトリの issue で「security contact needed」とだけ連絡してください。

報告には、可能な範囲で以下を含めてください。

- 影響を受けるバージョンまたは commit
- 再現手順
- 想定される影響
- 回避策の有無

## Response

メンテナーは報告を確認し、影響範囲、修正方針、公開タイミングを判断します。修正が必要な場合は、テストを伴う最小変更で対応し、必要に応じてリリースノートに記載します。

## Scope

qmrpy は研究・解析用 Python パッケージです。依存パッケージや外部ツールに由来する脆弱性は、可能な範囲で upstream の修正に追随します。
