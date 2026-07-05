# Changelog

このファイルは Keep a Changelog 形式に従います。

## [Unreleased]

## [1.1.0] - 2026-07-06

### Added

- NIfTI、DICOM、最小 qMRI-BIDS I/O helper を追加。
- `qmrpy` CLI を追加し、`info`、`fit`、`validate` を提供。
- T1rho spin-lock mapping を追加。
- MTR / MTsat mapping を追加。
- 辞書ベース MRF 同時 T1-T2 mapping を追加。
- 2プール近似の T2 water/fat separation を追加。
- ガバナンス文書（CONTRIBUTING、CHANGELOG、CODE_OF_CONDUCT、SECURITY、CITATION）を追加。
- pre-commit、mypy、coverage artifact、OS matrix を含む CI 品質ゲートを追加。

### Changed

- package keywords から未実装の diffusion を外し、relaxometry / magnetization transfer の実態に合わせた。
- core validation suite に T1rho、MT、MRF、T2 water/fat の synthetic recovery cases を追加。
- docs deploy workflow を locked dependency build に変更。

## [1.0.0] - 2026-01-15

### Added

- T1/T2/B0/B1/QSM/noise/simulation の主要モデルを統一 API で提供。
- `FitResult` による params 辞書互換 + `quality` / `diagnostics` メタデータスキーマを追加。
- core validation suite と JOSS 向け検証出力を整備。

[Unreleased]: https://github.com/SugimotoKohei/qmrpy/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/SugimotoKohei/qmrpy/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/SugimotoKohei/qmrpy/releases/tag/v1.0.0
