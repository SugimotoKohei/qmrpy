from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


class FitResult(dict[str, Any]):
    """Fitting result with param-dict compatibility and rich metadata access.

    This object behaves like a params dictionary for ergonomic access:
    ``result["t1_ms"]``.
    Additional metadata is available via attributes:
    ``result.quality`` and ``result.diagnostics``.

    Backward-compatible nested access is also supported:
    ``result["params"]``, ``result["quality"]``, ``result["diagnostics"]``.
    """

    __slots__ = ("quality", "diagnostics")

    def __init__(
        self,
        *,
        params: Mapping[str, Any] | None = None,
        quality: Mapping[str, Any] | None = None,
        diagnostics: Mapping[str, Any] | None = None,
        default_status: str = "ok",
    ) -> None:
        super().__init__(dict(params or {}))
        if quality is None:
            q = {"rmse": None, "n_points": None, "status": default_status}
        else:
            q = dict(quality)
            q.setdefault("rmse", None)
            q.setdefault("n_points", None)
            q.setdefault("status", default_status)
        self.quality: dict[str, Any] = q
        self.diagnostics: dict[str, Any] = dict(diagnostics or {})

    @property
    def params(self) -> dict[str, Any]:
        return self

    def __getitem__(self, key: str) -> Any:
        if key == "params":
            return self.params
        if key == "quality":
            return self.quality
        if key == "diagnostics":
            return self.diagnostics
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        if key == "params":
            return self.params
        if key == "quality":
            return self.quality
        if key == "diagnostics":
            return self.diagnostics
        return super().get(key, default)

    def __contains__(self, key: object) -> bool:
        if key in {"params", "quality", "diagnostics"}:
            return True
        return super().__contains__(key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "params": dict(self),
            "quality": dict(self.quality),
            "diagnostics": dict(self.diagnostics),
        }

    def copy(self) -> FitResult:
        return FitResult(
            params=self,
            quality=self.quality,
            diagnostics=self.diagnostics,
        )

    def __repr__(self) -> str:
        return (
            "FitResult("
            f"params={dict(self)!r}, "
            f"quality={self.quality!r}, "
            f"diagnostics={self.diagnostics!r}"
            ")"
        )


def is_nested_result(result: Mapping[str, Any]) -> bool:
    if isinstance(result, FitResult):
        return True
    return "params" in result and "quality" in result and "diagnostics" in result


def nest_result(
    result: Mapping[str, Any],
    *,
    param_keys: Iterable[str],
    rmse_keys: Iterable[str] = ("rmse", "res_rmse", "residual", "resid_l2"),
    n_points_keys: Iterable[str] = ("n_points",),
    default_status: str = "ok",
) -> FitResult:
    if isinstance(result, FitResult):
        return result

    if is_nested_result(result):
        params_raw = result.get("params", {})
        quality_raw = result.get("quality", {})
        diagnostics_raw = result.get("diagnostics", {})
        return FitResult(
            params=params_raw if isinstance(params_raw, Mapping) else {},
            quality=quality_raw if isinstance(quality_raw, Mapping) else {},
            diagnostics=diagnostics_raw if isinstance(diagnostics_raw, Mapping) else {},
            default_status=default_status,
        )

    pkeys = set(param_keys)
    rkeys = set(rmse_keys)
    nkeys = set(n_points_keys)

    params: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    quality: dict[str, Any] = {"rmse": None, "n_points": None, "status": default_status}

    for key, value in result.items():
        if key in pkeys:
            params[key] = value
            continue
        if key in rkeys and quality["rmse"] is None:
            quality["rmse"] = value
            continue
        if key in nkeys and quality["n_points"] is None:
            quality["n_points"] = value
            continue
        diagnostics[key] = value

    return FitResult(
        params=params,
        quality=quality,
        diagnostics=diagnostics,
        default_status=default_status,
    )
