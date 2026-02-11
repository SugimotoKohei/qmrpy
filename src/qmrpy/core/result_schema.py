from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def is_nested_result(result: Mapping[str, Any]) -> bool:
    return "params" in result and "quality" in result and "diagnostics" in result


def nest_result(
    result: Mapping[str, Any],
    *,
    param_keys: Iterable[str],
    rmse_keys: Iterable[str] = ("rmse", "res_rmse", "residual", "resid_l2"),
    n_points_keys: Iterable[str] = ("n_points",),
    default_status: str = "ok",
) -> dict[str, Any]:
    if is_nested_result(result):
        return dict(result)

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

    return {"params": params, "quality": quality, "diagnostics": diagnostics}
