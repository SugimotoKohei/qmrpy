from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Protocol

from .result_schema import nest_result

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


class FitModelProtocol(Protocol):
    def fit(self, signal: ArrayLike, **kwargs: Any) -> dict[str, Any]: ...


class FitImageModelProtocol(Protocol):
    def fit_image(self, signal: ArrayLike, **kwargs: Any) -> dict[str, Any]: ...


class ResultSchemaMixin:
    """Mixin that converts flat result dictionaries to nested result schema."""

    _PARAM_KEYS: ClassVar[tuple[str, ...]] = ()
    _RMSE_KEYS: ClassVar[tuple[str, ...]] = ("rmse", "res_rmse", "residual", "resid_l2")
    _N_POINTS_KEYS: ClassVar[tuple[str, ...]] = ("n_points",)

    def _nest(self, result: dict[str, Any]) -> dict[str, Any]:
        return nest_result(
            result,
            param_keys=self._PARAM_KEYS,
            rmse_keys=self._RMSE_KEYS,
            n_points_keys=self._N_POINTS_KEYS,
        )

    def fit(self, signal: ArrayLike, **kwargs: Any) -> dict[str, Any]:
        result = super().fit(signal, **kwargs)  # type: ignore[misc]
        return self._nest(result)

    def fit_image(self, signal: ArrayLike, **kwargs: Any) -> dict[str, Any]:
        result = super().fit_image(signal, **kwargs)  # type: ignore[misc]
        return self._nest(result)


class ResultAdapterBase:
    """Composition-based adapter to keep model internals unchanged."""

    _PARAM_KEYS: ClassVar[tuple[str, ...]] = ()
    _RMSE_KEYS: ClassVar[tuple[str, ...]] = ("rmse", "res_rmse", "residual", "resid_l2")
    _N_POINTS_KEYS: ClassVar[tuple[str, ...]] = ("n_points",)
    _IMPL_CLS: ClassVar[type[Any]]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._impl = self._IMPL_CLS(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._impl, name)

    def _nest(self, result: dict[str, Any]) -> dict[str, Any]:
        return nest_result(
            result,
            param_keys=self._PARAM_KEYS,
            rmse_keys=self._RMSE_KEYS,
            n_points_keys=self._N_POINTS_KEYS,
        )

    def fit(self, signal: ArrayLike, **kwargs: Any) -> dict[str, Any]:
        result = self._impl.fit(signal, **kwargs)
        return self._nest(result)

    def fit_image(self, signal: ArrayLike, **kwargs: Any) -> dict[str, Any]:
        result = self._impl.fit_image(signal, **kwargs)
        return self._nest(result)
