# --------------------------------------------------------------------------------
# Package bootstrap
# --------------------------------------------------------------------------------

"""Public package entry point for PySparkSelectors."""

from __future__ import annotations

from typing import Any

# import pyspark

__all__ = [
    "initialize",
    "BaseSelector",
    "DTypeSelector",
    "IndexSelector",
    "RegexSelector",
    "SelectorFilterCondition",
    "SelectorSelectionOperations",
    "SelectorcolumnOperations",
    "by_dtype",
    "by_index",
    "matches",
    "by_name",
    "exclude",
    "is_selector",
]

_INITIALIZED = False


def _ensure_runtime_state() -> None:
    """Populate the package namespace without re-importing the package graph on each access."""
    global _INITIALIZED
    if _INITIALIZED:
        return

    from .models import (
        BaseSelector,
        DTypeSelector,
        IndexSelector,
        RegexSelector,
        SelectorcolumnOperations,
        SelectorFilterCondition,
        SelectorSelectionOperations,
    )
    from .user_functions import (
        all,
        alpha,
        alphanumeric,
        binary,
        boolean,
        by_dtype,
        by_index,
        by_name,
        contains,
        date,
        datetime_,
        ends_with,
        exclude,
        first,
        floats,
        integer,
        is_selector,
        last,
        matches,
        numeric,
        starts_with,
        string,
        temporal,
    )

    globals().update(
        {
            "BaseSelector": BaseSelector,
            "DTypeSelector": DTypeSelector,
            "IndexSelector": IndexSelector,
            "RegexSelector": RegexSelector,
            "SelectorFilterCondition": SelectorFilterCondition,
            "SelectorSelectionOperations": SelectorSelectionOperations,
            "SelectorcolumnOperations": SelectorcolumnOperations,
            "by_dtype": by_dtype,
            "by_index": by_index,
            "matches": matches,
            "by_name": by_name,
            "exclude": exclude,
            "is_selector": is_selector,
            "alpha": alpha,
            "alphanumeric": alphanumeric,
            "all": all,
            "binary": binary,
            "boolean": boolean,
            "contains": contains,
            "date": date,
            "datetime_": datetime_,
            "ends_with": ends_with,
            "first": first,
            "floats": floats,
            "integer": integer,
            "last": last,
            "numeric": numeric,
            "starts_with": starts_with,
            "string": string,
            "temporal": temporal,
        }
    )

    initialize()
    _INITIALIZED = True


def initialize() -> None:
    """Initialize the selector patch layer after the package has loaded."""
    from pyspark.sql import DataFrame
    from pyspark.sql import functions as F
    from pyspark.sql.column import Column
    from pyspark.sql.group import GroupedData

    from .models import SelectorcolumnOperations
    from .spark_overrides import (
        _build_agg_override,
        _build_filter_override,
        _build_with_column_override,
        _build_with_columns_override,
        _make_column_only_wrapper,
        # _make_dispatching_column_wrapper,
        _patch_classes,
        make_expr_based_override,
        make_name_based_override,
    )
    from .utils import _get_column_method

    try:
        from pyspark.sql.connect.group import GroupedData as _ConnectGroupedData
    except ImportError:
        _ConnectGroupedData = None

    try:
        from pyspark.sql.classic.dataframe import DataFrame as _ClassicDataFrame
    except ImportError:
        _ClassicDataFrame = None

    try:
        from pyspark.sql.connect.dataframe import DataFrame as _ConnectDataFrame
    except ImportError:
        _ConnectDataFrame = None

    try:
        from pyspark.sql.connect.column import Column as _ConnectColumn
    except ImportError:
        _ConnectColumn = None

    _DATAFRAME_CLASSES = tuple(
        dict.fromkeys(cls for cls in (DataFrame, _ClassicDataFrame, _ConnectDataFrame) if cls is not None)
    )

    _f_callable_names = {name for name in dir(F) if callable(getattr(F, name, None))}

    for fx in dir(F):
        myfx = getattr(F, fx)
        if callable(myfx):
            setattr(SelectorcolumnOperations, fx, SelectorcolumnOperations.spark_wrapper(myfx))

    _column_method_names = set(dir(Column))
    if _ConnectColumn is not None:
        _column_method_names |= set(dir(_ConnectColumn))

    for _method_name in sorted(_column_method_names):
        if _method_name.startswith("_"):
            continue
        if _method_name in ("prefix", "suffix", "map_alias"):
            continue

        try:
            _column_attr = _get_column_method(_method_name)
        except AttributeError:
            continue

        if not callable(_column_attr):
            continue

        if _method_name in _f_callable_names:
            _column_wrapper = SelectorcolumnOperations._make_dispatching_column_wrapper(_method_name)
        else:
            _column_wrapper = _make_column_only_wrapper(_method_name)

        setattr(
            SelectorcolumnOperations,
            _method_name,
            SelectorcolumnOperations.spark_wrapper(_column_wrapper),
        )

    # alias requires special handling as  it isn't being properly overridden

    make_expr_based_override("select")
    make_expr_based_override("sort", alias_results=False)
    make_expr_based_override("orderBy", alias_results=False)
    make_name_based_override("drop")
    make_name_based_override("groupBy")

    for _cls in _DATAFRAME_CLASSES:
        _cls.groupby = _cls.groupBy

    _GROUPED_DATA_CLASSES = tuple(c for c in (GroupedData, _ConnectGroupedData) if c is not None)

    _patch_classes(_DATAFRAME_CLASSES, "withColumn", _build_with_column_override)
    _patch_classes(_DATAFRAME_CLASSES, "withColumns", _build_with_columns_override)
    _patch_classes(_GROUPED_DATA_CLASSES, "agg", _build_agg_override)
    _patch_classes(_DATAFRAME_CLASSES, "filter", _build_filter_override)

    for _cls in _DATAFRAME_CLASSES:
        _cls.where = _cls.filter

    SelectorcolumnOperations.cast = SelectorcolumnOperations.spark_wrapper(Column.cast)
    globals().setdefault("_INITIALIZED", True)


def __getattr__(name: str) -> Any:
    if name in globals():
        return globals()[name]
    _ensure_runtime_state()
    if name in globals():
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


# Importing the package should not require a second manual call to initialize(),
# but a direct call remains available for explicit runtime patching.
_ensure_runtime_state()

_initialize = initialize
