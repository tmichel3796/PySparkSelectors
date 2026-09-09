from collections.abc import Callable, Sequence
from typing import Any, Union

from pyspark.errors import PySparkTypeError, PySparkValueError
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.column import Column

from .models import (
    BaseSelector,
    SelectorFilterCondition,
)
from .utils import (
    _get_column_method,
    _resolve_selector_exprs,
    _resolve_selector_names,
    _true_original,
)

""" enable support for many different types of pyspark environments"""
# if your environment isn't supported then you can add it to this section.
# Spark supports multiple DataFrame implementations. Keep every available runtime
# class in the patch set: public/classic DataFrame and Spark Connect DataFrame.
# using try except as to allow passthrough if environment doesn't support these.
try:
    from pyspark.sql.classic.dataframe import DataFrame as _ClassicDataFrame
except ImportError:
    _ClassicDataFrame = None

try:
    from pyspark.sql.connect.dataframe import DataFrame as _ConnectDataFrame
except ImportError:
    _ConnectDataFrame = None

_DATAFRAME_CLASSES = tuple(
    dict.fromkeys(cls for cls in (DataFrame, _ClassicDataFrame, _ConnectDataFrame) if cls is not None)
)

try:
    from pyspark.sql.connect.column import Column as _ConnectColumn
except ImportError:
    _ConnectColumn = None

_COLUMN_CLASSES = tuple(dict.fromkeys(cls for cls in (Column, _ConnectColumn) if cls is not None))


# --------------------------------------------------
# ensure all versions of dataframes are overriden to support column selectors
# --------------------------------------------------


def _patch_classes(
    classes: Sequence[type],
    method_name: str,
    build_override: Callable[[Callable[..., Any]], Callable[..., Any]],
) -> None:
    """Patch `method_name` onto every class in `classes` with a fresh override.

    Shared "apply this override everywhere it needs to exist" helper --
    installs a selector-aware override of `method_name` onto each class in
    `classes` (e.g. both classic `pyspark.sql.DataFrame` and Spark Connect's
    `pyspark.sql.connect.dataframe.DataFrame`, when Connect is available), so
    the same selector syntax works identically no matter which backend
    actually produced the dataframe being operated on.

    Each class gets its OWN override function, built by calling
    `build_override` once per class with that class's own real, never-patched
    original implementation (via `_true_original`) -- classic and Connect
    implementations of the same method are separate functions and are not
    interchangeable, so one shared override closing over a single captured
    original would silently call the wrong implementation for the other
    class. Calling `build_override` fresh for each class (rather than
    reusing one closure variable across a loop) also avoids Python's
    late-binding closure pitfall, where every override would otherwise end
    up referencing whichever class's original happened to be captured last.

    Parameters
    ----------
    classes : sequence of type
        Every class to patch `method_name` onto.
    method_name : str
        The name of the method being overridden.
    build_override : callable
        Given that class's real original implementation (an unbound
        function called as ``original(self, *args, **kwargs)``), returns
        the override function to install in its place.

    Examples
    --------
    >>> _patch_classes(
    ...     _DATAFRAME_CLASSES, "sort",
    ...     lambda original: (lambda self, *cols, **kw: original(self, *cols, **kw)),
    ... )
    """
    for cls in classes:
        original = _true_original(cls, method_name)
        override = build_override(original)
        override._cs_original = original
        override.__name__ = method_name
        override.__qualname__ = f"{cls.__name__}.{method_name}"
        setattr(cls, method_name, override)


# --------------------------------------------------------------------------------
#  wrapper that will allow for column functions to be wrapped for use in selectors
#  we will focus on columns that are initialized in pyspark.sql.column
#
# --------------------------------------------------------------------------------


def _make_dispatching_column_wrapper(method_name: str) -> Callable[..., Column]:
    """Build a wrapper that dispatches to `Column` or `pyspark.sql.functions` by argument type.

    Resolves the `F`/`Column` name-collision described in the markdown above:
    `pyspark.sql.functions.<method_name>` and `pyspark.sql.Column.<method_name>`
    share a name but treat a plain (non-`Column`) argument differently. This
    wrapper picks whichever one matches the caller's intent based on the type of
    the argument actually passed, so both usages keep working.

    Parameters
    ----------
    method_name : str
        Name shared by a `pyspark.sql.functions` function and a `pyspark.sql.Column`
        instance method (e.g. `"contains"`), used to look up both versions.

    Returns
    -------
    callable
        A function `wrapper(c, *args, **kwargs)` that, when the first positional
        argument (if any) is a `pyspark.sql.Column`, calls
        `pyspark.sql.functions.<method_name>(c, *args, **kwargs)` (column-to-column
        semantics); otherwise calls `getattr(c, method_name)(*args, **kwargs)`
        (literal semantics via `Column`'s own bound method). Carries `Column`'s
        docstring for the method so `help()` on the resulting chained selector
        method shows accurate documentation.
    """
    column_method = _get_column_method(method_name)
    spark_func = getattr(F, method_name)

    def wrapper(c: Column, *args: Any, **kwargs: Any) -> Column:
        # `_COLUMN_CLASSES` (classic + Spark Connect `Column`, when Connect is
        # available) instead of the bare `Column` name -- otherwise a Spark
        # Connect `Column` argument would be misdetected as a plain literal
        # and routed to the wrong (literal-semantics) branch below.
        if args and isinstance(args[0], _COLUMN_CLASSES):
            return spark_func(c, *args, **kwargs)
        return getattr(c, method_name)(*args, **kwargs)

    wrapper.__name__ = method_name
    wrapper.__doc__ = column_method.__doc__
    # same rename-op marker carry-through as `_make_column_only_wrapper`
    # (defensive -- no current rename op collides with an `F` name).
    wrapper._is_rename_op = getattr(column_method, "_is_rename_op", False)
    return wrapper


# --------------------------------------------------
# used to override functions that require column expressions
# --------------------------------------------------


def make_expr_based_override(
    method_name: str,
    alias_results: bool = True,
    classes: Sequence[type] | None = None,
) -> None:
    """Patch a pyspark method that wants `Column` expressions.

    Factory (wrapper pattern) used by every fx override whose real pyspark
    method wants `pyspark.sql.Column` *expressions* rather than bare names --
    e.g. `sort`/`orderBy` need real expressions so a selector's chained
    ``.desc()`` transform actually takes effect, not just the underlying
    column name.

    Unlike earlier versions of this notebook, this function performs the
    patch itself (via `_patch_classes`) rather than returning a single
    override for the caller to assign -- classic `pyspark.sql.DataFrame` and
    Spark Connect's `DataFrame` each need their own override closing over
    their own real original implementation.

    Parameters
    ----------
    method_name : str
        The name of the `DataFrame` method to override (e.g. ``"select"``,
        ``"sort"``, ``"orderBy"``).
    alias_results : bool, default True
        Whether to re-alias each resolved expression back to the original
        column name it matched (see the generated override's docstring).
        Should be `False` for methods like `sort`/`orderBy` that don't
        change the output schema at all -- there, a chained
        ``.desc()``/``.asc()`` produces a non-evaluable Catalyst
        `SortOrder` expression, which `.alias(...)` cannot be applied to.
    classes : sequence of type, optional
        Every class to patch. Defaults to `_DATAFRAME_CLASSES` (classic
        `pyspark.sql.DataFrame` plus Spark Connect's `DataFrame`, when
        Spark Connect is importable).

    Examples
    --------
    >>> make_expr_based_override('sort', alias_results=False)
    >>> df.sort(by_dtype([T.DoubleType]).desc())
    DataFrame[...]
    """
    if classes is None:
        classes = _DATAFRAME_CLASSES

    def build_override(original):

        def override(self, *cols, **kwargs):

            if alias_results:
                resolved = [
                    expr.alias(name) if name is not None else expr for name, expr in _resolve_selector_exprs(self, cols)
                ]
            else:
                resolved = [expr for _, expr in _resolve_selector_exprs(self, cols)]

            return original(self, *resolved, **kwargs)

        override.__doc__ = (
            f"{method_name}(*cols, **kwargs) -- selector-aware override.\n\n"
            "Accepts a column selector (e.g. `by_dtype([T.DoubleType]).desc()`) "
            "anywhere a plain `pyspark.sql.Column` expression is normally "
            "accepted; the selector is resolved to `Column` expressions (with "
            "any chained transform, e.g. `.cast(...)`/`.desc()`, applied)"
            + (
                ", then each is re-aliased back to the original column name it "
                "matched, so a transform like `.upper()`/`.cast(...)` still keeps "
                "the source column's name instead of Spark's auto-generated "
                "expression name"
                if alias_results
                else ""
            )
            + f". The result is handed to the real `{method_name}` "
            "implementation. Plain columns/args pass through unchanged.\n\n"
            f"{original.__doc__ or ''}"
        )

        return override

    _patch_classes(classes, method_name, build_override)


# --------------------------------------------------
# override functions that only require column name  strings
# --------------------------------------------------


def make_name_based_override(method_name: str, classes: Sequence[type] | None = None) -> None:
    """Patch a pyspark method that only wants column-name strings.

    Factory (wrapper pattern) used by every fx override whose real pyspark
    method just wants a flat list of column-name strings -- resolves any
    `BaseSelector` passed positionally down to the names it matches, then
    calls straight through to the real pyspark method of the same name so
    the signature/behavior is otherwise identical. This is what collapses
    `groupBy`/`drop` into one-liners instead of each needing its own
    hand-written override.

    Unlike earlier versions of this notebook, this function performs the
    patch itself (via `_patch_classes`) rather than returning a single
    override for the caller to assign -- classic `pyspark.sql.DataFrame` and
    Spark Connect's `DataFrame` each need their own override closing over
    their own real original implementation.

    Parameters
    ----------
    method_name : str
        The name of the `DataFrame` method to override (e.g. ``"groupBy"``,
        ``"drop"``).
    classes : sequence of type, optional
        Every class to patch. Defaults to `_DATAFRAME_CLASSES` (classic
        `pyspark.sql.DataFrame` plus Spark Connect's `DataFrame`, when
        Spark Connect is importable).

    Examples
    --------
    >>> make_name_based_override('groupBy')
    >>> df.groupBy(by_dtype([T.StringType]))
    GroupedData[...]
    """
    if classes is None:
        classes = _DATAFRAME_CLASSES

    def build_override(original):

        def override(self, *cols, **kwargs):

            resolved = _resolve_selector_names(self, cols)

            return original(self, *resolved, **kwargs)

        override.__doc__ = (
            f"{method_name}(*cols, **kwargs) -- selector-aware override.\n\n"
            "Accepts a column selector (e.g. `by_dtype([T.StringType])`) anywhere a "
            "plain column-name string is normally accepted; the selector is "
            f"resolved to the column names it matches, then handed to the "
            f"real `{method_name}` implementation. Plain names/args pass "
            "through unchanged.\n\n"
            f"{original.__doc__ or ''}"
        )

        return override

    _patch_classes(classes, method_name, build_override)


# --------------------------------------------------
# used to override all functions in  pyspark.sql.columns for column selector support
# --------------------------------------------------


def _make_column_only_wrapper(method_name: str) -> Callable[..., Column]:
    """Build a wrapper that delegates to a `pyspark.sql.Column`-only bound method.

    For `Column` methods with no `pyspark.sql.functions` equivalent at all (e.g.
    `isNull`, `isin`, `between`, `alias`, `getItem`), there is no `F` version to
    dispatch to, so this simply calls the bound method directly.

    Parameters
    ----------
    method_name : str
        Name of the `pyspark.sql.Column` instance method to delegate to.

    Returns
    -------
    callable
        A function `wrapper(c, *args, **kwargs)` that returns
        `getattr(c, method_name)(*args, **kwargs)`, carrying `Column`'s own
        docstring for the method so `help()` on the resulting chained selector
        method shows accurate documentation.
    """
    column_method = _get_column_method(method_name)

    def wrapper(c: Column, *args: Any, **kwargs: Any) -> Column:
        return getattr(c, method_name)(*args, **kwargs)

    wrapper.__name__ = method_name
    wrapper.__doc__ = column_method.__doc__
    # carry the `@_rename_op` marker (e.g. `.prefix()`/`.suffix()`/
    # `.map_alias()`) through to this wrapper, so `spark_wrapper` -- which
    # actually receives this wrapper, not the raw `Column` method -- can see
    # it and flag the selector copy as having a rename op applied.
    wrapper._is_rename_op = getattr(column_method, "_is_rename_op", False)
    return wrapper


# --------------------------------------------------
# handle override for agg fx
# --------------------------------------------------


def _build_agg_override(original):
    """Build one `GroupedData.agg` override closing over `original`.

    Parameters
    ----------
    original : callable
        That class's real, never-patched `GroupedData.agg` implementation
        (via `_true_original`).

    Returns
    -------
    callable
        The override function to install in place of `original`.
    """

    def agg_patched(self, *exprs: Any, **kwargs: Any) -> DataFrame:
        """Aggregate, accepting column selectors (with chained aggregation transforms).

        Anywhere pyspark accepts a `Column` expression, a selector with chained
        aggregation transforms (e.g. `by_dtype([T.DoubleType]).sum()`) works too.

        Patches `GroupedData.agg` (both classic `pyspark.sql.group.GroupedData`
        and, when Spark Connect is available,
        `pyspark.sql.connect.group.GroupedData`).
        `by_dtype([T.DoubleType]).sum()`'s chained ``.sum()`` transform (already
        available via `SelectorcolumnOperations`'s spark-function wrapping
        loop) is honored via `_resolve_selector_exprs`, resolved against
        ``self._df`` (the underlying dataframe `GroupedData` was built from).
        Each resolved aggregation is then re-aliased back to the original
        matched column name -- instead of pyspark's default auto-generated
        name (`sum(column_name)`) -- matching the same "preserve the source
        column's name" behavior `select`/`withColumn`/`withColumns` already
        have. A selector with a `.prefix()`/`.suffix()`/`.map_alias()`
        chained on still renames as expected, since `name` here already
        reflects any queued rename (see `_resolve_selector_exprs`).

        Parameters
        ----------
        *exprs : Any
            Column selectors and/or plain `pyspark.sql.Column` aggregation
            expressions, same shape as real `GroupedData.agg`.
        **kwargs : Any
            Passed straight through to the real `GroupedData.agg`.

        Returns
        -------
        pyspark.sql.DataFrame
            One row per group, with one output column per resolved
            aggregation expression, named after the original column it
            aggregated (unless that column had a rename chained onto it, in
            which case the new name is used).

        Examples
        --------
        >>> df.groupBy('region').agg(by_dtype([T.DoubleType]).sum())
        DataFrame[...]
        """

        resolved = [
            expr.alias(name) if name is not None else expr for name, expr in _resolve_selector_exprs(self._df, exprs)
        ]

        return original(self, *resolved, **kwargs)

    return agg_patched


# --------------------------------------------------
# handle override for filter  fx
# --------------------------------------------------


def _build_filter_override(original):
    """Build one `DataFrame.filter` override closing over `original`.

    Parameters
    ----------
    original : callable
        That class's real, never-patched `DataFrame.filter` implementation
        (via `_true_original`).

    Returns
    -------
    callable
        The override function to install in place of `original`.
    """

    def filter_patched(self, condition: Union[BaseSelector, "SelectorFilterCondition", Column]) -> DataFrame:
        """Filter rows, accepting a column selector as the condition.

        Patches `DataFrame.filter` (and, via the `where` alias,
        `DataFrame.where`) -- both classic `pyspark.sql.DataFrame` and, when
        Spark Connect is available, its `pyspark.sql.connect.dataframe.DataFrame`.
        A selector matching multiple columns (e.g. ``by_dtype([T.IntegerType]) > 0``)
        is reduced down (ANDed together) into one boolean condition. Chained
        selector conditions and/or plain `pyspark.sql.Column` conditions
        compose with ``&``/``|``/``~`` -- see `SelectorSelectionOperations`
        and `SelectorFilterCondition`. Comma syntax
        (``df.filter(cond1, cond2)``) is intentionally not needed; ``&``
        alone composes any number of conditions together.

        Parameters
        ----------
        condition : BaseSelector, SelectorFilterCondition, or pyspark.sql.Column
            A column selector (optionally with a comparison/arithmetic transform
            chained onto it), a `SelectorFilterCondition` built by combining
            selectors/columns with ``& | ~``, or a plain
            `pyspark.sql.Column` boolean expression -- same shape as real
            `DataFrame.filter`.

        Returns
        -------
        pyspark.sql.DataFrame
            The filtered dataframe.

        Examples
        --------
        >>> df.filter(by_dtype([T.DoubleType]) > 0)
        DataFrame[...]
        >>> df.filter((by_dtype([T.IntegerType]) > 0) & (by_dtype([T.IntegerType]) < 9))
        DataFrame[...]
        """

        if isinstance(condition, (BaseSelector, SelectorFilterCondition)):
            combined = SelectorFilterCondition._to_expr(condition, self)

            return original(self, combined)

        return original(self, condition)

    return filter_patched


# --------------------------------------------------
# handle override for with_column fx
# --------------------------------------------------


def _build_with_column_override(original):
    """Build one `DataFrame.withColumn` override closing over `original`.

    Parameters
    ----------
    original : callable
        That class's real, never-patched `DataFrame.withColumn`
        implementation (via `_true_original`).

    Returns
    -------
    callable
        The override function to install in place of `original`.
    """

    def withColumn_patched(self, colName: str, col: BaseSelector | Column) -> DataFrame:
        """Add/replace one column, accepting a column selector that resolves to exactly one column.

        Patches `DataFrame.withColumn` (both classic `pyspark.sql.DataFrame`
        and, when Spark Connect is available, its
        `pyspark.sql.connect.dataframe.DataFrame`). Keeps its normal
        single-name/single-expression signature -- if `col` is a selector it
        must resolve to exactly one column; otherwise this raises, pointing
        the caller at `withColumns` for the multi-column case.

        Parameters
        ----------
        colName : str
            The name of the column to add or replace, same as real
            `DataFrame.withColumn`.
        col : BaseSelector or pyspark.sql.Column
            A column selector resolving to exactly one column (with whatever
            transforms are chained onto it, e.g. ``.cast(...)``/``+1``), or a
            plain `pyspark.sql.Column` expression.

        Returns
        -------
        pyspark.sql.DataFrame
            The dataframe with `colName` added/replaced.

        Raises
        ------
        pyspark.errors.PySparkValueError
            If `col` is a selector that resolves to zero or more than one
            column.

        Examples
        --------
        >>> df.withColumn('doubled', by_dtype([T.IntegerType]) * 2)
        DataFrame[...]
        """

        if isinstance(col, BaseSelector):
            pairs = _resolve_selector_exprs(self, [col])

            if len(pairs) != 1:
                raise PySparkValueError(
                    message=(
                        f"withColumn({colName!r}, ...) needs a selector resolving to exactly "
                        f"one column but got {len(pairs)} matches; use withColumns() for many at once."
                    ),
                )

            _, expr = pairs[0]
            return original(self, colName, expr)

        return original(self, colName, col)

    return withColumn_patched


# --------------------------------------------------
# handle override for with_columns fx
# --------------------------------------------------


def _build_with_columns_override(original):
    """Build one `DataFrame.withColumns` override closing over `original`.

    Parameters
    ----------
    original : callable
        That class's real, never-patched `DataFrame.withColumns`
        implementation (via `_true_original`).

    Returns
    -------
    callable
        The override function to install in place of `original`.
    """

    def withColumns_patched(self, *args: BaseSelector | dict[str, Column]) -> DataFrame:
        """Add/replace many columns in one call, accepting multiple column-selector mutations.

        Patches `DataFrame.withColumns` (both classic `pyspark.sql.DataFrame`
        and, when Spark Connect is available, its
        `pyspark.sql.connect.dataframe.DataFrame`). Real pyspark's
        `DataFrame.withColumns` only ever took one ``colsMap`` dict argument --
        that exact call shape (a single dict, no selector involved) still passes
        straight through unchanged, so base pyspark behavior is preserved. When
        multiple positional mutations are given (selectors and/or dicts), each
        resolves independently (its own selector match, its own chained
        transforms) into ``{name: expr}`` pairs and gets merged; a collision
        (two different mutations targeting the same output column name within
        the same call) raises rather than silently letting one clobber the
        other.

        Parameters
        ----------
        *args : BaseSelector or dict of {str: pyspark.sql.Column}
            One or more mutations. A single ``{name: Column}`` dict (pyspark's
            original call shape) passes straight through. Multiple positional
            args -- each a column selector (with whatever transforms are chained
            onto it, e.g. ``.cast(...)``/``.upper()``/``+1``) and/or a
            ``{name: Column}`` dict -- are resolved and merged into one call.

        Returns
        -------
        pyspark.sql.DataFrame
            The dataframe with every resolved mutation applied.

        Raises
        ------
        pyspark.errors.PySparkTypeError
            If a positional arg (when more than one is given) is neither a
            `BaseSelector` nor a `dict`.
        pyspark.errors.PySparkValueError
            If two different positional args resolve to the same output column
            name within the same call.

        Examples
        --------
        >>> df.withColumns(by_dtype([T.DoubleType]).cast('string'))
        DataFrame[...]
        >>> df.withColumns(by_dtype([T.DoubleType]).cast('string'), by_dtype([T.StringType]).upper())
        DataFrame[...]
        """

        # original pyspark call shape: withColumns({"name": col, ...}) -- exactly one
        # dict arg, no selector involved. pass straight through untouched.
        if len(args) == 1 and not isinstance(args[0], BaseSelector):
            return original(self, args[0])

        # one or more mutations, each either a column selector (with whatever
        # transforms are chained onto it, e.g. .cast(...)/.upper()/+1) or a plain
        # {name: Column} dict -- resolve each one independently, then merge them into a
        # single colsMap, raising on any collision instead of silently overwriting.
        merged = {}

        for arg in args:
            if isinstance(arg, BaseSelector):
                pairs = _resolve_selector_exprs(self, [arg])
                arg_map = {name: expr for name, expr in pairs}

            elif isinstance(arg, dict):
                arg_map = arg

            else:
                raise PySparkTypeError(
                    message=(
                        "withColumns() positional args must each be a column selector or a "
                        f"{{name: Column}} dict when passing multiple mutations, got {type(arg)!r}"
                    ),
                )

            collisions = set(arg_map) & set(merged)

            if collisions:
                raise PySparkValueError(
                    message=(
                        f"withColumns() column name collision on {sorted(collisions)!r} -- "
                        "more than one mutation in this call targets the same column name; "
                        "rename one of them or combine them into a single mutation instead "
                        "of letting one silently overwrite the other."
                    ),
                )

            merged.update(arg_map)

        return original(self, merged)

    return withColumns_patched
