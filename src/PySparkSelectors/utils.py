"""non-column selector imports"""

from collections.abc import Callable, Sequence
from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.column import Column

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
# return selector  names
# create  column names for functions that require columns to have them.
# --------------------------------------------------


def _resolve_selector_names(df: DataFrame, args: Sequence[Any]) -> list[Any]:
    """Turn any `BaseSelector` in `args` into the plain column names it matches.

    Shared resolver #1 -- used by the fx overrides that only need bare names
    (`groupBy`, `drop`), the same way `select_patched` originally resolved
    selectors for `select`.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        The dataframe to resolve any selector's matched column names against.
    args : sequence of Any
        The positional arguments originally passed to the overridden pyspark
        method (e.g. `DataFrame.groupBy`'s `*cols`). Any `BaseSelector` is
        expanded to the names it matches; anything else passes through
        untouched.

    Returns
    -------
    list of Any
        `args`, with every `BaseSelector` expanded into its matched column
        name strings, in order.

    Examples
    --------
    >>> _resolve_selector_names(df, (by_dtype([T.StringType]),))
    ['string_col', 'data_column_6']
    """
    from .models import BaseSelector

    resolved = []

    for a in args:
        if isinstance(a, BaseSelector):
            # a resolved name may legally contain a literal dot (see
            # `_quote_identifier`) -- these names get handed straight to a
            # real pyspark method expecting bare name strings (`groupBy`,
            # `drop`), which -- same as calling that method directly with a
            # dotted name -- would otherwise misparse an un-escaped dot as
            # struct-path access. quote defensively here since these names
            # came from our own selector resolution, not the caller's
            # original input.
            resolved.extend(_quote_identifier(name) for name in a.resolve(df))
        else:
            resolved.append(a)

    return resolved


# --------------------------------------------------
# return selector  expressions
# create  column expressions for functions that require columns to have them.
# --------------------------------------------------


def _resolve_selector_exprs(df: DataFrame, args: Sequence[Any]) -> list[tuple[str | None, Any]]:
    """Turn any `BaseSelector` in `args` into `(name, Column expr)` pairs.

    Shared resolver #2 -- applies whatever transforms are chained onto the
    selector (``.sum()``, ``.cast(...)``, ``+1``, ``.desc()``, etc, all
    already available via `SelectorcolumnOperations`). A selector with no
    transforms resolves to plain `F.col(name)` expressions.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        The dataframe to resolve any selector's matched columns against.
    args : sequence of Any
        The positional arguments originally passed to the overridden pyspark
        method. Any `BaseSelector` is expanded to `(name, expr)` pairs for
        every column it matches; anything else passes through as
        ``(None, value)``.

    Returns
    -------
    list of tuple of (str or None, Any)
        One `(name, expr)` pair per resolved column/argument, in order. When
        a rename (`.prefix()`/`.suffix()`/`.map_alias()`) has been queued on
        the selector (`self._name_transform`), `name` is the *new* name
        computed by applying it to the originally-matched column name,
        instead of the originally-matched name itself -- harmless when a
        caller re-aliases to it (it's already the correct name), and
        required by callers like `withColumns_patched` that use `name` as
        the output column's real key.

    Examples
    --------
    >>> _resolve_selector_exprs(df, (by_dtype([T.IntegerType]) + 1,))
    [('integer_col', Column<'(integer_col + 1)'>)]
    >>> _resolve_selector_exprs(df, (by_dtype([T.IntegerType]).prefix('int_'),))
    [('int_integer_col', Column<'integer_col'>)]
    """
    from .models import BaseSelector

    pairs = []

    for a in args:
        if isinstance(a, BaseSelector):
            names = a.resolve(df)
            exprs = a.resolve_columns(df) if a.transforms else [_quoted_col(n) for n in names]

            # a queued rename (.prefix()/.suffix()/.map_alias()) is pure name
            # metadata -- apply it here to each originally-matched name to get
            # the real output name, without ever having touched the Column
            # expression itself (so it can't be corrupted by whatever value
            # transform text, e.g. .cast(...)/arithmetic, is baked into `expr`).
            if a._name_transform is not None:
                pairs.extend((a._name_transform(name), expr) for name, expr in zip(names, exprs, strict=True))
            else:
                pairs.extend(zip(names, exprs, strict=True))
        else:
            pairs.append((None, a))

    return pairs


# --------------------------------------------------------------------------------
# support  for edge  case column selection, including support for <.>, <`>  in column names
# --------------------------------------------------------------------------------


def _quote_identifier(name: str) -> str:
    """Backtick-quote `name` if needed so pyspark treats it as one flat identifier.

    Parameters
    ----------
    name : str
        A real, flat column name (as returned by `BaseSelector.resolve`),
        which may legally contain a literal dot or backtick.

    Returns
    -------
    str
        `name` unchanged if it has no dot/backtick, otherwise the
        backtick-quoted form (with any embedded backtick doubled, Spark
        SQL's own backtick-escaping convention).

    Examples
    --------
    >>> _quote_identifier('integer_col')
    'integer_col'
    >>> _quote_identifier('meta.source')
    '`meta.source`'
    """

    if "." in name or "`" in name:
        escaped = name.replace("`", "``")
        return f"`{escaped}`"

    return name


def _quoted_col(name: str) -> Column:
    """Build `F.col(name)`, backtick-quoting `name` if needed to keep it flat.

    Parameters
    ----------
    name : str
        A real, flat column name (as returned by `BaseSelector.resolve`),
        which may legally contain a literal dot or backtick.

    Returns
    -------
    pyspark.sql.Column
        `F.col` built from `_quote_identifier(name)`, so pyspark always
        resolves it as one flat column name instead of a struct-path
        reference.

    Examples
    --------
    >>> _quoted_col('integer_col')
    Column<'integer_col'>
    >>> _quoted_col('meta.source')
    Column<'meta.source'>
    """

    return F.col(_quote_identifier(name))


# --------------------------------------------------
# allow for checking for columns across different types of pyspark dataframes
# spark can have multiple different types this  handles the changes.
# --------------------------------------------------


def _get_column_method(method_name: str) -> Callable[..., Any]:
    """Look up a `Column` instance method, checking classic then Spark Connect.

    Used wherever a method is looked up purely for introspection (its
    `__doc__`, or an existence/callability check) rather than to be bound and
    called directly -- e.g. by `_make_column_only_wrapper`/
    `_make_dispatching_column_wrapper` when building a chained selector
    method. Falls back to Spark Connect's `Column` when the classic
    `pyspark.sql.Column` doesn't define `method_name` at all, so a
    Connect-only method still gets picked up instead of silently never
    getting a selector-chainable wrapper.

    Parameters
    ----------
    method_name : str
        Name of the `Column` instance method to look up.

    Returns
    -------
    callable
        The method, from whichever of classic/Connect `Column` actually
        defines it (classic checked first).

    Raises
    ------
    AttributeError
        If neither classic nor (when available) Spark Connect's `Column`
        defines `method_name`.
    """
    if hasattr(Column, method_name):
        return getattr(Column, method_name)
    return getattr(_ConnectColumn, method_name)


# --------------------------------------------------
# protect against re-overriding the same functions on packages initialization.
# --------------------------------------------------


def _true_original(cls: type, method_name: str) -> Callable[..., Any]:
    """Re-run-safe replacement for ``getattr(cls, method_name)`` when capturing an original method.

    Every override in this notebook does ``_original_x = <the real pyspark
    method>`` once, then reassigns ``SomeClass.method = patched_version``.
    That's fine the first time a cell runs -- but if the SAME cell gets
    re-run (very normal during interactive experimentation),
    ``getattr(cls, method_name)`` on the second run fetches the
    *already-patched* version instead of the real pyspark one, and the newly
    defined patched function ends up calling itself forever the moment it's
    invoked -- causing infinite recursion.

    Parameters
    ----------
    cls : type
        The class the method lives on (e.g. `pyspark.sql.DataFrame`,
        `pyspark.sql.group.GroupedData`).
    method_name : str
        The name of the method to fetch the true original implementation of.

    Returns
    -------
    callable
        The true, never-patched method. Every patched function this returns
        the original for gets tagged with ``._cs_original`` pointing at the
        true, never-patched method -- so on a re-run, this peels back
        through any already-applied patch layer(s) and always hands back the
        one real pyspark implementation, no matter how many times the cell
        is executed.
    """
    current = getattr(cls, method_name)

    return getattr(current, "_cs_original", current)
