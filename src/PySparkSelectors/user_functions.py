import re
from collections.abc import Sequence
from typing import Any

import pyspark.sql.types as T
from pyspark.sql import DataFrame
from pyspark.sql.column import Column

from .models import BaseSelector, DTypeSelector, IndexSelector, RegexSelector

_TIMESTAMP_TYPES = tuple(t for t in (T.TimestampType, getattr(T, "TimestampNTZType", None)) if t is not None)


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


# --------------------------------------------------------------------------------
# the layout of functionality is that we have 3 column selector classes that resolve our columns.
# dtype_selector, index_selector, and regex_selector
# you will see that each of these above classes has a 1 main function that is used for all selection
# by_dtype, by_index, and matches.
# all other column selector  functions are built on top of these 3 main functions.
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# BY_INDEX SECTION
# create main by_index function that is used for all index based column selection
# --------------------------------------------------------------------------------


def by_index(*indexes: int | range, strict_index_bounds: bool = True, require_col_match: bool = True) -> IndexSelector:
    """Select columns by positional index.

    Parameters
    ----------
    *indexes : int or range
        One or more positional column indexes. Negative ints follow
        Python's own indexing convention (``-1`` = last column). A `range`
        object covers a span in one shot (e.g. ``range(1, 4)`` for columns 1-3).
    strict_index_bounds : bool, default True
        If `True`, an out-of-bounds index raises
        `pyspark.errors.PySparkValueError`. If `False`, out-of-bounds
        indexes are silently skipped.
    require_col_match : bool, default True
        If `True` (default), resolving this selector against a dataframe
        with zero matching columns raises `pyspark.errors.PySparkValueError`.
        Set to `False` to opt out and allow a zero-column match instead
        (only reachable when `strict_index_bounds` is also `False`).

    Returns
    -------
    IndexSelector
        A selector matching the column(s) at the requested position(s),
        composable with ``~ - & ^ |`` and chainable with arithmetic/spark
        functions.

    Examples
    --------
    >>> df.select(by_index(0, 1))
    DataFrame[...]
    >>> df.select(by_index(range(1, 4), -1))
    DataFrame[...]
    """
    return IndexSelector(indexes, strict_index_bounds=strict_index_bounds, require_col_match=require_col_match)


# --------------------------------------------------------------------------------
# create first column selector from by_index function
# --------------------------------------------------------------------------------


def first(require_col_match: bool = True) -> IndexSelector:
    """Select the first column.

    Parameters
    ----------
    require_col_match : bool, default True
        Forwarded to `by_index`. Set to `False` to opt out of the
        zero-match error and allow no matching columns.

    Returns
    -------
    IndexSelector
        Equivalent to ``by_index(0)``.

    Examples
    --------
    >>> df.select(first())
    DataFrame[...]
    """
    return by_index(0, require_col_match=require_col_match)


# --------------------------------------------------------------------------------
# create last column selector from  by_dtype function
# --------------------------------------------------------------------------------


def last(require_col_match: bool = True) -> IndexSelector:
    """Select the last column.

    Parameters
    ----------
    require_col_match : bool, default True
        Forwarded to `by_index`. Set to `False` to opt out of the
        zero-match error and allow no matching columns.

    Returns
    -------
    IndexSelector
        Equivalent to ``by_index(-1)``.

    Examples
    --------
    >>> df.select(last())
    DataFrame[...]
    """
    return by_index(-1, require_col_match=require_col_match)


# --------------------------------------------------------------------------------
# DTYPE SECTION
# create main by_dtype function that is used for all dtype based column selection
# --------------------------------------------------------------------------------


def by_dtype(dtypes: type | Sequence[type], require_col_match: bool = True) -> DTypeSelector:
    """Select columns whose real Spark SQL type is (or is a subclass of) any of `dtypes`.

    Parameters
    ----------
    dtypes : type or sequence of type
        One or more `pyspark.sql.types.DataType` subclasses to match, e.g.
        ``[T.IntegerType, T.StringType]``. A category base class (e.g.
        ``T.NumericType``) matches every concrete subclass of it too (see
        `DTypeSelector`).
    require_col_match : bool, default True
        If `True` (default), resolving this selector against a dataframe
        with zero matching columns raises `pyspark.errors.PySparkValueError`.
        Set to `False` to opt out and allow a zero-column match instead.

    Returns
    -------
    DTypeSelector
        A selector matching every column whose real datatype is in `dtypes`,
        composable with ``~ - & ^ |`` and chainable with arithmetic/spark
        functions (e.g. ``.cast(...)``, ``.sum()``, ``+ 1``).

    Examples
    --------
    >>> df.select(by_dtype([T.IntegerType, T.DoubleType]))
    DataFrame[...]
    >>> df.filter(by_dtype([T.IntegerType]) > 0)
    DataFrame[...]
    >>> df.select(by_dtype([T.BinaryType], require_col_match=False))
    DataFrame[...]
    """
    return DTypeSelector(dtypes, require_col_match=require_col_match)


# --------------------------------------------------------------------------------
# create string column selector from  by_dtype function
# --------------------------------------------------------------------------------


def string(require_col_match: bool = True) -> DTypeSelector:
    """Select every `pyspark.sql.types.StringType` column.

    Parameters
    ----------
    require_col_match : bool, default True
        Forwarded to `by_dtype`. Set to `False` to opt out of the
        zero-match error and allow no matching columns.

    Returns
    -------
    DTypeSelector
        Equivalent to ``by_dtype([T.StringType])``.

    Examples
    --------
    >>> df.select(string())
    DataFrame[...]
    """
    return by_dtype([T.StringType], require_col_match=require_col_match)


# --------------------------------------------------------------------------------
# create boolean column selector from  by_dtype function
# --------------------------------------------------------------------------------


def boolean(require_col_match: bool = True) -> DTypeSelector:
    """Select every `pyspark.sql.types.BooleanType` column.

    Parameters
    ----------
    require_col_match : bool, default True
        Forwarded to `by_dtype`. Set to `False` to opt out of the
        zero-match error and allow no matching columns.

    Returns
    -------
    DTypeSelector
        Equivalent to ``by_dtype([T.BooleanType])``.

    Examples
    --------
    >>> df.select(boolean())
    DataFrame[...]
    """
    return by_dtype([T.BooleanType], require_col_match=require_col_match)


# --------------------------------------------------------------------------------
# create binary column selector from  by_dtype function
# --------------------------------------------------------------------------------


def binary(require_col_match: bool = True) -> DTypeSelector:
    """Select every `pyspark.sql.types.BinaryType` column.

    Parameters
    ----------
    require_col_match : bool, default True
        Forwarded to `by_dtype`. Set to `False` to opt out of the
        zero-match error and allow no matching columns.

    Returns
    -------
    DTypeSelector
        Equivalent to ``by_dtype([T.BinaryType])``.

    Examples
    --------
    >>> df.select(binary())
    DataFrame[...]
    """
    return by_dtype([T.BinaryType], require_col_match=require_col_match)


# --------------------------------------------------------------------------------
# create integer column selector from  by_dtype function
# --------------------------------------------------------------------------------


def integer(require_col_match: bool = True) -> DTypeSelector:
    """Select every fixed-width integer column.

    Parameters
    ----------
    require_col_match : bool, default True
        Forwarded to `by_dtype`. Set to `False` to opt out of the
        zero-match error and allow no matching columns.

    Returns
    -------
    DTypeSelector
        Equivalent to ``by_dtype([T.IntegralType])`` -- matches
        `ByteType`/`ShortType`/`IntegerType`/`LongType`, pyspark's real
        integer type category.

    Examples
    --------
    >>> df.select(integer())
    DataFrame[...]
    """
    return by_dtype([T.IntegralType], require_col_match=require_col_match)


# --------------------------------------------------------------------------------
# create float column selector from  by_dtype function
# --------------------------------------------------------------------------------


def floats(require_col_match: bool = True) -> DTypeSelector:
    """Select every floating-point column (excluding `T.DecimalType`).

    Parameters
    ----------
    require_col_match : bool, default True
        Forwarded to `by_dtype`. Set to `False` to opt out of the
        zero-match error and allow no matching columns.

    Returns
    -------
    DTypeSelector
        Equivalent to ``by_dtype([T.FloatType, T.DoubleType])``.

    Examples
    --------
    >>> df.select(floats())
    DataFrame[...]
    """
    return by_dtype([T.FloatType, T.DoubleType], require_col_match=require_col_match)


# --------------------------------------------------------------------------------
# create numeric column selector from  by_dtype function
# --------------------------------------------------------------------------------


def numeric(require_col_match: bool = True) -> DTypeSelector:
    """Select every numeric column (integer, floating-point, or decimal).

    Notes
    -----
    Unlike a plain dtype-string match, this uses pyspark's real
    `T.NumericType` category, which includes `T.DecimalType` -- decimal
    columns are matched here even though their dtype string is parametrized
    per column (e.g. ``'decimal(10,2)'``).

    Parameters
    ----------
    require_col_match : bool, default True
        Forwarded to `by_dtype`. Set to `False` to opt out of the
        zero-match error and allow no matching columns.

    Returns
    -------
    DTypeSelector
        Equivalent to ``by_dtype([T.NumericType])``.

    Examples
    --------
    >>> df.select(numeric())
    DataFrame[...]
    """
    return by_dtype([T.NumericType], require_col_match=require_col_match)


# --------------------------------------------------------------------------------
# create date column selector from  by_dtype function
# --------------------------------------------------------------------------------


def date(require_col_match: bool = True) -> DTypeSelector:
    """Select every `pyspark.sql.types.DateType` column.

    Parameters
    ----------
    require_col_match : bool, default True
        Forwarded to `by_dtype`. Set to `False` to opt out of the
        zero-match error and allow no matching columns.

    Returns
    -------
    DTypeSelector
        Equivalent to ``by_dtype([T.DateType])``.

    Examples
    --------
    >>> df.select(date())
    DataFrame[...]
    """
    return by_dtype([T.DateType], require_col_match=require_col_match)


# --------------------------------------------------------------------------------
# create datetime column selector from  by_dtype function
# --------------------------------------------------------------------------------


def datetime_(require_col_match: bool = True) -> DTypeSelector:
    """Select every timestamp column (`T.TimestampType` and, if available, `T.TimestampNTZType`).

    Parameters
    ----------
    require_col_match : bool, default True
        Forwarded to `by_dtype`. Set to `False` to opt out of the
        zero-match error and allow no matching columns.

    Returns
    -------
    DTypeSelector
        Equivalent to ``by_dtype([T.TimestampType, T.TimestampNTZType])`` on
        Spark versions where `T.TimestampNTZType` exists, otherwise just
        ``by_dtype([T.TimestampType])``.

    Examples
    --------
    >>> df.select(datetime_())
    DataFrame[...]
    """
    return by_dtype(list(_TIMESTAMP_TYPES), require_col_match=require_col_match)


# --------------------------------------------------------------------------------
# create temporal column selector from  by_dtype function
# --------------------------------------------------------------------------------


def temporal(require_col_match: bool = True) -> DTypeSelector:
    """Select every date or timestamp column.

    pyspark has no single common base class covering `T.DateType` and every
    timestamp type, so this is a self-defined grouping rather than one
    category class.

    Parameters
    ----------
    require_col_match : bool, default True
        Forwarded to `by_dtype`. Set to `False` to opt out of the
        zero-match error and allow no matching columns.

    Returns
    -------
    DTypeSelector
        Equivalent to ``by_dtype([T.DateType, T.TimestampType, T.TimestampNTZType])``
        on Spark versions where `T.TimestampNTZType` exists, otherwise
        ``by_dtype([T.DateType, T.TimestampType])``.

    Examples
    --------
    >>> df.select(temporal())
    DataFrame[...]
    """
    return by_dtype([T.DateType, *_TIMESTAMP_TYPES], require_col_match=require_col_match)


# --------------------------------------------------------------------------------
# COLUMN NAME BASED SELECTORS SECTION
# create matches function used for all COLUMN NAME BASED SELECTORS based column selection
# entirely done using regular expressions
# --------------------------------------------------------------------------------


def matches(pattern: str, require_col_match: bool = True) -> RegexSelector:
    """Select columns whose name matches a regex pattern.

    Parameters
    ----------
    pattern : str
        A regular expression, matched against each column name via
        `re.search`.
    require_col_match : bool, default True
        If `True` (default), resolving this selector against a dataframe
        with zero matching columns raises `pyspark.errors.PySparkValueError`.
        Set to `False` to opt out and allow a zero-column match instead.

    Returns
    -------
    RegexSelector
        A selector matching every column whose name matches `pattern`,
        composable with ``~ - & ^ |`` and chainable with arithmetic/spark
        functions.

    Examples
    --------
    >>> df.select(matches(r"_id$"))
    DataFrame[...]
    >>> df.select(matches(r"_id$").upper())
    DataFrame[...]
    """
    return RegexSelector(pattern, require_col_match=require_col_match)


# --------------------------------------------------------------------------------
# create starts_with column selector from matches function
# --------------------------------------------------------------------------------


def starts_with(*prefixes: str, require_col_match: bool = True) -> RegexSelector:
    """Select columns whose name starts with any of `prefixes`.

    Parameters
    ----------
    *prefixes : str
        One or more literal prefixes to match against the start of each
        column name.
    require_col_match : bool, default True
        Forwarded to `matches`. Set to `False` to opt out of the
        zero-match error and allow no matching columns.

    Returns
    -------
    RegexSelector
        Equivalent to ``matches(r"^(prefix1|prefix2|...)")``.

    Examples
    --------
    >>> df.select(starts_with('project_'))
    DataFrame[...]
    """
    return matches(r"^(" + "|".join(re.escape(p) for p in prefixes) + r")", require_col_match=require_col_match)


# --------------------------------------------------------------------------------
# create ends_with column selector from matches function
# --------------------------------------------------------------------------------


def ends_with(*suffixes: str, require_col_match: bool = True) -> RegexSelector:
    """Select columns whose name ends with any of `suffixes`.

    Parameters
    ----------
    *suffixes : str
        One or more literal suffixes to match against the end of each
        column name.
    require_col_match : bool, default True
        Forwarded to `matches`. Set to `False` to opt out of the
        zero-match error and allow no matching columns.

    Returns
    -------
    RegexSelector
        Equivalent to ``matches(r"(suffix1|suffix2|...)$")``.

    Examples
    --------
    >>> df.select(ends_with('_id'))
    DataFrame[...]
    """
    return matches(r"(" + "|".join(re.escape(s) for s in suffixes) + r")$", require_col_match=require_col_match)


# --------------------------------------------------------------------------------
# create contains column selector from matches function
# --------------------------------------------------------------------------------


def contains(*substrings: str, require_col_match: bool = True) -> RegexSelector:
    """Select columns whose name contains any of `substrings`.

    Parameters
    ----------
    *substrings : str
        One or more literal substrings to search for anywhere in each
        column name.
    require_col_match : bool, default True
        Forwarded to `matches`. Set to `False` to opt out of the
        zero-match error and allow no matching columns.

    Returns
    -------
    RegexSelector
        Equivalent to ``matches(r"(substring1|substring2|...)")``.

    Examples
    --------
    >>> df.select(contains('project'))
    DataFrame[...]
    """
    return matches("(" + "|".join(re.escape(s) for s in substrings) + ")", require_col_match=require_col_match)


# --------------------------------------------------------------------------------
# create by_name column selector from matches function
# --------------------------------------------------------------------------------


def by_name(*names: str, require_col_match: bool = True) -> RegexSelector:
    """Select columns whose name exactly matches any of `names`.

    Parameters
    ----------
    *names : str
        One or more exact column names to select.
    require_col_match : bool, default True
        Forwarded to `matches`. Set to `False` to opt out of the
        zero-match error and allow no matching columns.

    Returns
    -------
    RegexSelector
        Equivalent to ``matches(r"^(name1|name2|...)$")``.

    Examples
    --------
    >>> df.select(by_name('string_col', 'data_column_5'))
    DataFrame[...]
    """
    return matches(r"^(" + "|".join(re.escape(n) for n in names) + r")$", require_col_match=require_col_match)


# --------------------------------------------------------------------------------
# create exclude column selector from matches function
# --------------------------------------------------------------------------------


def exclude(*names: str, require_col_match: bool = True) -> RegexSelector:
    """Select every column EXCEPT `names`.

    Parameters
    ----------
    *names : str
        One or more exact column names to exclude.
    require_col_match : bool, default True
        Forwarded to both `all` and `by_name`. Set to `False` to opt out of
        the zero-match error and allow no matching columns (e.g. when
        excluding every column on the dataframe).

    Returns
    -------
    RegexSelector
        Equivalent to ``all() - by_name(*names)``.

    Examples
    --------
    >>> df.select(exclude('string_col'))
    DataFrame[...]
    """
    return all(require_col_match=require_col_match) - by_name(*names, require_col_match=require_col_match)


# --------------------------------------------------------------------------------
# create alphabetical column selector from matches function
# --------------------------------------------------------------------------------


def alpha(require_col_match: bool = True) -> RegexSelector:
    """Select columns whose name consists only of alphabetic characters.

    Parameters
    ----------
    require_col_match : bool, default True
        Forwarded to `matches`. Set to `False` to opt out of the
        zero-match error and allow no matching columns.

    Returns
    -------
    RegexSelector
        Equivalent to ``matches(r"^[a-zA-Z]+$")``.

    Examples
    --------
    >>> df.select(alpha())
    DataFrame[...]
    """
    return matches(r"^[a-zA-Z]+$", require_col_match=require_col_match)


# --------------------------------------------------------------------------------
# create alphanumeric column selector from matches function
# --------------------------------------------------------------------------------


def alphanumeric(require_col_match: bool = True) -> RegexSelector:
    """Select columns whose name consists only of letters and/or digits.

    Parameters
    ----------
    require_col_match : bool, default True
        Forwarded to `matches`. Set to `False` to opt out of the
        zero-match error and allow no matching columns.

    Returns
    -------
    RegexSelector
        Equivalent to ``matches(r"^[a-zA-Z0-9]+$")``.

    Examples
    --------
    >>> df.select(alphanumeric())
    DataFrame[...]
    """
    return matches(r"^[a-zA-Z0-9]+$", require_col_match=require_col_match)


# --------------------------------------------------------------------------------
# create all column selector from matches function
# --------------------------------------------------------------------------------


def all(require_col_match: bool = True) -> RegexSelector:
    """Select every column.

    Parameters
    ----------
    require_col_match : bool, default True
        Whether resolving raises `pyspark.errors.PySparkValueError` when the
        dataframe has zero columns. Forwarded to `RegexSelector`.

    Returns
    -------
    RegexSelector
        A selector matching every column name (built from a match-everything
        regex, since `IndexSelector` has no "all of them" spelling without
        already knowing the dataframe's column count).

    Examples
    --------
    >>> df.select(all())
    DataFrame[...]
    """
    return RegexSelector(r".*", require_col_match=require_col_match)


# --------------------------------------------------------------------------------
# create boolean to detect column selectors.
# --------------------------------------------------------------------------------


def is_selector(x: Any) -> bool:
    """Return whether `x` is one of this framework's column selector objects.

    Parameters
    ----------
    x : Any
        The value to check -- e.g. the result of `by_dtype`, `by_index`,
        `matches`, or any of those combined with ``~ - & ^ |``.

    Returns
    -------
    bool
        `True` if `x` is a `BaseSelector` instance (or subclass instance),
        `False` for anything else (plain column names, `pyspark.sql.Column`
        results, literals, etc).

    Examples
    --------
    >>> is_selector(by_dtype([T.IntegerType]))
    True
    >>> is_selector('string_col')
    False
    """
    return isinstance(x, BaseSelector)
