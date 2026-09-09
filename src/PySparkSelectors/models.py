import functools
import operator
import re
from collections.abc import Callable, Sequence
from typing import Any, Union

import pyspark.sql.types as T
from pyspark.errors import PySparkNotImplementedError, PySparkTypeError, PySparkValueError
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.column import Column

from .utils import (
    _get_column_method,
    _quoted_col,
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

# --------------------------------------------------------------------------------
# Make column selectors  allow column level  transformations
# including support for arithmetic, comparison, and spark functions
# --------------------------------------------------------------------------------


class SelectorcolumnOperations:
    """Mixin providing per-column transform chaining for selector objects.

    Every selector class (`DTypeSelector`, `IndexSelector`, `RegexSelector`, and
    any future selector) inherits from this class (via `BaseSelector`) so that
    arithmetic operators (``+``, ``-``, ``*``, ...), comparison operators
    (``==``, ``>``, ...), and any function available on
    `pyspark.sql.functions` (``.upper()``, ``.sum()``, ``.cast(...)``, ...) can
    be chained directly onto a selector, e.g. ``by_dtype([T.IntegerType]) + 1`` or
    ``by_dtype([T.StringType]).upper()``. Each chained call does not mutate the
    selector in place -- it returns a new, immutable copy with the extra
    transform appended, so the original selector is always safe to reuse.

    Parameters
    ----------
    transforms : list of callable, optional
        A list of one-argument functions, each accepting and returning a
        `pyspark.sql.Column`, representing the transform pipeline already
        chained onto this selector. Defaults to an empty list when not given.

    Attributes
    ----------
    transforms : list of callable
        The transform pipeline described above. Applied, in order, to every
        matched column's `pyspark.sql.Column` expression by `resolve_columns`.

    Examples
    --------
    >>> by_dtype([T.IntegerType]) + 1
    <DTypeSelector ...>
    >>> by_dtype([T.StringType]).upper()
    <DTypeSelector ...>
    """

    def __init__(self, transforms: list[Callable[[Column], Column]] | None = None) -> None:
        """Initialize the transform pipeline.

        Parameters
        ----------
        transforms : list of callable, optional
            Existing transform pipeline to start from. Defaults to an empty
            list when not given.
        """
        self.transforms = transforms or []

        # tracks a pending rename (`.prefix()`/`.suffix()`/`.map_alias()`) as
        # pure name metadata -- a `str -> str` function applied to the
        # originally-matched column name -- kept completely separate from
        # `self.transforms` (the *value* pipeline). This is what
        # `_resolve_selector_exprs` uses to compute each matched column's
        # final output name; it deliberately never touches the `Column`
        # expression itself mid-chain, so it can't be corrupted by whatever
        # value transform text (`.cast(...)`, arithmetic, ...) precedes it,
        # and it never appears in `self.transforms`, so `~`/`&`/`|`'s
        # "does this selector already have a transform chained onto it?"
        # filter-condition check isn't tripped by a rename alone -- see
        # "prefix / suffix and map name fx's" section.
        self._name_transform: Callable[[str], str] | None = None

    # --------------------------------------------------
    # utility used by every operator/function
    # --------------------------------------------------

    def _copy(self, transform: Callable[[Column], Column]) -> "SelectorcolumnOperations":
        """Return an immutable copy of this selector with one more transform appended.

        Parameters
        ----------
        transform : callable
            A one-argument function accepting and returning a
            `pyspark.sql.Column`, appended to the end of the copy's transform
            pipeline.

        Returns
        -------
        SelectorcolumnOperations
            A new instance of the same concrete class, with all of this
            instance's attributes copied over and `transform` appended to
            `transforms`. `self` itself is left unmodified.
        """
        # allows to immutably create new versions of the class depending on the transformation that occurs.
        new = self.__class__.__new__(self.__class__)

        new.__dict__.update(self.__dict__)

        new.transforms = self.transforms + [transform]

        return new

    def _with_name_transform(self, name_transform: Callable[[str], str]) -> "SelectorcolumnOperations":
        """Return an immutable copy of this selector with a rename queued.

        Backs `.prefix()`/`.suffix()`/`.map_alias()`. Unlike `_copy`, this
        never touches `self.transforms` (the value pipeline) or the
        selector's `Column` expressions at all -- it only records how the
        *matched column name* should be transformed once real names are
        known (in `_resolve_selector_exprs`), composing with any
        rename already queued so multiple renames chain in order
        (e.g. ``.prefix('a_').suffix('_b')``).

        Parameters
        ----------
        name_transform : callable
            A one-argument function accepting a column's current name (`str`)
            and returning its new name (`str`).

        Returns
        -------
        SelectorcolumnOperations
            A new instance of the same concrete class, with all of this
            instance's attributes copied over and `_name_transform` set to
            `name_transform` composed after any already queued. `self` itself
            is left unmodified.
        """
        new = self.__class__.__new__(self.__class__)

        new.__dict__.update(self.__dict__)

        previous = self._name_transform

        if previous is None:
            new._name_transform = name_transform
        else:
            new._name_transform = lambda name: name_transform(previous(name))

        return new

    def prefix(self, prefix: str) -> "SelectorcolumnOperations":
        """Queue a rename prepending `prefix` to every matched column's name.

        Mirrors polars' `Expr.prefix`, but implemented at the selector level
        (via `_with_name_transform`) rather than by dispatching to
        `pyspark.sql.Column.prefix` -- this keeps the rename as pure name
        metadata instead of an early `.alias(...)` baked into the value
        expression, so it composes correctly with any value transform
        (`.cast(...)`, arithmetic, ...) chained before or after it, and does
        not affect `~`/`&`/`|`'s filter-condition detection.

        Parameters
        ----------
        prefix : str
            The string to prepend to each matched column's current name.

        Returns
        -------
        SelectorcolumnOperations
            A new selector with the rename queued.

        Examples
        --------
        >>> df.select(by_dtype([T.IntegerType]).prefix('int_'))
        DataFrame[...]
        >>> df.select((by_dtype([T.IntegerType]) + 1).prefix('int_'))
        DataFrame[...]
        """
        return self._with_name_transform(lambda name: f"{prefix}{name}")

    def suffix(self, suffix: str) -> "SelectorcolumnOperations":
        """Queue a rename appending `suffix` to every matched column's name.

        Mirrors polars' `Expr.suffix`. See `prefix` for why this is
        implemented at the selector level instead of dispatching to
        `pyspark.sql.Column.suffix`.

        Parameters
        ----------
        suffix : str
            The string to append to each matched column's current name.

        Returns
        -------
        SelectorcolumnOperations
            A new selector with the rename queued.

        Examples
        --------
        >>> df.select(by_dtype([T.IntegerType]).suffix('_int'))
        DataFrame[...]
        """
        return self._with_name_transform(lambda name: f"{name}{suffix}")

    def map_alias(self, func: Callable[[str], str]) -> "SelectorcolumnOperations":
        """Queue a rename computed by applying `func` to each matched column's name.

        Mirrors polars' `Expr.map_alias`. `.prefix()`/`.suffix()` are both
        just `.map_alias()` with a prepend/append built in. See `prefix` for
        why this is implemented at the selector level instead of dispatching
        to `pyspark.sql.Column.map_alias`.

        Parameters
        ----------
        func : callable
            A one-argument function accepting a column's current name (`str`)
            and returning its new name (`str`).

        Returns
        -------
        SelectorcolumnOperations
            A new selector with the rename queued.

        Examples
        --------
        >>> df.select(by_dtype([T.StringType]).map_alias(lambda n: n.upper()))
        DataFrame[...]
        """
        return self._with_name_transform(func)

    # --------------------------------------------------
    # operator decorator
    # --------------------------------------------------

    @staticmethod
    def operator_wrapper(op: Callable[[Any, Any], Any]) -> Callable[..., "SelectorcolumnOperations"]:
        """Build a dunder-method implementation that chains a Python operator.

        Used to define every arithmetic/comparison dunder on this class
        (``__add__``, ``__gt__``, ...) in one line each, e.g.
        ``__add__ = operator_wrapper(operator.add)``.

        Parameters
        ----------
        op : callable
            A 2-argument function from the standard library `operator` module
            (or anything with the same shape), applied as ``op(column, other)``
            to each matched column's `pyspark.sql.Column` expression once the
            selector is resolved against a real dataframe.

        Returns
        -------
        callable
            A dunder-method-shaped function ``overload(self, other)`` that
            appends ``lambda c: op(c, other)`` to the selector's transform
            pipeline via `_copy`, and returns the resulting new selector.

        Examples
        --------
        >>> __add__ = operator_wrapper(operator.add)
        >>> by_dtype([T.IntegerType]) + 1
        <DTypeSelector ...>
        """

        def overload(self, other):

            return self._copy(lambda c: op(c, other))

        return overload

    # --------------------------------------------------
    # spark function decorator
    # --------------------------------------------------

    @staticmethod
    def spark_wrapper(spark_func: Callable[..., Column]) -> Callable[..., "SelectorcolumnOperations"]:
        """Build a chainable method that wraps one `pyspark.sql.functions` function.

        Used both explicitly (e.g. `cast`) and automatically, via the
        ``for fx in dir(F): ...`` loop below, to make every callable in
        `pyspark.sql.functions` available as a chained method on any selector,
        e.g. ``by_dtype([T.StringType]).upper()`` (wrapping `pyspark.sql.functions.upper`).

        Parameters
        ----------
        spark_func : callable
            A function from `pyspark.sql.functions` (or any function with the
            same shape, e.g. ``lambda c, dtype: c.cast(dtype)``) whose first
            argument is the `pyspark.sql.Column` to operate on.

        Returns
        -------
        callable
            A chainable method ``wrapper(self, *args, **kwargs)`` that appends
            ``lambda c: spark_func(c, *args, **kwargs)`` to the selector's
            transform pipeline via `_copy`, and returns the resulting new
            selector. Carries `spark_func`'s own name/docstring (via
            `functools.wraps`) so ``help(some_selector.upper)`` shows the real
            `pyspark.sql.functions.upper` documentation.

        Examples
        --------
        >>> upper = spark_wrapper(F.upper)
        >>> by_dtype([T.StringType]).upper()
        <DTypeSelector ...>
        """

        @functools.wraps(spark_func)
        def wrapper(self, *args, **kwargs):

            return self._copy(lambda c: spark_func(c, *args, **kwargs))

        return wrapper

    def _make_dispatching_column_wrapper(method_name: str) -> Callable[..., Column]:
        """Build a wrapper that dispatches to `Column` or `pyspark.sql.functions` by argument type.

        Resolves the `F`/`Column` name-collision. this is where there are functions
        in `pyspark.sql.functions.<method_name>` and `pyspark.sql.Column.<method_name>`
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

    def resolve_columns(self, df: DataFrame) -> list[Column]:
        """Resolve this selector's matched columns into transformed `Column` expressions.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            The dataframe to resolve the selector's matched column names
            against, and to build each `pyspark.sql.Column` expression from.

        Returns
        -------
        list of pyspark.sql.Column
            One `pyspark.sql.Column` expression per matched column, in
            `resolve`'s order, with every transform in `self.transforms`
            applied in sequence (e.g. ``by_dtype([T.IntegerType]) + 1`` returns
            ``F.col(name) + 1`` for every matched int column).

        Examples
        --------
        >>> by_dtype([T.IntegerType]).resolve_columns(df)
        [Column<'(integer_col + 0)'>, ...]
        """
        cols = [_quoted_col(c) for c in self.resolve(df)]

        for transform in self.transforms:
            cols = [transform(c) for c in cols]

        return cols

    # --------------------------------------------------
    # apply transforms
    # --------------------------------------------------

    __add__ = operator_wrapper(operator.add)
    __sub__ = operator_wrapper(operator.sub)
    __mul__ = operator_wrapper(operator.mul)
    __truediv__ = operator_wrapper(operator.truediv)
    __mod__ = operator_wrapper(operator.mod)
    __pow__ = operator_wrapper(operator.pow)
    # __len__          = operator_wrapper(operator.len)
    __floordiv__ = operator_wrapper(operator.floordiv)

    __eq__ = operator_wrapper(operator.eq)
    __gt__ = operator_wrapper(operator.gt)
    __ge__ = operator_wrapper(operator.ge)
    __lt__ = operator_wrapper(operator.lt)
    __le__ = operator_wrapper(operator.le)
    __ne__ = operator_wrapper(operator.ne)

    # __contains__    = operator_wrapper(operator.contains)
    # __and__         = operator_wrapper(operator.and)
    # __or__          = operator_wrapper(operator.or)

    __getitem__ = operator_wrapper(operator.getitem)
    # __setitem__     = operator_wrapper(operator.setitem)
    # __delitem__     = operator_wrapper(operator.delitem)
    __invert__ = operator_wrapper(operator.invert)


# --------------------------------------------------------------------------------
# Allow support for column selector level operations
# --------------------------------------------------------------------------------


class SelectorSelectionOperations:
    """Mixin providing set-algebra and filter-condition operators for selectors.

    Every selector class (`DTypeSelector`, `IndexSelector`, `RegexSelector`,
    and any future selector) inherits from this class (via `BaseSelector`) so
    that ``~ - & ^ |`` compose selectors together, e.g.
    ``by_dtype([T.IntegerType]) | by_dtype([T.StringType])`` (union) or
    ``~by_dtype([T.StringType])`` (every column that is NOT a string column).

    These same operators (``&``, ``|``, ``~``) switch to "expression mode"
    once a selector already has a comparison/arithmetic transform chained
    onto it (e.g. ``by_dtype([T.IntegerType]) > 0``), or the other operand isn't a
    bare selector at all (a plain `pyspark.sql.Column`). In that mode, they
    combine row-conditions for `filter`/`where`
    (e.g. ``df.filter(by_dtype([T.IntegerType]) > 0)``) instead of intersecting/
    unioning which columns are matched -- see `SelectorFilterCondition`.

    Parameters
    ----------
    resolver : callable, optional
        A one-argument function ``resolver(df) -> list[str]`` built by
        `_selector_copy` when a set operator (``~ - & ^ |``) combines two
        selectors. When present, `resolve` (defined on `BaseSelector`
        subclasses) must call this instead of its own default matching
        logic. Defaults to an empty list when not given.

    Attributes
    ----------
    _resolver : callable or list
        The resolver function described above, or an empty list when this
        selector has not been produced by a set operator.

    Examples
    --------
    >>> by_dtype([T.IntegerType]) | by_dtype([T.StringType])
    <DTypeSelector ...>
    >>> ~by_dtype([T.StringType])
    <DTypeSelector ...>
    """

    def __init__(self, resolver: Callable[[DataFrame], list[str]] | None = None) -> None:
        """Initialize the combinator resolver.

        Parameters
        ----------
        resolver : callable, optional
            Existing resolver function to start from. Defaults to an empty
            list when not given.
        """

        self._resolver = resolver or []

    def _selector_copy(self, resolver: Callable[[DataFrame], list[str]]) -> "SelectorSelectionOperations":
        """Return an immutable copy of this selector with a combined resolver.

        Parameters
        ----------
        resolver : callable
            A one-argument function ``resolver(df) -> list[str]`` implementing
            the combined set-algebra result (union, intersection, ...).

        Returns
        -------
        SelectorSelectionOperations
            A new instance of the same concrete class, with all of this
            instance's attributes copied over and `_resolver` set to
            `resolver`. `self` itself is left unmodified.
        """

        new = self.__class__.__new__(self.__class__)

        new.__dict__.update(self.__dict__)

        new._resolver = resolver

        return new

    def __and__(
        self, other: Union["BaseSelector", Column, "SelectorFilterCondition"]
    ) -> Union["SelectorSelectionOperations", "SelectorFilterCondition"]:
        """Intersect two selectors' matched columns, or AND two row conditions.

        Parameters
        ----------
        other : BaseSelector, pyspark.sql.Column, or SelectorFilterCondition
            The right-hand operand of ``&``.

        Returns
        -------
        SelectorSelectionOperations or SelectorFilterCondition
            A new selector matching only columns present in both `self` and
            `other` (set-algebra mode), or a `SelectorFilterCondition` ANDing
            the two row conditions together (expression mode -- triggered
            when `self` already has a chained transform, or `other` isn't a
            bare selector).

        Examples
        --------
        >>> by_dtype([T.DoubleType, T.IntegerType]) & by_dtype([T.DoubleType, T.StringType])
        <DTypeSelector ...>
        >>> df.filter((by_dtype([T.IntegerType]) > 0) & (by_dtype([T.IntegerType]) < 9))
        DataFrame[...]
        """

        # expression mode: `self` already has a comparison/arithmetic transform
        # chained onto it (e.g. `by_dtype([T.IntegerType]) == 0`), or `other` isn't a raw
        # selector at all (a plain pyspark Column like `F.col('x') == None`, or
        # another transformed selector). in that case `&` means "AND these row
        # conditions together", not "intersect these two selectors' matched
        # columns" -- so it's deferred into a SelectorFilterCondition instead of
        # the name-set intersection below.
        if self.transforms or not isinstance(other, BaseSelector) or getattr(other, "transforms", None):
            return SelectorFilterCondition(self, "and", other)

        left = self
        right = other

        def _resolve(df):

            right_cols = set(right.resolve(df))

            return [dim for dim in left.resolve(df) if dim in right_cols]

        return self._selector_copy(_resolve)

    def __or__(
        self, other: Union["BaseSelector", Column, "SelectorFilterCondition"]
    ) -> Union["SelectorSelectionOperations", "SelectorFilterCondition"]:
        """Union two selectors' matched columns, or OR two row conditions.

        Parameters
        ----------
        other : BaseSelector, pyspark.sql.Column, or SelectorFilterCondition
            The right-hand operand of ``|``.

        Returns
        -------
        SelectorSelectionOperations or SelectorFilterCondition
            A new selector matching every column present in `self` or
            `other` (set-algebra mode), or a `SelectorFilterCondition` ORing
            the two row conditions together (expression mode -- same trigger
            as `__and__`).

        Examples
        --------
        >>> by_dtype([T.IntegerType]) | by_dtype([T.StringType])
        <DTypeSelector ...>
        """

        # same expression-mode guard as __and__ -- see the comment there.
        if self.transforms or not isinstance(other, BaseSelector) or getattr(other, "transforms", None):
            return SelectorFilterCondition(self, "or", other)

        left = self
        right = other

        def _resolve(df):

            left_cols = set(left.resolve(df))
            right_cols = set(right.resolve(df))

            return [c for c in df.columns if c in left_cols or c in right_cols]

        return self._selector_copy(_resolve)

    def __sub__(self, other: "BaseSelector") -> "SelectorSelectionOperations":
        """Return columns matched by `self` but not by `other` (set difference).

        Parameters
        ----------
        other : BaseSelector
            The selector whose matched columns are excluded from `self`'s.

        Returns
        -------
        SelectorSelectionOperations
            A new selector matching every column in `self` that is not also
            in `other`.

        Examples
        --------
        >>> by_dtype([T.StringType, T.IntegerType, T.DoubleType]) - by_dtype([T.IntegerType, T.DoubleType])
        <DTypeSelector ...>
        """

        left = self
        right = other

        def _resolve(df):

            right_cols = set(right.resolve(df))

            return [dim for dim in left.resolve(df) if dim not in right_cols]

        return self._selector_copy(_resolve)

    def __xor__(self, other: "BaseSelector") -> "SelectorSelectionOperations":
        """Return columns matched by exactly one of `self`/`other` (symmetric difference).

        Parameters
        ----------
        other : BaseSelector
            The selector to compare against `self`.

        Returns
        -------
        SelectorSelectionOperations
            A new selector matching every column present in exactly one of
            `self` or `other`, but not both.

        Examples
        --------
        >>> by_dtype([T.DoubleType, T.IntegerType]) ^ by_dtype([T.IntegerType, T.StringType])
        <DTypeSelector ...>
        """

        left = self
        right = other

        def _resolve(df):

            left_cols = set(left.resolve(df))
            right_cols = set(right.resolve(df))

            return [c for c in df.columns if (c in left_cols) != (c in right_cols)]

        return self._selector_copy(_resolve)

    def __invert__(self) -> Union["SelectorSelectionOperations", "SelectorFilterCondition"]:
        """Complement this selector's matched columns, or negate a row condition.

        Returns
        -------
        SelectorSelectionOperations or SelectorFilterCondition
            A new selector matching every column NOT matched by `self`
            (set-algebra mode), or a `SelectorFilterCondition` negating the
            row condition (expression mode -- triggered when `self` already
            has a chained comparison/arithmetic transform, e.g.
            ``~(by_dtype([T.IntegerType]) > 0)``).

        Examples
        --------
        >>> ~by_dtype([T.StringType])
        <DTypeSelector ...>
        >>> df.filter(~(by_dtype([T.IntegerType]) > 0))
        DataFrame[...]
        """

        # same expression-mode guard as __and__/__or__ -- if a comparison/arithmetic
        # transform is already chained onto this selector (e.g. `~(by_dtype([T.IntegerType]) > 0)`),
        # `~` has to negate that row condition, not complement the matched
        # *columns*. without this check `~` would silently throw away the
        # `> 0` transform and return "every column NOT of dtype int" instead -- a
        # completely different (and wrong) result.
        if self.transforms:
            return SelectorFilterCondition(self, "invert")

        original = self

        return self._selector_copy(lambda df: [c for c in df.columns if c not in original.resolve(df)])


# --------------------------------------------------------------------------------
# Filter condition for row-level filtering
# --------------------------------------------------------------------------------


class SelectorFilterCondition:
    """Lazy boolean row-condition combinator used by `filter`/`where`.

    Companion to `SelectorSelectionOperations`: once ``&``/``|``/``~`` are
    used on a selector that already has a comparison/arithmetic transform
    chained onto it (e.g. ``by_dtype([T.IntegerType]) == 0``), or against a plain
    `pyspark.sql.Column` (e.g. ``F.col('data_column_4') == None``), those
    operators no longer mean "combine which columns are selected" -- they
    mean "AND/OR these row conditions together".
    `SelectorSelectionOperations.__and__`/`__or__`/`__invert__` detect that
    case and return one of these instead of a new selector.

    Resolving into a real boolean `pyspark.sql.Column` is deferred (via
    `to_column`) since it requires a `DataFrame` to resolve each selector's
    matched columns against -- that dataframe isn't known yet at
    ``&``/``|``/``~`` time, only once `filter`/`where` actually runs. A
    selector operand that matches more than one column (e.g.
    ``by_dtype([T.IntegerType]) > 0`` matching several int columns) is folded down
    with ``&`` across its own matches first, so combining conditions composes
    cleanly no matter how many real columns are on either side.

    Parameters
    ----------
    left : BaseSelector, SelectorFilterCondition, or pyspark.sql.Column
        The left-hand operand of this condition.
    op : {"and", "or", "invert", None}, optional
        The boolean operator combining `left` and `right`. `None` means
        `left` is passed through unchanged (used when a single selector is
        given directly to `filter`).
    right : BaseSelector, SelectorFilterCondition, or pyspark.sql.Column, optional
        The right-hand operand, required when `op` is ``"and"``/``"or"``,
        unused when `op` is ``"invert"``/`None`.

    Examples
    --------
    >>> df.filter((by_dtype([T.IntegerType]) > 0) & (by_dtype([T.IntegerType]) < 9))
    DataFrame[...]
    >>> df.filter(~(by_dtype([T.IntegerType]) > 0))
    DataFrame[...]
    """

    def __init__(
        self,
        left: Union["BaseSelector", "SelectorFilterCondition", Column],
        op: str | None = None,
        right: Union["BaseSelector", "SelectorFilterCondition", Column] | None = None,
    ) -> None:
        """Initialize the condition tree node.

        Parameters
        ----------
        left : BaseSelector, SelectorFilterCondition, or pyspark.sql.Column
            The left-hand operand of this condition.
        op : {"and", "or", "invert", None}, optional
            The boolean operator combining `left` and `right`.
        right : BaseSelector, SelectorFilterCondition, or pyspark.sql.Column, optional
            The right-hand operand, when `op` requires one.
        """

        self._left = left
        self._op = op
        self._right = right

    @staticmethod
    def _to_expr(operand: Union["BaseSelector", "SelectorFilterCondition", Column, Any], df: DataFrame) -> Column:
        """Resolve one operand (selector, condition, or plain value) into one `Column`.

        Parameters
        ----------
        operand : BaseSelector, SelectorFilterCondition, or Any
            The operand to resolve. A `SelectorFilterCondition` recurses via
            `to_column`; a `BaseSelector` resolves to all of its matched
            columns' expressions, ANDed together; anything else (a plain
            `pyspark.sql.Column` or literal) passes through unchanged.
        df : pyspark.sql.DataFrame
            The dataframe to resolve selector-matched columns against.

        Returns
        -------
        pyspark.sql.Column
            The resolved boolean (or other) expression.

        Raises
        ------
        pyspark.errors.PySparkValueError
            If `operand` is a `BaseSelector` that matches zero columns
            against `df`.
        """

        if isinstance(operand, SelectorFilterCondition):
            return operand.to_column(df)

        if isinstance(operand, BaseSelector):
            exprs = operand.resolve_columns(df)

            if not exprs:
                raise PySparkValueError(
                    error_class="CANNOT_BE_EMPTY",
                    message_parameters={"item": "columns matched by the selector"},
                )

            combined = exprs[0]

            for expr in exprs[1:]:
                combined = combined & expr

            return combined

        return operand  # plain pyspark Column / literal condition, passed through as-is

    def to_column(self, df: DataFrame) -> Column:
        """Resolve this condition tree into one final boolean `pyspark.sql.Column`.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            The dataframe to resolve every selector-matched column against.

        Returns
        -------
        pyspark.sql.Column
            The fully resolved boolean expression, ready to be passed to
            `pyspark.sql.DataFrame.filter`.

        Raises
        ------
        pyspark.errors.PySparkValueError
            If `self._op` is set to something other than
            ``"and"``/``"or"``/``"invert"``/`None`.

        Examples
        --------
        >>> SelectorFilterCondition(by_dtype([T.IntegerType]) > 0).to_column(df)
        Column<'(integer_col > 0)'>
        """

        left_expr = self._to_expr(self._left, df)

        if self._op is None:
            return left_expr

        if self._op == "invert":
            return ~left_expr

        right_expr = self._to_expr(self._right, df)

        if self._op == "and":
            return left_expr & right_expr

        if self._op == "or":
            return left_expr | right_expr

        raise PySparkValueError(
            message=f"unsupported filter condition operator: {self._op!r}",
        )

    def __and__(self, other: Union["BaseSelector", "SelectorFilterCondition", Column]) -> "SelectorFilterCondition":
        """Return a new condition ANDing `self` with `other`.

        Parameters
        ----------
        other : BaseSelector, SelectorFilterCondition, or pyspark.sql.Column
            The right-hand operand of ``&``.

        Returns
        -------
        SelectorFilterCondition
            A new condition node combining `self` and `other` with ``"and"``.
        """
        return SelectorFilterCondition(self, "and", other)

    def __rand__(self, other: Union["BaseSelector", Column]) -> "SelectorFilterCondition":
        """Return a new condition ANDing `other` with `self` (reflected `&`).

        Parameters
        ----------
        other : BaseSelector or pyspark.sql.Column
            The left-hand operand, when `other & self` is evaluated because
            `other` doesn't implement `__and__` for `self`'s type.

        Returns
        -------
        SelectorFilterCondition
            A new condition node combining `other` and `self` with ``"and"``.
        """
        return SelectorFilterCondition(other, "and", self)

    def __or__(self, other: Union["BaseSelector", "SelectorFilterCondition", Column]) -> "SelectorFilterCondition":
        """Return a new condition ORing `self` with `other`.

        Parameters
        ----------
        other : BaseSelector, SelectorFilterCondition, or pyspark.sql.Column
            The right-hand operand of ``|``.

        Returns
        -------
        SelectorFilterCondition
            A new condition node combining `self` and `other` with ``"or"``.
        """
        return SelectorFilterCondition(self, "or", other)

    def __ror__(self, other: Union["BaseSelector", Column]) -> "SelectorFilterCondition":
        """Return a new condition ORing `other` with `self` (reflected `|`).

        Parameters
        ----------
        other : BaseSelector or pyspark.sql.Column
            The left-hand operand, when `other | self` is evaluated because
            `other` doesn't implement `__or__` for `self`'s type.

        Returns
        -------
        SelectorFilterCondition
            A new condition node combining `other` and `self` with ``"or"``.
        """
        return SelectorFilterCondition(other, "or", self)

    def __invert__(self) -> "SelectorFilterCondition":
        """Return a new condition negating `self`.

        Returns
        -------
        SelectorFilterCondition
            A new condition node wrapping `self` with ``"invert"``.
        """
        return SelectorFilterCondition(self, "invert")


# ==========================================================
# define exactly what a column selector is and  is capable of doing
# ==========================================================


class BaseSelector(SelectorSelectionOperations, SelectorcolumnOperations):
    """Abstract base class for every column selector (`by_dtype`, `by_index`, `matches`).

    Combines `SelectorSelectionOperations` (set algebra: ``~ - & ^ |``) and
    `SelectorcolumnOperations` (chained column transforms: arithmetic,
    comparisons, every `pyspark.sql.functions` function) into one mixin base
    that every concrete selector class inherits from. A concrete subclass
    only needs to implement `resolve` -- everything else (transform chaining,
    set-algebra, filter-condition composition) is inherited for free.

    Parameters
    ----------
    transforms : list of callable, optional
        Initial transform pipeline, forwarded to
        `SelectorcolumnOperations.__init__`.
    resolver : callable, optional
        Initial combinator resolver, forwarded to
        `SelectorSelectionOperations.__init__`.

    Notes
    -----
    MRO is ``BaseSelector -> SelectorSelectionOperations -> SelectorcolumnOperations``,
    so `SelectorSelectionOperations`'s set-algebra `__invert__` (correct unary
    signature) wins by default -- no override needed on this class for that
    one. `__sub__` still needs disambiguation here since both mixins define
    it, and unguarded MRO would make set-difference the silent default,
    breaking arithmetic subtraction (e.g. ``some_selector - 5``, which has no
    ``.resolve()``).
    """

    def __init__(
        self,
        transforms: list[Callable[[Column], Column]] | None = None,
        resolver: Callable[[DataFrame], list[str]] | None = None,
        require_col_match: bool = True,
    ) -> None:
        """Initialize both parent mixins' state.

        Parameters
        ----------
        transforms : list of callable, optional
            Initial transform pipeline.
        resolver : callable, optional
            Initial combinator resolver.
        require_col_match : bool, default True
            Whether `resolve` raises `pyspark.errors.PySparkValueError` when
            this selector matches zero columns. Set to `False` to opt out
            and allow a zero-column match to pass through silently.
        """
        SelectorcolumnOperations.__init__(self, transforms)
        SelectorSelectionOperations.__init__(self, resolver)

        self.require_col_match = require_col_match

    # --------------------------------------------------
    # each selector implements this
    # --------------------------------------------------

    def resolve(self, df: DataFrame) -> list[str]:
        """Resolve this selector's matched column names against a dataframe.

        Must be implemented by every concrete subclass (`DTypeSelector`,
        `IndexSelector`, `RegexSelector`, ...). Implementations should check
        `self._resolver` first (set by `_selector_copy` when a set operator
        ``~ - & ^ |`` combines two selectors) before falling back to their
        own type-specific matching logic, and should pass their final
        matched-name list through `_check_matched` before returning it.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            The dataframe to resolve matched column names against.

        Returns
        -------
        list of str
            The matched column names, in dataframe-column order.

        Raises
        ------
        pyspark.errors.PySparkNotImplementedError
            Always, on `BaseSelector` itself -- subclasses must override this.
        """

        raise PySparkNotImplementedError(
            error_class="NOT_IMPLEMENTED",
            message_parameters={"feature": "BaseSelector.resolve"},
        )

    def _check_matched(self, matched: list[str]) -> list[str]:
        """Raise if `matched` is empty, unless `self.require_col_match` is `False`.

        Single shared empty-match guard used by every concrete selector's
        `resolve` (`DTypeSelector`, `IndexSelector`, `RegexSelector`), on
        both its own type-specific matching path and its combined-resolver
        path (`self._resolver`, built by `~ - & ^ |`) -- so a set-algebra
        combination that ends up matching nothing (e.g. `all() - integer()`
        on a dataframe with only one integer column) is caught exactly the
        same way as a plain selector matching nothing (e.g. `numeric()` on a
        dataframe with no numeric columns at all).

        Parameters
        ----------
        matched : list of str
            The column names this selector resolved to.

        Returns
        -------
        list of str
            `matched`, unchanged, when non-empty or when `self.require_col_match`
            is `False`.

        Raises
        ------
        pyspark.errors.PySparkValueError
            If `matched` is empty and `self.require_col_match` is `True`.

        Examples
        --------
        >>> by_dtype([T.StringType]).resolve(df)  # df has no string columns
        Traceback (most recent call last):
            ...
        pyspark.errors.exceptions.base.PySparkValueError: ...
        >>> by_dtype([T.StringType], require_col_match=False).resolve(df)
        []
        """

        if not matched and self.require_col_match:
            raise PySparkValueError(
                error_class="CANNOT_BE_EMPTY",
                message_parameters={"item": f"columns matched by {self.__class__.__name__}"},
            )

        return matched

    # --------------------------------------------------
    # MRO is now BaseSelector -> SelectorSelectionOperations -> SelectorcolumnOperations,
    # so SelectorSelectionOperations' set-algebra __invert__ (correct unary signature)
    # wins by default -- no override needed here anymore.
    #
    # __sub__ still needs disambiguation: both mixins define it, and unguarded
    # MRO would now make set-difference the silent default, breaking arithmetic
    # subtraction (e.g. some_selector - 5) since a plain value has no .resolve().
    # --------------------------------------------------

    def __sub__(self, other: Union["BaseSelector", Any]) -> "BaseSelector":
        """Dispatch `-` to set-difference (selector operand) or arithmetic (plain value).

        Parameters
        ----------
        other : BaseSelector or Any
            A selector operand triggers set-difference (`SelectorSelectionOperations.__sub__`);
            anything else (e.g. an int/float) triggers arithmetic subtraction
            (`SelectorcolumnOperations.__sub__`).

        Returns
        -------
        BaseSelector
            A new selector, either matching the set-difference of columns
            (selector operand) or with ``- other`` chained as a transform
            (plain value operand).

        Examples
        --------
        >>> by_dtype([T.StringType, T.IntegerType]) - by_dtype([T.IntegerType])
        <DTypeSelector ...>
        >>> by_dtype([T.IntegerType]) - 5
        <DTypeSelector ...>
        """

        if isinstance(other, SelectorSelectionOperations):
            return SelectorSelectionOperations.__sub__(self, other)

        return SelectorcolumnOperations.__sub__(self, other)


# --------------------------------------------------------------------------------
# Create  column selector for selecting columns by their real Spark SQL type
# --------------------------------------------------------------------------------


class DTypeSelector(BaseSelector):
    """Select columns whose real Spark SQL type is (or is a subclass of) any given type.

    Matches against each column's real `pyspark.sql.types.DataType` instance
    (`df.schema[name].dataType`) via `isinstance`, not against dtype strings.
    Passing a category base class (e.g. `pyspark.sql.types.NumericType`)
    matches every concrete subclass of it (`ByteType`, `ShortType`,
    `IntegerType`, `LongType`, `FloatType`, `DoubleType`, `DecimalType`, ...),
    the same way `isinstance` already works for any Python class hierarchy --
    no separate "category" concept needed on top of it.

    Parameters
    ----------
    dtypes : type or sequence of type
        One or more `pyspark.sql.types.DataType` subclasses to match against
        each column's real datatype (e.g. `[T.IntegerType, T.StringType]`, or
        a single category base class like `T.NumericType`).
    transforms : list of callable, optional
        Initial transform pipeline, forwarded to `BaseSelector.__init__`.

    Attributes
    ----------
    dtypes : tuple of type
        The requested `DataType` subclasses, deduplicated, ready to pass
        straight to `isinstance`.

    Examples
    --------
    >>> df.select(by_dtype([T.IntegerType, T.DoubleType]))
    DataFrame[...]
    >>> df.select(by_dtype([T.DoubleType]).cast('string'))
    DataFrame[...]
    >>> df.select(by_dtype([T.NumericType]))
    DataFrame[...]
    """

    def __init__(
        self,
        dtypes: type | Sequence[type],
        transforms: list[Callable[[Column], Column]] | None = None,
        require_col_match: bool = True,
    ) -> None:
        """Initialize the requested `DataType` subclass set.

        Parameters
        ----------
        dtypes : type or sequence of type
            One or more `pyspark.sql.types.DataType` subclasses to match.
        transforms : list of callable, optional
            Initial transform pipeline.
        require_col_match : bool, default True
            Whether `resolve` raises `pyspark.errors.PySparkValueError` when
            zero columns match `dtypes`. Forwarded to `BaseSelector.__init__`.

        Raises
        ------
        pyspark.errors.PySparkTypeError
            If `dtypes` (or any item in it) is not a class, or not a subclass
            of `pyspark.sql.types.DataType`.
        """

        super().__init__(transforms, require_col_match=require_col_match)

        # a bare single class (e.g. by_dtype(T.IntegerType)) is convenient
        # shorthand for a one-item sequence -- normalize it the same way.
        if isinstance(dtypes, type):
            dtypes = (dtypes,)

        dtypes = tuple(dict.fromkeys(dtypes))  # dedupe, preserve order

        self._validate_dtype_classes(dtypes)

        self.dtypes = dtypes

    def _validate_dtype_classes(self, dtypes: Sequence[type]) -> None:
        """Raise if any requested item isn't a real `pyspark.sql.types.DataType` subclass.

        Parameters
        ----------
        dtypes : sequence of type
            The candidate classes to validate.

        Raises
        ------
        pyspark.errors.PySparkTypeError
            If any item is not a class, or is a class but not a subclass of
            `pyspark.sql.types.DataType`.
        """

        bad = [d for d in dtypes if not (isinstance(d, type) and issubclass(d, T.DataType))]

        if bad:
            raise PySparkTypeError(
                message=(
                    f"by_dtype() received non-DataType item(s): {bad!r} -- every item must "
                    "be a class (not an instance or a string) that subclasses "
                    "pyspark.sql.types.DataType, e.g. T.IntegerType or T.NumericType."
                ),
            )

    def resolve(self, df: DataFrame) -> list[str]:
        """Resolve to every column name on `df` whose real datatype is in `self.dtypes`.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            The dataframe to resolve matched column names against.

        Returns
        -------
        list of str
            Matched column names, in dataframe-column order.

        Raises
        ------
        pyspark.errors.PySparkValueError
            Raised by `_check_matched` (via `BaseSelector`) if zero columns
            match and `self.require_col_match` is `True`.

        Examples
        --------
        >>> DTypeSelector([T.IntegerType]).resolve(df)
        ['integer_col']
        """

        # if a set-operation (~ - & ^ |) built a combined resolver via
        # _selector_copy, honor it first. otherwise fall back to the
        # normal dtype-matching behavior. without this check, resolve()
        # would always recompute off of self.dtypes and silently ignore
        # any selector-selection-operations combinator applied to it.
        if self._resolver:
            return self._check_matched(self._resolver(df))

        return self._check_matched(
            [field.name for field in df.schema.fields if isinstance(field.dataType, self.dtypes)]
        )


# --------------------------------------------------------------------------------
# create column selector for selecting index position in  table
# --------------------------------------------------------------------------------


class IndexSelector(BaseSelector):
    """Select columns by positional index.

    Mirrors Python's own negative-indexing rule: ``-1`` is the last column,
    ``-2`` is second-to-last, etc, resolved against ``len(df.columns)``
    since the selector itself has no dataframe until `resolve` runs.

    Parameters
    ----------
    indexes : tuple of (int or range)
        One or more positional indexes to select. Each item is either a bare
        int, or a `range` object covering a span in one shot (e.g.
        ``range(1, 4)`` for columns 1-3). Negative ints/ranges are supported
        (e.g. ``range(-3, 0)`` for the last 3 columns), same rule as a bare
        negative int.
    transforms : list of callable, optional
        Initial transform pipeline, forwarded to `BaseSelector.__init__`.
    strict_index_bounds : bool, default True
        Controls what happens when a requested index is out of bounds for
        the dataframe actually being resolved against: `True` raises
        immediately, naming the bad index and the dataframe's real column
        count; `False` silently skips any out-of-bounds index instead of
        raising, so partial matches are tolerated.

    Attributes
    ----------
    indexes : tuple of (int or range)
        The requested indexes, as given.
    strict_index_bounds : bool
        Whether an out-of-bounds index raises (`True`) or is skipped
        (`False`).

    Notes
    -----
    Duplicate positions (e.g. an index reachable both directly and via an
    overlapping range) are de-duplicated, preserving first-seen order.
    ``~ - & ^ |`` all work already -- `IndexSelector` only implements
    `resolve`; the set-algebra (and filter-condition) operators live entirely
    on `SelectorSelectionOperations`/`BaseSelector`, which every child class
    (this one, `DTypeSelector`, `RegexSelector`) inherits for free.

    Examples
    --------
    >>> df.select(by_index(0, 1))
    DataFrame[...]
    >>> df.select(by_index(range(1, 4), -1))
    DataFrame[...]
    >>> df.select(by_index(0, strict_index_bounds=False))
    DataFrame[...]
    """

    def __init__(
        self,
        indexes: tuple[int | range, ...],
        transforms: list[Callable[[Column], Column]] | None = None,
        strict_index_bounds: bool = True,
        require_col_match: bool = True,
    ) -> None:
        """Initialize the requested indexes and out-of-bounds behavior.

        Parameters
        ----------
        indexes : tuple of (int or range)
            One or more positional indexes to select.
        transforms : list of callable, optional
            Initial transform pipeline.
        strict_index_bounds : bool, default True
            Whether an out-of-bounds index raises (`True`) or is skipped
            (`False`).
        require_col_match : bool, default True
            Whether `resolve` raises `pyspark.errors.PySparkValueError` when
            zero columns match (only reachable when `strict_index_bounds` is
            `False`, or via a combined `~ - & ^ |` resolver). Forwarded to
            `BaseSelector.__init__`.
        """

        super().__init__(transforms, require_col_match=require_col_match)

        self.indexes = indexes
        self.strict_index_bounds = strict_index_bounds

    @staticmethod
    def _positions_from(item: int | range) -> list[int]:
        """Expand one raw index arg (an int, or a range) into a flat list of ints.

        Parameters
        ----------
        item : int or range
            A single index, or a `range` covering a span of indexes.

        Returns
        -------
        list of int
            ``[item]`` when `item` is a bare int, or ``list(item)`` when
            `item` is a `range`.
        """

        if isinstance(item, range):
            return list(item)

        return [item]

    def resolve(self, df: DataFrame) -> list[str]:
        """Resolve to the column name(s) at every requested position on `df`.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            The dataframe to resolve matched column names against.

        Returns
        -------
        list of str
            Matched column names, in first-seen (requested) order, with
            duplicate positions removed.

        Raises
        ------
        pyspark.errors.PySparkValueError
            If `self.strict_index_bounds` is `True` and any requested index is out
            of range for `df`'s column count. Also raised by
            `_check_matched` (via `BaseSelector`) if zero columns match and
            `self.require_col_match` is `True`.

        Examples
        --------
        >>> IndexSelector((0, -1)).resolve(df)
        ['string_col', 'integer_col']
        """

        # if a set-operation (~ - & ^ |) built a combined resolver via
        # _selector_copy, honor it first -- same pattern as DTypeSelector.resolve.
        if self._resolver:
            return self._check_matched(self._resolver(df))

        columns = df.columns
        n = len(columns)

        seen = set()
        ordered_positions = []

        for item in self.indexes:
            for pos in self._positions_from(item):
                actual = pos if pos >= 0 else n + pos

                if actual < 0 or actual >= n:
                    if self.strict_index_bounds:
                        raise PySparkValueError(
                            errorClass="VALUE_OUT_OF_BOUNDS",
                            messageParameters={
                                "arg_name": "by_index() column index",
                                "lower_bound": str(-n),
                                "upper_bound": str(n - 1),
                                "actual": str(pos),
                            },
                        )

                    continue

                if actual not in seen:
                    seen.add(actual)
                    ordered_positions.append(actual)

        return self._check_matched([columns[p] for p in ordered_positions])


# --------------------------------------------------------------------------------
# create column selector that can select column's by their column name regex pattern
# --------------------------------------------------------------------------------


class RegexSelector(BaseSelector):
    """Select columns whose name matches a regex pattern.

    pyspark has a native regex column selector, `pyspark.sql.DataFrame.colRegex`
    -- but it has to be called directly off a `DataFrame` instance
    (``df.colRegex(...)``, not a free-standing selector built ahead of time),
    and it returns a single opaque `pyspark.sql.Column` representing "all
    matching columns" without ever exposing which real column *names*
    matched. Every override in this notebook (`select`, `agg`, `sort`,
    `withColumns`, etc, via `_resolve_selector_exprs`) depends on getting back
    real ``(name, expr)`` pairs so a chained transform (``.cast(...)``,
    ``.desc()``, ``.upper()``, ...) can be applied per matched column and the
    result still knows what to call each output column -- `colRegex` can't
    give us that, so `resolve` instead matches each real column name in
    `df.columns` against the pattern directly, via `re.search`.

    Parameters
    ----------
    pattern : str
        A regular expression, matched against each column name via
        `re.search` (not anchored -- matches anywhere in the name unless the
        pattern itself anchors with ``^``/``$``).
    transforms : list of callable, optional
        Initial transform pipeline, forwarded to `BaseSelector.__init__`.

    Attributes
    ----------
    pattern : str
        The requested regex pattern, as given.

    Notes
    -----
    ``~ - & ^ |`` all work already, same reasoning as `IndexSelector`.

    Examples
    --------
    >>> df.select(matches(r"_id$"))
    DataFrame[...]
    >>> df.select(matches(r"_id$").upper())
    DataFrame[...]
    """

    def __init__(
        self,
        pattern: str,
        transforms: list[Callable[[Column], Column]] | None = None,
        require_col_match: bool = True,
    ) -> None:
        """Initialize the requested regex pattern.

        Parameters
        ----------
        pattern : str
            A regular expression, matched against each column name.
        transforms : list of callable, optional
            Initial transform pipeline.
        require_col_match : bool, default True
            Whether `resolve` raises `pyspark.errors.PySparkValueError` when
            zero columns match `pattern`. Forwarded to `BaseSelector.__init__`.
        """

        super().__init__(transforms, require_col_match=require_col_match)

        self.pattern = pattern

    def resolve(self, df: DataFrame) -> list[str]:
        """Resolve to every column name on `df` matching `self.pattern`.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            The dataframe to resolve matched column names against.

        Returns
        -------
        list of str
            Matched column names, in dataframe-column order.

        Raises
        ------
        pyspark.errors.PySparkValueError
            Raised by `_check_matched` (via `BaseSelector`) if zero columns
            match and `self.require_col_match` is `True`.

        Examples
        --------
        >>> RegexSelector(r"_id$").resolve(df)
        ['string_col', 'data_column_5']
        """

        # if a set-operation (~ - & ^ |) built a combined resolver via
        # _selector_copy, honor it first -- same pattern as DTypeSelector.resolve.
        if self._resolver:
            return self._check_matched(self._resolver(df))

        # case-insensitive by design: pyspark's own column-name resolution
        # (`df.col_name`, `df['Col_Name']`, `df.select('col_name')`, joins,
        # ...) is case-insensitive by default, so a case-sensitive regex here
        # would silently miss columns a plain pyspark reference would still
        # find. re.IGNORECASE keeps this selector's matching behavior
        # consistent with that.
        return self._check_matched(
            [dim for dim in df.columns if re.search(self.pattern, dim, flags=re.IGNORECASE) is not None]
        )
