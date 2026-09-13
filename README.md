# PySpark Column Selectors

A small object-oriented framework for selecting DataFrame columns by **dtype**, **position**, or **name pattern** instead of typing out column names by hand. A selector can be passed almost anywhere a column or list of columns is normally accepted, and selectors can be combined with set operations and chained with ordinary column transforms.

![PyPI version](https://img.shields.io/pypi/v/PySpark_Column_Selectors.svg)

Adds column selector functionality to pyspark

-   GitHub: <https://github.com/tmichel3796/PySpark_Column_Selectors/>
-   PyPI package: <https://pypi.org/project/PySpark_Column_Selectors/>
-   Created by: **Trevor A. Michel** trevormichel.com \| GitHub <https://github.com/tmichel3796> \| PyPI <https://pypi.org/user/tmichel3796/>
-   Free software: MIT License

## Features

## Dependencies

-   Python 3.8+
-   PySpark 3.4 or newer (for full functionality)
    -   PySpark 3.3 still works, except `T.TimestampNTZType` won't exist
    -   `temporal()`/`datetime_()` fall back to matching only `T.TimestampType` on those older versions instead of raising.
    -   `withColumns` with multiple selector-driven mutations in a single call (see [Using selectors with other DataFrame operations](#using-selectors-with-other-dataframe-operations)) requires PySpark 3.3+, since `DataFrame.withColumns` itself didn't exist before then.
-   No third-party packages beyond PySpark itself -- everything else used (`functools`, `operator`, `re`, `typing`) is part of the Python standard library.
-   Spark Connect (`pyspark.sql.connect`) is supported automatically when present, but is not required -- this framework works the same either way.

## Importing

``` python
import PySparkSelectors as scs
```

Every example below assumes this import, and calls each selector as `scs.<function_name>(...)`.

------------------------------------------------------------------------

## Example data

Every example in this document runs against the same small DataFrame:

``` python
df = spark.createDataFrame(
    [
        (1, "north", "east", 120.5, 88, True, "2024-01-15", 910, "2024-06-01 08:30:00"),
        (2, "south", "west", 340.0, 95, False, "2024-03-22", 875, "2024-06-02 14:45:00"),
    ],
    ["id", "name", "region", "elevation", "totalscore", "isactive", "signupdate", "score2024", "lastseen"],
)
```

| id | name | region | elevation | totalscore | isactive | signupdate | score2024 | lastseen |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| 1 | north | east | 120.5 | 88 | true | 2024-01-15 | 910 | 2024-06-01 08:30:00 |
| 2 | south | west | 340.0 | 95 | false | 2024-03-22 | 875 | 2024-06-02 14:45:00 |

Dtypes: `id` (int), `name`/`region` (string), `elevation` (double), `totalscore`/`score2024` (int), `isactive` (boolean), `signupdate` (date), `lastseen` (timestamp).

------------------------------------------------------------------------

## Dtype-based selectors

Dtype selectors match against the real `pyspark.sql.types` classes, not dtype strings. Import the types module alongside the selectors:

``` python
import pyspark.sql.types as T
```

### `by_dtype([...])`

Matches any `pyspark.sql.types.DataType` class (or classes) you pass in. A category base class (e.g. `T.NumericType`) matches every concrete subclass of it too -- `by_dtype([T.NumericType])` matches integer, floating-point, *and* decimal columns in one call.

``` python
df.select(scs.by_dtype([T.StringType]))
```

| name  | region |
|-------|--------|
| north | east   |
| south | west   |

### `string()`

``` python
df.select(scs.string())
```

| name  | region |
|-------|--------|
| north | east   |
| south | west   |

### `boolean()`

``` python
df.select(scs.boolean())
```

| isactive |
|----------|
| true     |
| false    |

### `binary()`

There are no binary columns in this DataFrame, so this raises an error by default rather than returning nothing -- see [Handling zero matches](#handling-zero-matches) below.

### `integer()`

Fixed-width integer columns (tinyint/smallint/int/bigint).

``` python
df.select(scs.integer())
```

| id  | totalscore | score2024 |
|-----|------------|-----------|
| 1   | 88         | 910       |
| 2   | 95         | 875       |

### `floats()`

Floating-point columns (float/double).

``` python
df.select(scs.floats())
```

| elevation |
|-----------|
| 120.5     |
| 340.0     |

### `numeric()`

Every fixed-width numeric column (integer or floating-point).

``` python
df.select(scs.numeric())
```

| id  | elevation | totalscore | score2024 |
|-----|-----------|------------|-----------|
| 1   | 120.5     | 88         | 910       |
| 2   | 340.0     | 95         | 875       |

### `date()`

``` python
df.select(scs.date())
```

| signupdate |
|------------|
| 2024-01-15 |
| 2024-03-22 |

### `datetime_()`

``` python
df.select(scs.datetime_())
```

| lastseen            |
|---------------------|
| 2024-06-01 08:30:00 |
| 2024-06-02 14:45:00 |

### `temporal()`

Date or timestamp columns.

``` python
df.select(scs.temporal())
```

| signupdate | lastseen            |
|------------|---------------------|
| 2024-01-15 | 2024-06-01 08:30:00 |
| 2024-03-22 | 2024-06-02 14:45:00 |

------------------------------------------------------------------------

## Position-based selectors

### `by_index(*indexes)`

Selects one or more columns by position. Negative positions work the same way Python indexing does (`-1` is the last column), and a `range(...)` can be mixed in with plain integers to grab a span in one call.

``` python
df.select(scs.by_index(range(0, 1),3))
```

| id  | name  | elevation |
|-----|-------|-----------|
| 1   | north | 120.5     |
| 2   | south | 340.0     |

### `first()`

``` python
df.select(scs.first())
```

| id  |
|-----|
| 1   |
| 2   |

### `last()`

``` python
df.select(scs.last())
```

| lastseen            |
|---------------------|
| 2024-06-01 08:30:00 |
| 2024-06-02 14:45:00 |

### Handle column not in range error {#handle-column-not-in-range-error}

By default, `by_index` raises if any requested position is out of range for the DataFrame (this DataFrame has 9 columns, at positions 0-8):

``` python
df.select(scs.by_index(9))
# raises pyspark.errors.PySparkValueError -- position 9 is out of range
```

Passing `strict_index_bounds=False` skips any out-of-range position instead of raising, so only the in-range positions are matched:

``` python
df.select(scs.by_index(0, 9, strict_index_bounds=False))
```

| id  |
|-----|
| 1   |
| 2   |

> Note: if every requested position ends up out of range, `strict_index_bounds=False` alone still isn't enough to avoid an error -- the selector would then match zero columns, which raises on its own (see [Handling zero matches](#handling-zero-matches)). To allow that case too, also pass `require_col_match=False`.

------------------------------------------------------------------------

## Name-based selectors

All name-based matching below is case-insensitive -- `scs.by_name("REGION")` matches a column named `region` just as well as `scs.by_name("region")` does, and column names may contain a literal dot (e.g. `meta.source`) without being mistaken for a nested field.

### `matches(pattern)`

Selects columns whose name matches a regular expression.

``` python
df.select(scs.matches("score"))
```

| totalscore | score2024 |
|------------|-----------|
| 88         | 910       |
| 95         | 875       |

`matches` takes any pattern the `re` module can compile, not just a plain substring -- anchors, character classes, and alternation all work:

``` python
df.select(scs.matches(r"^s.*[aeiou]$|\d"))
```

| signupdate | score2024 |
|------------|-----------|
| 2024-01-15 | 910       |
| 2024-03-22 | 875       |

This pattern matches a column name if it either starts with `s` and ends in a vowel (`signupdate`), or contains a digit anywhere in the name (`score2024`).

### `starts_with(*prefixes)`

``` python
df.select(scs.starts_with("e"))
```

| elevation |
|-----------|
| 120.5     |
| 340.0     |

### `ends_with(*suffixes)`

``` python
df.select(scs.ends_with("date"))
```

| signupdate |
|------------|
| 2024-01-15 |
| 2024-03-22 |

### `contains(*substrings)`

``` python
df.select(scs.contains("2024"))
```

| score2024 |
|-----------|
| 910       |
| 875       |

### `by_name(*names)`

Selects an exact list of column names.

``` python
df.select(scs.by_name("id", "region"))
```

| id  | region |
|-----|--------|
| 1   | east   |
| 2   | west   |

### `exclude(*names)`

Selects every column except the given names.

``` python
df.select(scs.exclude("region"))
```

| id | name | elevation | totalscore | isactive | signupdate | score2024 | lastseen |
|---------|---------|---------|---------|---------|---------|---------|---------|
| 1 | north | 120.5 | 88 | true | 2024-01-15 | 910 | 2024-06-01 08:30:00 |
| 2 | south | 340.0 | 95 | false | 2024-03-22 | 875 | 2024-06-02 14:45:00 |

### `alpha()`

Column names made up only of letters. `score2024` is the only column name containing a digit, so it's the only one excluded here.

``` python
df.select(scs.alpha())
```

| id  | name  | region | elevation | totalscore | isactive | signupdate | lastseen            |
|---------|---------|---------|---------|---------|---------|---------|---------|
| 1   | north | east   | 120.5     | 88         | true     | 2024-01-15 | 2024-06-01 08:30:00 |
| 2   | south | west   | 340.0     | 95         | false    | 2024-03-22 | 2024-06-02 14:45:00 |

### `alphanumeric()`

Column names made up only of letters and/or digits -- every column name in this DataFrame qualifies, including `score2024`.

``` python
df.select(scs.alphanumeric())
```

Returns every column, identical to the example data table above.

### `all()`

``` python
df.select(scs.all())
```

Returns every column, identical to the example data table above.

------------------------------------------------------------------------

## Combining selectors

Selectors support set-style operators so more specific selections can be built without listing columns by hand. Combined selectors always resolve columns in the same order they appear in the underlying DataFrame.

### Union (`|`)

``` python
df.select(scs.string() | scs.starts_with("e"))
```

| name  | region | elevation |
|-------|--------|-----------|
| north | east   | 120.5     |
| south | west   | 340.0     |

### Intersection (`&`)

``` python
df.select(scs.numeric() & scs.matches("score"))
```

| totalscore | score2024 |
|------------|-----------|
| 88         | 910       |
| 95         | 875       |

### Difference (`-`)

``` python
df.select(scs.all() - scs.numeric())
```

| name  | region | isactive | signupdate | lastseen            |
|-------|--------|----------|------------|---------------------|
| north | east   | true     | 2024-01-15 | 2024-06-01 08:30:00 |
| south | west   | false    | 2024-03-22 | 2024-06-02 14:45:00 |

### Symmetric difference (`^`)

Columns matched by exactly one side, not both.

``` python
df.select(scs.contains("score") ^ scs.numeric())
```

| id  | elevation |
|-----|-----------|
| 1   | 120.5     |
| 2   | 340.0     |

### Complement (`~`)

``` python
df.select(~scs.temporal())
```

| id  | name  | region | elevation | totalscore | isactive | score2024 |
|-----|-------|--------|-----------|------------|----------|-----------|
| 1   | north | east   | 120.5     | 88         | true     | 910       |
| 2   | south | west   | 340.0     | 95         | false    | 875       |

------------------------------------------------------------------------

## Chaining transforms onto a selector

A transform can be chained onto a selector before it's resolved. The transform is applied to every column the selector matches, and the original column name is preserved in the result.

### `.upper()`

``` python
df.select(scs.string().upper())
```

| name  | region |
|-------|--------|
| NORTH | EAST   |
| SOUTH | WEST   |

### `.cast(...)`

``` python
df.select(scs.floats().cast("string"))
```

| elevation |
|-----------|
| 120.5     |
| 340.0     |

The values look the same, but `elevation` is now a `string` column instead of a `double` column.

### Arithmetic

``` python
df.select(scs.integer() + 1)
```

| id  | totalscore | score2024 |
|-----|------------|-----------|
| 2   | 89         | 911       |
| 3   | 96         | 876       |

Because every column function available in PySpark is also available as a chained method on a selector, this works with things like `.substr(...)`, `.isNull()`, `.between(...)`, and comparison operators too, not just a fixed list of built-ins.

### Renaming columns:

`.prefix(...)`, `.suffix(...)`, and `.map_alias(...)` Three functions used to modify a columns name.

#### `.prefix(prefix)`

Prepends `prefix` to every matched column's name.

``` python
df.select(scs.by_name("id", "region").prefix("src_"))
```

| src_id | src_region |
|--------|------------|
| 1      | east       |
| 2      | west       |

#### `.suffix(suffix)`

Appends `suffix` to every matched column's name.

``` python
df.select(scs.numeric().suffix("_raw"))
```

| id_raw | elevation_raw | totalscore_raw | score2024_raw |
|--------|---------------|----------------|---------------|
| 1      | 120.5         | 88             | 910           |
| 2      | 340.0         | 95             | 875           |

#### `.map_alias(func)`

Renames every matched column by passing its current name to `func`, a one-argument function returning the new name. `.prefix(...)` and `.suffix(...)` are both just `.map_alias(...)` with a prepend/append built in -- `.map_alias(...)` is for anything more custom.

``` python
df.select(scs.by_name("name", "region").map_alias(lambda n: n.upper()))
```

| NAME  | REGION |
|-------|--------|
| north | east   |
| south | west   |

Note this renames the *columns*, not the *values* -- compare to `.upper()` above, which uppercases the values but keeps the column names unchanged.

------------------------------------------------------------------------

## Using selectors with row filtering

A selector used inside a comparison becomes a row condition instead of a column pick, and conditions built this way can be combined with `&`, `|`, and `~` the same way plain columns can. `filter`/`where` still return every original column for the matching rows -- only the relevant column(s) are shown below for readability.

### A single condition

``` python
df.filter(scs.by_name("totalscore") > 90)
```

| id  | totalscore |
|-----|------------|
| 2   | 95         |

*(the row for `id=1` is dropped, since `totalscore` there is 88)*

### Combining conditions with `&`

``` python
df.filter((scs.by_name("totalscore") > 90) & (scs.by_name("elevation") > 200))
```

| id  | totalscore | elevation |
|-----|------------|-----------|
| 2   | 95         | 340.0     |

------------------------------------------------------------------------

## Using selectors with other DataFrame operations {#using-selectors-with-other-dataframe-operations}

### `select` with multiple selectors

``` python
df.select(scs.string(), scs.integer())
```

| name  | region | id  | totalscore | score2024 |
|-------|--------|-----|------------|-----------|
| north | east   | 1   | 88         | 910       |
| south | west   | 2   | 95         | 875       |

### `groupBy` / `agg`

``` python
df.groupBy(scs.by_name("region")).agg(scs.integer().sum())
```

| region | sum(id) | sum(totalscore) | sum(score2024) |
|--------|---------|-----------------|----------------|
| east   | 1       | 88              | 910            |
| west   | 2       | 95              | 875            |

### `drop`

``` python
df.drop(scs.by_name("lastseen", "signupdate"))
```

Removes `lastseen` and `signupdate`, leaving `id`, `name`, `region`, `elevation`, `totalscore`, `isactive`, `score2024`.

### `sort` / `orderBy`

``` python
df.sort(scs.by_name("elevation").desc())
```

| id  | elevation |
|-----|-----------|
| 2   | 340.0     |
| 1   | 120.5     |

*(every other original column is still present in the result; only `elevation` is shown here for readability)*

### `withColumns` with multiple selector-driven mutations

``` python
df.withColumns(
    scs.floats().cast("string"),
    scs.string().upper(),
)
```

| id  | name  | region | elevation |
|-----|-------|--------|-----------|
| 1   | NORTH | EAST   | 120.5     |
| 2   | SOUTH | WEST   | 340.0     |

Both mutations happen in the same call: every double column is cast to string, and every string column is uppercased.

------------------------------------------------------------------------

## Checking whether something is a selector

``` python
scs.is_selector(scs.numeric())      # True
scs.is_selector("totalscore")       # False
```

------------------------------------------------------------------------

## Handling zero matches {#handling-zero-matches}

By default, if a selector resolves to zero columns against a given DataFrame, it raises an error rather than silently returning nothing. This also applies to combined selectors -- for example, subtracting a selector that covers every matching column from itself will raise, since the result has nothing left in it.

``` python
df.select(scs.binary())
# raises pyspark.errors.PySparkValueError -- zero columns matched
```

Every selector function accepts a `require_col_match` keyword argument, which defaults to `True`. Passing `require_col_match=False` opts out of this check for that particular selector, allowing a zero-column match to pass through silently instead of raising:

``` python
df.select(scs.binary(require_col_match=False))
# returns a result with zero columns instead of raising
```

This also applies to `by_dtype`, `by_index`, and `matches` directly:

``` python
df.select(scs.all() - scs.by_index(range(0,8))) 
# There are 9 total columns in the df. 
# As such the above column selector expression returns all columns - all columns.
# as such no columns of data are returned.  
# Result: raises pyspark.errors.PySparkValueError -- zero columns matched
```

To bypass the no columns returned error, see the below examples. All column selector functions have access to require_col_match by_index is the only one that offers access to strict_index_bounds as shown in [Handle column not in range error](#handle-column-not-in-range-error)

``` python
df.select(scs.by_dtype([T.BinaryType], require_col_match=False))
df.select(scs.by_index(9, strict_index_bounds=False, require_col_match=False))
df.select(scs.matches("no_such_pattern", require_col_match=False))
```

## Author

PySparkSelectors was created in 2026 by Trevor A. Michel.

Built with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.
