"""Tests for PySparkSelectors package."""

from datetime import date, datetime
from decimal import Decimal

import pyspark.sql.types as T
import pytest
from pyspark.errors import PySparkTypeError, PySparkValueError
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    BinaryType,
    BooleanType,
    ByteType,
    DateType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    MapType,
    ShortType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

import PySparkSelectors
import PySparkSelectors as scs


@pytest.fixture(scope="session")
def spark():
    """Create a SparkSession for testing."""
    return SparkSession.builder.appName("PySparkSelectorsTest").master("local[*]").getOrCreate()


@pytest.fixture(scope="function")
def df_test(spark):
    """Create the default test dataframe with all data types.

    This fixture provides the base dataset used across most tests:
    - 8 rows (Alice through Hannah)
    - 14 columns covering all PySpark data types
    """
    schema = StructType(
        [
            StructField("byte_col", ByteType(), False),
            StructField("short_col", ShortType(), False),
            StructField("int_col", IntegerType(), False),
            StructField("long_col", LongType(), False),
            StructField("float_col", FloatType(), False),
            StructField("double_col", DoubleType(), False),
            StructField("decimal_col", DecimalType(10, 2), False),
            StructField("string_col", StringType(), False),
            StructField("binary_col", BinaryType(), False),
            StructField("boolean_col", BooleanType(), False),
            StructField("date_col", DateType(), False),
            StructField("timestamp_col", TimestampType(), False),
            StructField("array_col", ArrayType(IntegerType()), False),
            StructField("map_col", MapType(StringType(), IntegerType()), False),
        ]
    )

    data = [
        (
            1,
            100,
            1000,
            10000,
            1.1,
            10.01,
            Decimal("100.01"),
            "Alice",
            bytearray(b"A"),
            True,
            date(2026, 1, 1),
            datetime(2026, 1, 1, 8, 0, 0),
            [1, 2],
            {"a": 1},
        ),
        (
            2,
            200,
            2000,
            20000,
            2.2,
            20.02,
            Decimal("200.02"),
            "Bob",
            bytearray(b"B"),
            False,
            date(2026, 1, 2),
            datetime(2026, 1, 2, 9, 0, 0),
            [3, 4],
            {"b": 2},
        ),
        (
            3,
            300,
            3000,
            30000,
            3.3,
            30.03,
            Decimal("300.03"),
            "Charlie",
            bytearray(b"C"),
            True,
            date(2026, 1, 3),
            datetime(2026, 1, 3, 10, 0, 0),
            [5, 6],
            {"c": 3},
        ),
        (
            4,
            400,
            4000,
            40000,
            4.4,
            40.04,
            Decimal("400.04"),
            "Diana",
            bytearray(b"D"),
            False,
            date(2026, 1, 4),
            datetime(2026, 1, 4, 11, 0, 0),
            [7, 8],
            {"d": 4},
        ),
        (
            5,
            500,
            5000,
            50000,
            5.5,
            50.05,
            Decimal("500.05"),
            "Ethan",
            bytearray(b"E"),
            True,
            date(2026, 1, 5),
            datetime(2026, 1, 5, 12, 0, 0),
            [9, 10],
            {"e": 5},
        ),
        (
            6,
            600,
            6000,
            60000,
            6.6,
            60.06,
            Decimal("600.06"),
            "Fiona",
            bytearray(b"F"),
            False,
            date(2026, 1, 6),
            datetime(2026, 1, 6, 13, 0, 0),
            [11, 12],
            {"f": 6},
        ),
        (
            7,
            700,
            7000,
            70000,
            7.7,
            70.07,
            Decimal("700.07"),
            "George",
            bytearray(b"G"),
            True,
            date(2026, 1, 7),
            datetime(2026, 1, 7, 14, 0, 0),
            [13, 14],
            {"g": 7},
        ),
        (
            8,
            800,
            8000,
            80000,
            8.8,
            80.08,
            Decimal("800.08"),
            "Hannah",
            bytearray(b"H"),
            False,
            date(2026, 1, 8),
            datetime(2026, 1, 8, 15, 0, 0),
            [15, 16],
            {"h": 8},
        ),
    ]

    return spark.createDataFrame(data, schema)


# ============================================================================
# Test: Package Initialization
# ============================================================================


def test_import():
    """Verify the package can be imported."""
    assert PySparkSelectors is not None
    assert hasattr(scs, "initialize")


def test_initialize_patches_dataframe():
    """Verify initialize() patches DataFrame.select."""
    scs.initialize()
    assert hasattr(DataFrame.select, "_cs_original")


# ============================================================================
# Test: Basic Type Selectors (by_dtype)
# ============================================================================


class TestByDtypeSelector:
    """Tests for scs.by_dtype() selector."""

    def test_select_integer_columns(self, df_test):
        """Select all integer-type columns."""
        result = df_test.select(scs.by_dtype([T.IntegerType]))
        assert set(result.columns) == {"int_col"}
        # Verify content: int_col should have values 1000-8000
        rows = result.collect()
        assert len(rows) == 8
        assert rows[0]["int_col"] == 1000

    def test_select_string_columns(self, df_test):
        """Select all string columns."""
        result = df_test.select(scs.by_dtype([T.StringType]))
        assert set(result.columns) == {"string_col"}
        # Verify content: string_col should have values Alice through Hannah
        rows = result.collect()
        assert rows[0]["string_col"] == "Alice"
        assert rows[1]["string_col"] == "Bob"
        assert rows[7]["string_col"] == "Hannah"

    def test_select_double_columns(self, df_test):
        """Select all double-type columns."""
        result = df_test.select(scs.by_dtype([T.DoubleType]))
        assert set(result.columns) == {"double_col"}
        rows = result.collect()
        assert rows[0]["double_col"] == pytest.approx(10.01)

    def test_select_multiple_types(self, df_test):
        """Select multiple types at once."""
        result = df_test.select(scs.by_dtype([T.IntegerType, T.StringType]))
        assert set(result.columns) == {"int_col", "string_col"}

    def test_numeric_includes_decimal(self, df_test):
        """Verify scs.numeric() includes decimal columns."""
        result = df_test.select(scs.numeric())
        numeric_cols = {"byte_col", "short_col", "int_col", "long_col", "float_col", "double_col", "decimal_col"}
        assert set(result.columns) == numeric_cols

    def test_invalid_dtype_raises_error(self, df_test):
        """Invalid data type should raise PySparkTypeError at construction."""
        with pytest.raises(PySparkTypeError):
            scs.by_dtype(["not_a_type"])

    def test_numeric_category_matches_subtypes(self, df_test):
        """NumericType category should match all numeric subtypes."""
        result = df_test.select(scs.by_dtype([T.NumericType]))
        numeric_cols = {"byte_col", "short_col", "int_col", "long_col", "float_col", "double_col", "decimal_col"}
        assert set(result.columns) == numeric_cols


# ============================================================================
# Test: Named Column Selectors
# ============================================================================


class TestNamedSelectors:
    """Tests for name-based selectors."""

    def test_by_name_exact_match(self, df_test):
        """Select column by exact name."""
        result = df_test.select(scs.by_name("string_col"))
        assert result.columns == ["string_col"]
        rows = result.collect()
        assert rows[0]["string_col"] == "Alice"

    def test_by_name_case_insensitive(self, spark):
        """by_name should be case-insensitive."""
        case_df = spark.createDataFrame([(1, "a")], ["Mixed_Case_Col", "other"])
        result = case_df.select(scs.by_name("mixed_case_col"))
        assert result.columns == ["Mixed_Case_Col"]

    def test_all_selector(self, df_test):
        """all() should select every column."""
        result = df_test.select(scs.all())
        assert len(result.columns) == 14
        assert set(result.columns) == set(df_test.columns)

    def test_first_selector(self, df_test):
        """first() should select the first column."""
        result = df_test.select(scs.first())
        assert result.columns == ["byte_col"]

    def test_last_selector(self, df_test):
        """last() should select the last column."""
        result = df_test.select(scs.last())
        assert result.columns == ["map_col"]

    def test_exclude_by_name(self, df_test):
        """exclude() should drop named column."""
        result = df_test.select(scs.exclude("string_col"))
        assert "string_col" not in result.columns
        assert len(result.columns) == 13


# ============================================================================
# Test: Index-Based Selectors
# ============================================================================


class TestIndexSelector:
    """Tests for scs.by_index() selector."""

    def test_by_index_single(self, df_test):
        """Select single column by index."""
        result = df_test.select(scs.by_index(0))
        assert result.columns == ["byte_col"]

    def test_by_index_multiple(self, df_test):
        """Select multiple columns by index."""
        result = df_test.select(scs.by_index(0, 1))
        assert result.columns == ["byte_col", "short_col"]

    def test_by_index_with_range(self, df_test):
        """Select columns using range."""
        result = df_test.select(scs.by_index(range(1, 4), -1))
        assert result.columns == ["short_col", "int_col", "long_col", "map_col"]

    def test_by_index_negative_index(self, df_test):
        """Negative index should select from end."""
        result = df_test.select(scs.by_index(-1))
        assert result.columns == ["map_col"]

    def test_by_index_out_of_bounds_strict(self, df_test):
        """Out of bounds with strict_index_bounds=True should raise."""
        with pytest.raises(PySparkValueError):
            df_test.select(scs.by_index(0, 9999))

    def test_by_index_out_of_bounds_lenient(self, df_test):
        """Out of bounds with strict_index_bounds=False should be tolerated."""
        result = df_test.select(scs.by_index(0, 9999, strict_index_bounds=False))
        assert result.columns == ["byte_col"]


# ============================================================================
# Test: Pattern-Based Selectors
# ============================================================================


class TestPatternSelectors:
    """Tests for regex and string pattern selectors."""

    def test_matches_regex(self, df_test):
        """matches() should select columns matching regex."""
        result = df_test.select(scs.matches(r"(?i)col"))
        # Should match any column with 'col' in the name (case-insensitive)
        expected_cols = {
            "byte_col",
            "short_col",
            "int_col",
            "long_col",
            "float_col",
            "double_col",
            "decimal_col",
            "string_col",
            "binary_col",
            "boolean_col",
            "date_col",
            "timestamp_col",
            "array_col",
            "map_col",
        }
        assert set(result.columns) == expected_cols

    def test_matches_case_insensitive(self, spark):
        """matches() should be case-insensitive."""
        case_df = spark.createDataFrame([(1, "a")], ["Mixed_Case_Col", "other"])
        result = case_df.select(scs.matches(r"^mixed"))
        assert result.columns == ["Mixed_Case_Col"]

    def test_starts_with(self, df_test):
        """starts_with() should select columns starting with string."""
        result = df_test.select(scs.starts_with("float"))
        assert result.columns == ["float_col"]

    def test_ends_with(self, df_test):
        """ends_with() should select columns ending with string."""
        result = df_test.select(scs.ends_with("t_col"))
        # Should match short_col, int_col, float_col (byte_col is 'byte' not matching 't_col')
        expected = {"short_col", "int_col", "float_col"}
        assert set(result.columns) == expected

    def test_contains(self, df_test):
        """contains() should select columns containing string."""
        result = df_test.select(scs.contains("string_"))
        assert result.columns == ["string_col"]


# ============================================================================
# Test: Type-Category Selectors
# ============================================================================


class TestTypeCategorySelectors:
    """Tests for type category convenience selectors."""

    def test_integer_selector(self, df_test):
        """integer() should select all integer types."""
        result = df_test.select(scs.integer())
        expected = {"byte_col", "short_col", "int_col", "long_col"}
        assert set(result.columns) == expected

    def test_floats_selector(self, df_test):
        """floats() should select float types."""
        result = df_test.select(scs.floats())
        assert set(result.columns) == {"float_col", "double_col"}

    def test_string_selector(self, df_test):
        """string() should select string type."""
        result = df_test.select(scs.string())
        assert result.columns == ["string_col"]

    def test_date_selector(self, df_test):
        """date() should select date columns."""
        result = df_test.select(scs.date())
        assert result.columns == ["date_col"]

    def test_datetime_selector(self, df_test):
        """datetime_() should select timestamp columns."""
        result = df_test.select(scs.datetime_())
        assert result.columns == ["timestamp_col"]

    def test_temporal_selector(self, df_test):
        """temporal() should select both date and datetime columns."""
        result = df_test.select(scs.temporal())
        assert set(result.columns) == {"date_col", "timestamp_col"}


# ============================================================================
# Test: Selector Selection Operations (Set Algebra)
# ============================================================================


class TestSelectorSelectionOperations:
    """Tests for set algebra operations on selectors."""

    def test_invert_operation(self, df_test):
        """~ (invert) should select complement of columns."""
        result = df_test.select(~scs.by_dtype([T.StringType]))
        assert "string_col" not in result.columns
        assert len(result.columns) == 13

    def test_difference_operation(self, df_test):
        """- (difference) should select A minus B."""
        result = df_test.select(
            scs.by_dtype([T.StringType, T.IntegerType, T.DoubleType]) - scs.by_dtype([T.IntegerType, T.DoubleType])
        )
        # Only string columns should remain
        assert result.columns == ["string_col"]

    def test_intersection_operation(self, df_test):
        """& (intersection) should select columns in both."""
        result = df_test.select(
            scs.by_dtype([T.DoubleType, T.IntegerType]) & scs.by_dtype([T.DoubleType, T.StringType])
        )
        # Only double columns are in both
        assert result.columns == ["double_col"]

    def test_symmetric_difference_operation(self, df_test):
        """^ (symmetric difference) should select XOR of columns."""
        result = df_test.select(
            scs.by_dtype([T.DoubleType, T.IntegerType]) ^ scs.by_dtype([T.IntegerType, T.StringType])
        )
        # Should be (double + string) minus (int that's in both)
        # Maintains order, so double first, then string
        expected = {"double_col", "string_col"}
        assert set(result.columns) == expected

    def test_union_operation(self, df_test):
        """| (union) should select columns in either."""
        result = df_test.select(scs.by_dtype([T.StringType]) | scs.by_dtype([T.DoubleType]))
        assert set(result.columns) == {"double_col", "string_col"}

    def test_is_selector_true_for_selector(self, df_test):
        """is_selector should return True for selector objects."""
        assert scs.is_selector(scs.by_dtype([T.DoubleType])) is True
        assert scs.is_selector(scs.by_dtype([T.IntegerType]) | scs.by_dtype([T.StringType])) is True

    def test_is_selector_false_for_non_selector(self, df_test):
        """is_selector should return False for non-selector objects."""
        assert scs.is_selector("string_col") is False
        assert scs.is_selector(F.col("string_col")) is False


# ============================================================================
# Test: Arithmetic and Column Operations
# ============================================================================


class TestArithmeticOperations:
    """Tests for arithmetic operations on selected columns."""

    def test_add_operation(self, df_test):
        """Addition on selected columns should work."""
        result = df_test.select(scs.by_dtype([T.IntegerType]) + 1)
        rows = result.collect()
        # int_col values are 1000-8000, so +1 should give 1001-8001
        # Access by column position since name is auto-generated
        assert rows[0][0] == 1001

    def test_subtract_operation(self, df_test):
        """Subtraction on selected columns should work."""
        result = df_test.select(scs.by_dtype([T.IntegerType]) - 1)
        rows = result.collect()
        assert rows[0][0] == 999

    def test_multiply_operation(self, df_test):
        """Multiplication on selected columns should work."""
        result = df_test.select(scs.by_dtype([T.IntegerType]) * 2)
        rows = result.collect()
        assert rows[0][0] == 2000

    def test_string_upper_transform(self, df_test):
        """String upper() transform should work on selected columns."""
        result = df_test.select(scs.by_dtype([T.StringType]).upper())
        rows = result.collect()
        # string_col has values Alice, Bob, etc.; .upper() should make ALICE, BOB, etc.
        # Access by position since alias doesn't work on selector expressions
        assert rows[0][0] == "ALICE"
        assert rows[1][0] == "BOB"

    def test_string_lower_transform(self, df_test):
        """String lower() transform should work on selected columns."""
        result = df_test.select(scs.by_dtype([T.StringType]).lower())
        rows = result.collect()
        # string_col has values Alice, Bob, etc.; .lower() should make alice, bob, etc.
        assert rows[0][0] == "alice"
        assert rows[1][0] == "bob"

    def test_multiple_transforms(self, df_test):
        """Multiple selector-driven transforms should work together."""
        result = df_test.select((scs.by_dtype([T.IntegerType]) + 1), scs.by_dtype([T.StringType]).upper())
        rows = result.collect()
        # First column is int+1, second is string.upper()
        assert rows[0][0] == 1001
        assert rows[0][1] == "ALICE"


# ============================================================================
# Test: DataFrame Method Overrides
# ============================================================================


class TestFilterMethod:
    """Tests for filter() with selector conditions."""

    def test_filter_single_condition(self, df_test):
        """Filter with single selector condition."""
        result = df_test.select(scs.by_dtype([T.IntegerType])).filter(scs.by_dtype([T.IntegerType]) > 5000)
        rows = result.collect()
        # int_col values are 1000-8000, so >5000 gives 6000, 7000, 8000
        assert len(rows) == 3
        assert rows[0]["int_col"] == 6000

    def test_filter_and_conditions(self, df_test):
        """Filter with AND of two conditions."""
        result = df_test.select(scs.by_dtype([T.IntegerType])).filter(
            (scs.by_dtype([T.IntegerType]) > 0) & (scs.by_dtype([T.IntegerType]) < 5000)
        )
        rows = result.collect()
        # Should get 1000, 2000, 3000, 4000
        assert len(rows) == 4

    def test_filter_selector_and_plain_column(self, df_test):
        """Filter mixing selector and plain column conditions."""
        result = df_test.filter((scs.by_dtype([T.IntegerType]) > 0) & (F.col("string_col").isNotNull()))
        rows = result.collect()
        # All rows should pass (all have int_col > 0 and non-null string_col)
        assert len(rows) == 8

    def test_filter_inverted_condition(self, df_test):
        """Filter with inverted selector condition."""
        result = df_test.select(scs.by_dtype([T.IntegerType])).filter(~(scs.by_dtype([T.IntegerType]) > 5000))
        rows = result.collect()
        # Should get 1000-5000
        assert len(rows) == 5


class TestCastMethod:
    """Tests for cast() to change column types."""

    def test_cast_to_string(self, df_test):
        """Cast integer columns to string."""
        result = df_test.select(scs.by_dtype([T.IntegerType]).cast("string"))
        # Check dtypes
        dtypes = result.dtypes
        # All int columns should now be string
        assert all(dtype == "string" for col, dtype in dtypes)
        # Check content
        rows = result.collect()
        assert rows[0]["int_col"] == "1000"


class TestWithColumnsMethod:
    """Tests for withColumns() to add/mutate columns."""

    def test_withcolumns_single_mutation(self, df_test):
        """Add single selector-driven mutation."""
        result = df_test.withColumns(scs.by_dtype([T.DoubleType]).cast("string"))
        assert "double_col" in result.columns
        # Check it's now string type
        double_dtype = [dtype for col, dtype in result.dtypes if col == "double_col"][0]
        assert double_dtype == "string"
        rows = result.collect()
        assert rows[0]["double_col"] == "10.01"

    def test_withcolumns_multiple_mutations(self, df_test):
        """Apply multiple mutations in one withColumns call."""
        result = df_test.withColumns(scs.by_dtype([T.DoubleType]).cast("string"), scs.by_dtype([T.StringType]).upper())
        rows = result.collect()
        # double_col should be cast to string
        assert rows[0]["double_col"] == "10.01"
        # string_col should be uppercased
        assert rows[0]["string_col"] == "ALICE"

    def test_withcolumns_duplicate_column_raises(self, df_test):
        """Targeting same column twice should raise error."""
        with pytest.raises(PySparkValueError):
            df_test.withColumns(scs.by_dtype([T.IntegerType]) + 1, scs.by_dtype([T.IntegerType]).cast("string"))

    def test_withcolumn_single_selector(self, df_test):
        """withColumn with selector should work."""
        result = df_test.withColumn("new_col", scs.by_dtype([T.IntegerType]) + 1)
        assert "new_col" in result.columns
        rows = result.collect()
        assert rows[0]["new_col"] == 1001


class TestDropMethod:
    """Tests for drop() to remove columns."""

    def test_drop_by_selector(self, df_test):
        """Drop columns selected by selector."""
        # Drop the intersection of (string+double) and (double) = just double
        result = df_test.drop(scs.by_dtype([T.StringType, T.DoubleType]) & scs.by_dtype([T.DoubleType]))
        assert "double_col" not in result.columns
        assert "string_col" in result.columns


class TestGroupByAggMethod:
    """Tests for groupBy() and agg() with selectors."""

    def test_groupby_agg_basic(self, df_test):
        """Group by string column, sum double columns."""
        result = df_test.groupBy(scs.by_dtype([T.StringType])).agg(scs.by_dtype([T.DoubleType]).sum())
        rows = result.collect()
        # Each row should have string_col and the sum of double_col
        assert len(rows) == 8  # 8 unique names
        # Find Alice's row and check the aggregated value (second column)
        alice_row = [r for r in rows if r["string_col"] == "Alice"][0]
        assert alice_row[1] == pytest.approx(10.01)


class TestSortMethod:
    """Tests for sort() and orderBy() with selectors."""

    def test_sort_descending(self, df_test):
        """Sort by double column in descending order."""
        result = df_test.sort(scs.by_dtype([T.DoubleType]).desc())
        rows = result.collect()
        # Should be sorted descending: 80.08, 70.07, ..., 10.01
        assert rows[0]["double_col"] == pytest.approx(80.08)
        assert rows[7]["double_col"] == pytest.approx(10.01)

    def test_orderby_descending(self, df_test):
        """orderBy should work same as sort."""
        result = df_test.orderBy(scs.by_dtype([T.DoubleType]).desc())
        rows = result.collect()
        assert rows[0]["double_col"] == pytest.approx(80.08)
        assert rows[7]["double_col"] == pytest.approx(10.01)


class TestPivotMethod:
    """Tests for pivot() with selectors."""

    def test_pivot_basic(self, df_test):
        """Group, pivot, and aggregate with selectors."""
        result = df_test.groupBy(scs.by_dtype([T.StringType])).pivot("int_col").agg(scs.by_dtype([T.DoubleType]).sum())

        # Should have string_col plus pivot columns for each unique int_col value
        rows = result.collect()
        assert len(rows) == 8  # 8 unique names


# ============================================================================
# Test: Rename Operations (prefix, suffix, map_alias)
# ============================================================================


class TestRenameOperations:
    """Tests for column renaming with prefix, suffix, map_alias."""

    def test_prefix_operation(self, df_test):
        """Prefix operation should prepend to column names."""
        result = df_test.select(scs.by_name("string_col").prefix("pre_"))
        assert result.columns == ["pre_string_col"]
        rows = result.collect()
        assert rows[0]["pre_string_col"] == "Alice"

    def test_suffix_operation(self, df_test):
        """Suffix operation should append to column names."""
        result = df_test.select(scs.by_name("string_col").suffix("_suf"))
        assert result.columns == ["string_col_suf"]
        rows = result.collect()
        assert rows[0]["string_col_suf"] == "Alice"

    def test_prefix_and_suffix_chained(self, df_test):
        """Prefix and suffix should compose correctly."""
        result = df_test.select(scs.by_name("string_col").prefix("pre_").suffix("_suf"))
        assert result.columns == ["pre_string_col_suf"]

    def test_map_alias_function(self, df_test):
        """map_alias with custom function should work."""
        result = df_test.select(scs.by_name("string_col").map_alias(lambda n: n.upper()))
        assert result.columns == ["STRING_COL"]
        rows = result.collect()
        assert rows[0]["STRING_COL"] == "Alice"

    def test_prefix_after_transform(self, df_test):
        """Prefix after transform (cast/upper) should use original name."""
        result = df_test.select(scs.by_dtype([T.StringType]).upper().prefix("up_"))
        # Output column should be up_string_col, not up_upper(string_col)
        assert result.columns == ["up_string_col"]
        rows = result.collect()
        assert rows[0]["up_string_col"] == "ALICE"

    def test_prefix_after_arithmetic(self, df_test):
        """Prefix after arithmetic should use original name."""
        result = df_test.select((scs.by_dtype([T.IntegerType]) + 1).prefix("plus_one_"))
        # Output should be plus_one_int_col, not plus_one_(int_col + 1)
        assert result.columns == ["plus_one_int_col"]

    def test_prefix_on_inverted_selector(self, df_test):
        """Prefix with inverted selector should work (complement mode)."""
        result = df_test.select(~scs.by_dtype([T.StringType]).prefix("not_string_"))
        # Should get all non-string columns with not_string_ prefix
        assert "not_string_byte_col" in result.columns
        assert "not_string_int_col" in result.columns
        assert "string_col" not in result.columns

    def test_withcolumn_rename(self, df_test):
        """Renaming in withColumn should add new column."""
        result = df_test.withColumn("renamed_int", scs.by_name("int_col"))
        assert "renamed_int" in result.columns
        assert "int_col" in result.columns
        rows = result.collect()
        assert rows[0]["renamed_int"] == 1000

    def test_withcolumns_rename(self, df_test):
        """Renaming in withColumns should apply cast + rename."""
        result = df_test.withColumns(scs.by_dtype([T.IntegerType]).cast("string").prefix("str_"))
        assert "str_int_col" in result.columns
        rows = result.collect()
        assert rows[0]["str_int_col"] == "1000"

    def test_groupby_agg_rename(self, df_test):
        """Rename in agg should affect aggregated output column."""
        result = df_test.groupBy(scs.by_name("string_col")).agg(scs.by_dtype([T.DoubleType]).sum().prefix("total_"))
        assert "total_double_col" in result.columns
        rows = result.collect()
        alice_row = [r for r in rows if r["string_col"] == "Alice"][0]
        assert alice_row["total_double_col"] == pytest.approx(10.01)


# ============================================================================
# Test: Compatibility and Edge Cases
# ============================================================================


class TestCompatibility:
    """Tests for compatibility features."""

    def test_decimal_in_numeric_selector(self, spark):
        """Decimal should be included in numeric() selector."""
        dec_df = spark.createDataFrame(
            [(1, 1.5)],
            ["int_col", "dbl_col"],
        ).withColumn("dec_col", F.col("dbl_col").cast("decimal(10,2)"))

        result = dec_df.select(scs.numeric())
        assert set(result.columns) == {"int_col", "dbl_col", "dec_col"}

    def test_integral_type_category(self, spark):
        """Integer category should match only integral types, not decimal."""
        dec_df = spark.createDataFrame(
            [(1, 1.5)],
            ["int_col", "dbl_col"],
        ).withColumn("dec_col", F.col("dbl_col").cast("decimal(10,2)"))

        result = dec_df.select(scs.integer())
        # Should only have int_col, not decimal
        assert result.columns == ["int_col"]

    def test_dot_in_column_name_select(self, spark):
        """Columns with dots should be selectable by name."""
        dot_df = spark.createDataFrame(
            [(1, "a"), (2, "b")],
            ["plain_col", "meta.source"],
        )
        result = dot_df.select(scs.by_name("meta.source"))
        assert result.columns == ["meta.source"]
        rows = result.collect()
        assert rows[0]["meta.source"] == "a"

    def test_dot_in_column_name_filter(self, spark):
        """Filter using column with dot should work."""
        dot_df = spark.createDataFrame(
            [(1, "a"), (2, "b")],
            ["plain_col", "meta.source"],
        )
        result = dot_df.filter(scs.by_name("meta.source") == "a")
        rows = result.collect()
        assert len(rows) == 1
        assert rows[0]["meta.source"] == "a"

    def test_dot_in_renamed_column(self, spark):
        """Prefix with dot should work in renamed column."""
        dot_df = spark.createDataFrame(
            [(1, "a"), (2, "b")],
            ["plain_col", "other"],
        )
        result = dot_df.withColumns(scs.by_name("plain_col").prefix("meta."))
        assert "meta.plain_col" in result.columns
        rows = result.collect()
        assert rows[0]["meta.plain_col"] == 1


class TestComplexScenarios:
    """Tests for complex real-world scenarios."""

    def test_combined_selectors_with_filters_and_transforms(self, df_test):
        """Complex query: select, filter, transform."""
        result = df_test.select(scs.by_dtype([T.StringType]).upper(), scs.by_dtype([T.IntegerType])).filter(
            scs.by_dtype([T.IntegerType]) > 3000
        )
        rows = result.collect()
        assert len(rows) == 5  # int_col values 4000-8000
        # First column is uppercased string, second is the int value
        assert rows[0][0] == "DIANA"
        assert rows[0][1] == 4000

    def test_multiple_selectors_same_call(self, df_test):
        """Using multiple different selectors in one call."""
        result = df_test.select(
            scs.by_index(0),
            scs.by_name("string_col"),
            scs.ends_with("_col"),
        )
        # byte_col, string_col, and ends_with matches
        # ends_with should match many columns
        assert "byte_col" in result.columns
        assert "string_col" in result.columns

    def test_selector_composition_with_arithmetic(self, df_test):
        """Compose selectors with arithmetic operations."""
        result = df_test.select((scs.by_index(0) + 10), (scs.by_index(2) + 10))
        rows = result.collect()
        # byte_col=1-8 (+10 = 11-18), int_col=1000-8000 (+10 = 1010-8010)
        assert rows[0][0] == 11
        assert rows[0][1] == 1010
