# This source file is a PySpark script that executes as a linear pipeline without
# any reusable functions or methods. All logic is inline at module level.
# 
# However, the core business transformations (date parsing, amount cleaning,
# deduplication, column selection) represent meaningful logic worth testing.
# We test these transformation rules by recreating them in isolated test scenarios
# using PySpark's local mode.

import pytest
from decimal import Decimal
from datetime import date, datetime

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, regexp_replace, to_date, when, lit, current_timestamp,
    max as spark_max
)
from pyspark.sql.types import (
    StructType, StructField, StringType, DecimalType, DateType, TimestampType
)


@pytest.fixture(scope="session")
def spark():
    """Create a local SparkSession for testing."""
    session = SparkSession.builder \
        .master("local[1]") \
        .appName("Silver_Deposits_Test") \
        .config("spark.sql.shuffle.partitions", "1") \
        .getOrCreate()
    yield session
    session.stop()


@pytest.fixture
def sample_bronze_schema():
    return StructType([
        StructField("deposit_id", StringType(), True),
        StructField("account_id", StringType(), True),
        StructField("account_number", StringType(), True),
        StructField("phone_number", StringType(), True),
        StructField("email_address", StringType(), True),
        StructField("branch_id", StringType(), True),
        StructField("deposit_timestamp", StringType(), True),
        StructField("amount", StringType(), True),
        StructField("ingestion_timestamp", TimestampType(), True),
    ])


class TestDepositDateTransformation:
    """Test Transformation (a): Convert deposit_timestamp to deposit_date."""

    def test_valid_date_parsing(self, spark):
        """Happy path: standard dd-MM-yyyy HH:mm format is parsed correctly."""
        df = spark.createDataFrame(
            [("08-01-2026 14:00",)],
            ["deposit_timestamp"]
        )
        result = df.withColumn(
            "deposit_date",
            to_date(col("deposit_timestamp"), "dd-MM-yyyy HH:mm")
        )
        row = result.collect()[0]
        assert row["deposit_date"] == date(2026, 1, 8)

    def test_date_parsing_different_time(self, spark):
        """Verify time component is discarded, only date is kept."""
        df = spark.createDataFrame(
            [("25-12-2023 23:59",)],
            ["deposit_timestamp"]
        )
        result = df.withColumn(
            "deposit_date",
            to_date(col("deposit_timestamp"), "dd-MM-yyyy HH:mm")
        )
        row = result.collect()[0]
        assert row["deposit_date"] == date(2023, 12, 25)

    def test_date_parsing_null_input(self, spark):
        """Null deposit_timestamp should result in null deposit_date."""
        df = spark.createDataFrame(
            [(None,)],
            ["deposit_timestamp"]
        )
        result = df.withColumn(
            "deposit_date",
            to_date(col("deposit_timestamp"), "dd-MM-yyyy HH:mm")
        )
        row = result.collect()[0]
        assert row["deposit_date"] is None

    def test_date_parsing_invalid_format(self, spark):
        """Invalid format string should result in null."""
        df = spark.createDataFrame(
            [("2026-01-08 14:00:00",)],  # yyyy-MM-dd format, not dd-MM-yyyy
            ["deposit_timestamp"]
        )
        result = df.withColumn(
            "deposit_date",
            to_date(col("deposit_timestamp"), "dd-MM-yyyy HH:mm")
        )
        row = result.collect()[0]
        assert row["deposit_date"] is None

    def test_date_parsing_empty_string(self, spark):
        """Empty string should result in null."""
        df = spark.createDataFrame(
            [("",)],
            ["deposit_timestamp"]
        )
        result = df.withColumn(
            "deposit_date",
            to_date(col("deposit_timestamp"), "dd-MM-yyyy HH:mm")
        )
        row = result.collect()[0]
        assert row["deposit_date"] is None


class TestAmountTransformation:
    """Test Transformation (b): Strip '$' and commas, cast to Decimal."""

    def test_amount_with_dollar_and_comma(self, spark):
        """Happy path: '$44,196.98' becomes 44196.98."""
        df = spark.createDataFrame(
            [("$44,196.98",)],
            ["amount"]
        )
        result = df.withColumn(
            "amount",
            regexp_replace(col("amount"), "[$,]", "").cast(DecimalType(15, 2))
        )
        row = result.collect()[0]
        assert row["amount"] == Decimal("44196.98")

    def test_amount_with_dollar_no_comma(self, spark):
        """Amount like '$48604.41' without commas."""
        df = spark.createDataFrame(
            [("$48604.41",)],
            ["amount"]
        )
        result = df.withColumn(
            "amount",
            regexp_replace(col("amount"), "[$,]", "").cast(DecimalType(15, 2))
        )
        row = result.collect()[0]
        assert row["amount"] == Decimal("48604.41")

    def test_amount_with_multiple_commas(self, spark):
        """Amount like '$1,234,567.89' with multiple commas."""
        df = spark.createDataFrame(
            [("$1,234,567.89",)],
            ["amount"]
        )
        result = df.withColumn(
            "amount",
            regexp_replace(col("amount"), "[$,]", "").cast(DecimalType(15, 2))
        )
        row = result.collect()[0]
        assert row["amount"] == Decimal("1234567.89")

    def test_amount_plain_number(self, spark):
        """Amount without any special characters."""
        df = spark.createDataFrame(
            [("1000.50",)],
            ["amount"]
        )
        result = df.withColumn(
            "amount",
            regexp_replace(col("amount"), "[$,]", "").cast(DecimalType(15, 2))
        )
        row = result.collect()[0]
        assert row["amount"] == Decimal("1000.50")

    def test_amount_null(self, spark):
        """Null amount should remain null."""
        df = spark.createDataFrame(
            [(None,)],
            schema=StructType([StructField("amount", StringType(), True)])
        )
        result = df.withColumn(
            "amount",
            regexp_replace(col("amount"), "[$,]", "").cast(DecimalType(15, 2))
        )
        row = result.collect()[0]
        assert row["amount"] is None

    def test_amount_zero(self, spark):
        """Amount of '$0.00'."""
        df = spark.createDataFrame(
            [("$0.00",)],
            ["amount"]
        )
        result = df.withColumn(
            "amount",
            regexp_replace(col("amount"), "[$,]", "").cast(DecimalType(15, 2))
        )
        row = result.collect()[0]
        assert row["amount"] == Decimal("0.00")

    def test_amount_invalid_string(self, spark):
        """Non-numeric string after stripping should result in null."""
        df = spark.createDataFrame(
            [("$abc",)],
            ["amount"]
        )
        result = df.withColumn(
            "amount",
            regexp_replace(col("amount"), "[$,]", "").cast(DecimalType(15, 2))
        )
        row = result.collect()[0]
        assert row["amount"] is None


class TestDeduplication:
    """Test deduplication logic on account_id."""

    def test_dedup_removes_duplicate_account_ids(self, spark, sample_bronze_schema):
        """Only one record per account_id should remain after deduplication."""
        ts = datetime(2024, 1, 1, 12, 0, 0)
        data = [
            ("D001", "A001", "ACC001", "555-0001", "a@b.com", "B01", "08-01-2026 14:00", "$100.00", ts),
            ("D002", "A001", "ACC001", "555-0001", "a@b.com", "B01", "09-01-2026 15:00", "$200.00", ts),
            ("D003", "A002", "ACC002", "555-0002", "c@d.com", "B02", "10-01-2026 16:00", "$300.00", ts),
        ]
        df = spark.createDataFrame(data, schema=sample_bronze_schema)
        deduplicated = df.dropDuplicates(["account_id"])

        # Should have 2 unique account_ids
        assert deduplicated.count() == 2
        account_ids = [row["account_id"] for row in deduplicated.collect()]
        assert "A001" in account_ids
        assert "A002" in account_ids

    def test_dedup_no_duplicates(self, spark, sample_bronze_schema):
        """When no duplicates exist, all records are retained."""
        ts = datetime(2024, 1, 1, 12, 0, 0)
        data = [
            ("D001", "A001", "ACC001", "555-0001", "a@b.com", "B01", "08-01-2026 14:00", "$100.00", ts),
            ("D002", "A002", "ACC002", "555-0002", "c@d.com", "B02", "09-01-2026 15:00", "$200.00", ts),
        ]
        df = spark.createDataFrame(data, schema=sample_bronze_schema)
        deduplicated = df.dropDuplicates(["account_id"])
        assert deduplicated.count() == 2

    def test_dedup_all_same_account_id(self, spark, sample_bronze_schema):
        """All records with same account_id should collapse to one."""
        ts = datetime(2024, 1, 1, 12, 0, 0)
        data = [
            ("D001", "A001", "ACC001", "555-0001", "a@b.com", "B01", "08-01-2026 14:00", "$100.00", ts),
            ("D002", "A001", "ACC001", "555-0001", "a@b.com", "B01", "09-01-2026 15:00", "$200.00", ts),
            ("D003", "A001", "ACC001", "555-0001", "a@b.com", "B01", "10-01-2026 16:00", "$300.00", ts),
        ]
        df = spark.createDataFrame(data, schema=sample_bronze_schema)
        deduplicated = df.dropDuplicates(["account_id"])
        assert deduplicated.count() == 1


class TestLatestIngestionFilter:
    """Test filtering by latest ingestion_timestamp."""

    def test_filter_latest_batch(self, spark, sample_bronze_schema):
        """Only records with the max ingestion_timestamp should be retained."""
        ts_old = datetime(2024, 1, 1, 10, 0, 0)
        ts_new = datetime(2024, 1, 2, 12, 0, 0)
        data = [
            ("D001", "A001", "ACC001", "555-0001", "a@b.com", "B01", "08-01-2026 14:00", "$100.00", ts_old),
            ("D002", "A002", "ACC002", "555-0002", "c@d.com", "B02", "09-01-2026 15:00", "$200.00", ts_new),
            ("D003", "A003", "ACC003", "555-0003", "e@f.com", "B03", "10-01-2026 16:00", "$300.00", ts_new),
        ]
        df = spark.createDataFrame(data, schema=sample_bronze_schema)

        latest_ts = df.select(spark_max(col("ingestion_timestamp"))).collect()[0][0]
        filtered = df.filter(col("ingestion_timestamp") == latest_ts)

        assert filtered.count() == 2
        deposit_ids = [row["deposit_id"] for row in filtered.collect()]
        assert "D001" not in deposit_ids
        assert "D002" in deposit_ids
        assert "D003" in deposit_ids

    def test_filter_all_same_timestamp(self, spark, sample_bronze_schema):
        """When all records have same timestamp, all are retained."""
        ts = datetime(2024, 1, 1, 12, 0, 0)
        data = [
            ("D001", "A001", "ACC001", "555-0001", "a@b.com", "B01", "08-01-2026 14:00", "$100.00", ts),
            ("D002", "A002", "ACC002", "555-0002", "c@d.com", "B02", "09-01-2026 15:00", "$200.00", ts),
        ]
        df = spark.createDataFrame(data, schema=sample_bronze_schema)

        latest_ts = df.select(spark_max(col("ingestion_timestamp"))).collect()[0][0]
        filtered = df.filter(col("ingestion_timestamp") == latest_ts)

        assert filtered.count() == 2


class TestColumnSelectionAndCasting:
    """Test the final select/cast/alias logic for silver schema."""

    def test_column_renaming_email(self, spark, sample_bronze_schema):
        """email_address should be aliased to 'email' in silver output."""
        ts = datetime(2024, 1, 1, 12, 0, 0)
        data = [
            ("D001", "A001", "ACC001", "555-0001", "user@example.com", "B01", "08-01-2026 14:00", "$1,000.00", ts),
        ]
        df = spark.createDataFrame(data, schema=sample_bronze_schema)

        # Apply transformations
        transformed = df.withColumn(
            "deposit_date",
            to_date(col("deposit_timestamp"), "dd-MM-yyyy HH:mm")
        ).withColumn(
            "amount",
            regexp_replace(col("amount"), "[$,]", "").cast(DecimalType(15, 2))
        )

        silver_df = transformed.select(
            col("deposit_id").cast(StringType()).alias("deposit_id"),
            col("account_id").cast(StringType()).alias("account_id"),
            col("account_number").cast(StringType()).alias("account_number"),
            col("phone_number").cast(StringType()).alias("phone_number"),
            col("email_address").cast(StringType()).alias("email"),
            col("branch_id").cast(StringType()).alias("branch_id"),
            col("deposit_date").cast(DateType()).alias("deposit_date"),
            col("amount")
        )

        row = silver_df.collect()[0]
        assert row["email"] == "user@example.com"
        assert row["deposit_id"] == "D001"
        assert row["account_id"] == "A001"
        assert row["account_number"] == "ACC001"
        assert row["phone_number"] == "555-0001"
        assert row["branch_id"] == "B01"
        assert row["deposit_date"] == date(2026, 1, 8)
        assert row["amount"] == Decimal("1000.00")

    def test_silver_schema_columns(self, spark, sample_bronze_schema):
        """Silver output should have exactly the expected columns."""
        ts = datetime(2024, 1, 1, 12, 0, 0)
        data = [
            ("D001", "A001", "ACC001", "555-0001", "a@b.com", "B01", "08-01-2026 14:00", "$100.00", ts),
        ]
        df = spark.createDataFrame(data, schema=sample_bronze_schema)

        transformed = df.withColumn(
            "deposit_date",
            to_date(col("deposit_timestamp"), "dd-MM-yyyy HH:mm")
        ).withColumn(
            "amount",
            regexp_replace(col("amount"), "[$,]", "").cast(DecimalType(15, 2))
        )

        silver_df = transformed.select(
            col("deposit_id").cast(StringType()).alias("deposit_id"),
            col("account_id").cast(StringType()).alias("account_id"),
            col("account_number").cast(StringType()).alias("account_number"),
            col("phone_number").cast(StringType()).alias("phone_number"),
            col("email_address").cast(StringType()).alias("email"),
            col("branch_id").cast(StringType()).alias("branch_id"),
            col("deposit_date").cast(DateType()).alias("deposit_date"),
            col("amount")
        )

        expected_columns = [
            "deposit_id", "account_id", "account_number",
            "phone_number", "email", "branch_id", "deposit_date", "amount"
        ]
        assert silver_df.columns == expected_columns


class TestEndToEndTransformation:
    """Integration test combining all transformation steps."""

    def test_full_pipeline_logic(self, spark, sample_bronze_schema):
        """Test the complete transformation pipeline end-to-end."""
        ts_old = datetime(2024, 1, 1, 10, 0, 0)
        ts_new = datetime(2024, 1, 2, 12, 0, 0)
        data = [
            # Old batch - should be filtered out
            ("D001", "A001", "ACC001", "555-0001", "old@test.com", "B01", "01-01-2026 10:00", "$500.00", ts_old),
            # New batch - duplicate account_id A002
            ("D002", "A002", "ACC002", "555-0002", "new1@test.com", "B02", "08-01-2026 14:00", "$44,196.98", ts_new),
            ("D003", "A002", "ACC002", "555-0002", "new1@test.com", "B02", "09-01-2026 15:00", "$1,000.00", ts_new),
            # New batch - unique account_id
            ("D004", "A003", "ACC003", "555-0003", "new2@test.com", "B03", "15-03-2026 09:30", "$48604.41", ts_new),
        ]
        df = spark.createDataFrame(data, schema=sample_bronze_schema)

        # Step 1: Filter latest batch
        latest_ts = df.select(spark_max(col("ingestion_timestamp"))).collect()[0][0]
        latest_batch = df.filter(col("ingestion_timestamp") == latest_ts)
        assert latest_batch.count() == 3

        # Step 2: Deduplicate
        deduplicated = latest_batch.dropDuplicates(["account_id"])
        assert deduplicated.count() == 2

        # Step 3: Transform
        transformed = deduplicated.withColumn(
            "deposit_date",
            to_date(col("deposit_timestamp"), "dd-MM-yyyy HH:mm")
        ).withColumn(
            "amount",
            regexp_replace(col("amount"), "[$,]", "").cast(DecimalType(15, 2))
        )

        # Step 4: Select
        silver_df = transformed.select(
            col("deposit_id").cast(StringType()).alias("deposit_id"),
            col("account_id").cast(StringType()).alias("account_id"),
            col("account_number").cast(StringType()).alias("account_number"),
            col("phone_number").cast(StringType()).alias("phone_number"),
            col("email_address").cast(StringType()).alias("email"),
            col("branch_id").cast(StringType()).alias("branch_id"),
            col("deposit_date").cast(DateType()).alias("deposit_date"),
            col("amount")
        )

        results = silver_df.collect()
        assert len(results) == 2

        # Verify A003 record
        a003_rows = [r for r in results if r["account_id"] == "A003"]
        assert len(a003_rows) == 1
        a003 = a003_rows[0]
        assert a003["deposit_date"] == date(2026, 3, 15)
        assert a003["amount"] == Decimal("48604.41")
        assert a003["email"] == "new2@test.com"