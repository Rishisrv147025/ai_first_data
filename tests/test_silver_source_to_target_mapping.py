# This source file is a PySpark script that executes transformations procedurally
# (no reusable functions or methods). However, we can test the core transformation
# logic by recreating the key steps with a local SparkSession and verifying behavior.

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, to_date
from pyspark.sql.types import (
    StructType, StructField, StringType, TimestampType, DecimalType, DateType
)
from datetime import datetime, date
from decimal import Decimal


@pytest.fixture(scope="session")
def spark():
    """Create a local SparkSession for testing."""
    session = SparkSession.builder \
        .master("local[1]") \
        .appName("TestSilverDeposits") \
        .config("spark.sql.shuffle.partitions", "1") \
        .getOrCreate()
    yield session
    session.stop()


@pytest.fixture
def sample_schema():
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


class TestLatestBatchFiltering:
    """Test Step 1: Filtering records by latest ingestion timestamp."""

    def test_filters_only_latest_batch(self, spark, sample_schema):
        ts_old = datetime(2024, 1, 1, 10, 0, 0)
        ts_new = datetime(2024, 1, 2, 10, 0, 0)
        data = [
            ("D1", "A1", "ACC1", "111", "a@b.com", "B1", "08-01-2026 14:00", "$1,000.00", ts_old),
            ("D2", "A2", "ACC2", "222", "c@d.com", "B2", "09-01-2026 15:00", "$2,000.00", ts_new),
            ("D3", "A3", "ACC3", "333", "e@f.com", "B3", "10-01-2026 16:00", "$3,000.00", ts_new),
        ]
        bronze_df = spark.createDataFrame(data, schema=sample_schema)

        from pyspark.sql.functions import max as spark_max
        latest_ts = bronze_df.select(spark_max(col("ingestion_timestamp"))).collect()[0][0]
        latest_batch_df = bronze_df.filter(col("ingestion_timestamp") == latest_ts)

        assert latest_batch_df.count() == 2
        ids = [row["deposit_id"] for row in latest_batch_df.collect()]
        assert "D2" in ids
        assert "D3" in ids
        assert "D1" not in ids

    def test_all_records_same_timestamp(self, spark, sample_schema):
        ts = datetime(2024, 1, 1, 10, 0, 0)
        data = [
            ("D1", "A1", "ACC1", "111", "a@b.com", "B1", "08-01-2026 14:00", "$1,000.00", ts),
            ("D2", "A2", "ACC2", "222", "c@d.com", "B2", "09-01-2026 15:00", "$2,000.00", ts),
        ]
        bronze_df = spark.createDataFrame(data, schema=sample_schema)

        from pyspark.sql.functions import max as spark_max
        latest_ts = bronze_df.select(spark_max(col("ingestion_timestamp"))).collect()[0][0]
        latest_batch_df = bronze_df.filter(col("ingestion_timestamp") == latest_ts)

        assert latest_batch_df.count() == 2


class TestDeduplication:
    """Test Step 2: Deduplication on account_id."""

    def test_deduplicates_on_account_id(self, spark, sample_schema):
        ts = datetime(2024, 1, 1, 10, 0, 0)
        data = [
            ("D1", "A1", "ACC1", "111", "a@b.com", "B1", "08-01-2026 14:00", "$1,000.00", ts),
            ("D2", "A1", "ACC1", "111", "a@b.com", "B1", "09-01-2026 15:00", "$2,000.00", ts),
            ("D3", "A2", "ACC2", "222", "c@d.com", "B2", "10-01-2026 16:00", "$3,000.00", ts),
        ]
        df = spark.createDataFrame(data, schema=sample_schema)
        deduplicated_df = df.dropDuplicates(["account_id"])

        assert deduplicated_df.count() == 2
        account_ids = [row["account_id"] for row in deduplicated_df.collect()]
        assert "A1" in account_ids
        assert "A2" in account_ids

    def test_no_duplicates_returns_all(self, spark, sample_schema):
        ts = datetime(2024, 1, 1, 10, 0, 0)
        data = [
            ("D1", "A1", "ACC1", "111", "a@b.com", "B1", "08-01-2026 14:00", "$1,000.00", ts),
            ("D2", "A2", "ACC2", "222", "c@d.com", "B2", "09-01-2026 15:00", "$2,000.00", ts),
        ]
        df = spark.createDataFrame(data, schema=sample_schema)
        deduplicated_df = df.dropDuplicates(["account_id"])

        assert deduplicated_df.count() == 2


class TestDepositTimestampToDate:
    """Test Step 3a: Converting deposit_timestamp string to date."""

    def test_valid_timestamp_conversion(self, spark):
        data = [("08-01-2026 14:00",), ("25-12-2025 09:30",)]
        df = spark.createDataFrame(data, ["deposit_timestamp"])
        result_df = df.withColumn(
            "deposit_date",
            to_date(col("deposit_timestamp"), "dd-MM-yyyy HH:mm")
        )
        rows = result_df.collect()

        assert rows[0]["deposit_date"] == date(2026, 1, 8)
        assert rows[1]["deposit_date"] == date(2025, 12, 25)

    def test_invalid_timestamp_returns_null(self, spark):
        data = [("not-a-date",), ("",), (None,)]
        df = spark.createDataFrame(data, ["deposit_timestamp"])
        result_df = df.withColumn(
            "deposit_date",
            to_date(col("deposit_timestamp"), "dd-MM-yyyy HH:mm")
        )
        rows = result_df.collect()

        for row in rows:
            assert row["deposit_date"] is None

    def test_different_format_returns_null(self, spark):
        # Format is yyyy-MM-dd instead of dd-MM-yyyy
        data = [("2026-01-08 14:00",)]
        df = spark.createDataFrame(data, ["deposit_timestamp"])
        result_df = df.withColumn(
            "deposit_date",
            to_date(col("deposit_timestamp"), "dd-MM-yyyy HH:mm")
        )
        rows = result_df.collect()
        assert rows[0]["deposit_date"] is None


class TestAmountCleansing:
    """Test Step 3b: Removing '$' and commas from amount, casting to decimal."""

    def test_amount_with_dollar_and_commas(self, spark):
        data = [("$44,196.98",), ("$48604.41",), ("$1,234,567.89",)]
        df = spark.createDataFrame(data, ["amount"])
        result_df = df.withColumn(
            "amount",
            regexp_replace(col("amount"), "[$,]", "").cast(DecimalType(15, 2))
        )
        rows = result_df.collect()

        assert rows[0]["amount"] == Decimal("44196.98")
        assert rows[1]["amount"] == Decimal("48604.41")
        assert rows[2]["amount"] == Decimal("1234567.89")

    def test_amount_without_special_chars(self, spark):
        data = [("1000.50",), ("0.01",)]
        df = spark.createDataFrame(data, ["amount"])
        result_df = df.withColumn(
            "amount",
            regexp_replace(col("amount"), "[$,]", "").cast(DecimalType(15, 2))
        )
        rows = result_df.collect()

        assert rows[0]["amount"] == Decimal("1000.50")
        assert rows[1]["amount"] == Decimal("0.01")

    def test_amount_zero(self, spark):
        data = [("$0.00",), ("0",)]
        df = spark.createDataFrame(data, ["amount"])
        result_df = df.withColumn(
            "amount",
            regexp_replace(col("amount"), "[$,]", "").cast(DecimalType(15, 2))
        )
        rows = result_df.collect()

        assert rows[0]["amount"] == Decimal("0.00")
        assert rows[1]["amount"] == Decimal("0.00")

    def test_amount_null_value(self, spark):
        data = [(None,)]
        df = spark.createDataFrame(data, ["amount"])
        result_df = df.withColumn(
            "amount",
            regexp_replace(col("amount"), "[$,]", "").cast(DecimalType(15, 2))
        )
        rows = result_df.collect()

        assert rows[0]["amount"] is None

    def test_amount_invalid_string(self, spark):
        data = [("abc",), ("$abc",)]
        df = spark.createDataFrame(data, ["amount"])
        result_df = df.withColumn(
            "amount",
            regexp_replace(col("amount"), "[$,]", "").cast(DecimalType(15, 2))
        )
        rows = result_df.collect()

        # Invalid cast should produce None
        assert rows[0]["amount"] is None
        assert rows[1]["amount"] is None

    def test_amount_negative_value(self, spark):
        data = [("-$1,500.00",), ("$-1,500.00",)]
        df = spark.createDataFrame(data, ["amount"])
        result_df = df.withColumn(
            "amount",
            regexp_replace(col("amount"), "[$,]", "").cast(DecimalType(15, 2))
        )
        rows = result_df.collect()

        # "-1500.00" after removing $ and ,
        assert rows[0]["amount"] == Decimal("-1500.00")
        # "-1500.00" after removing $ and ,
        assert rows[1]["amount"] == Decimal("-1500.00")


class TestColumnSelectionAndCasting:
    """Test Step 4: Select and cast columns to silver schema."""

    def test_column_renaming_email_address_to_email(self, spark, sample_schema):
        ts = datetime(2024, 1, 1, 10, 0, 0)
        data = [
            ("D1", "A1", "ACC1", "111-222-3333", "user@example.com", "B1", "08-01-2026 14:00", "$1,000.00", ts),
        ]
        df = spark.createDataFrame(data, schema=sample_schema)

        # Apply transformations
        transformed_df = df.withColumn(
            "deposit_date",
            to_date(col("deposit_timestamp"), "dd-MM-yyyy HH:mm")
        ).withColumn(
            "amount",
            regexp_replace(col("amount"), "[$,]", "").cast(DecimalType(15, 2))
        )

        silver_df = transformed_df.select(
            col("deposit_id").cast(StringType()).alias("deposit_id"),
            col("account_id").cast(StringType()).alias("account_id"),
            col("account_number").cast(StringType()).alias("account_number"),
            col("phone_number").cast(StringType()).alias("phone_number"),
            col("email_address").cast(StringType()).alias("email"),
            col("branch_id").cast(StringType()).alias("branch_id"),
            col("deposit_date").cast(DateType()).alias("deposit_date"),
            col("amount")
        )

        # Verify column names
        assert silver_df.columns == [
            "deposit_id", "account_id", "account_number",
            "phone_number", "email", "branch_id", "deposit_date", "amount"
        ]

        row = silver_df.collect()[0]
        assert row["email"] == "user@example.com"
        assert row["deposit_id"] == "D1"
        assert row["deposit_date"] == date(2026, 1, 8)
        assert row["amount"] == Decimal("1000.00")

    def test_full_pipeline_integration(self, spark, sample_schema):
        """End-to-end test of the transformation pipeline logic."""
        ts_old = datetime(2024, 1, 1, 10, 0, 0)
        ts_new = datetime(2024, 1, 2, 10, 0, 0)
        data = [
            ("D1", "A1", "ACC1", "111", "a@b.com", "B1", "08-01-2026 14:00", "$44,196.98", ts_old),
            ("D2", "A1", "ACC1", "111", "a@b.com", "B1", "09-01-2026 15:00", "$2,000.00", ts_new),
            ("D3", "A2", "ACC2", "222", "c@d.com", "B2", "10-01-2026 16:00", "$48604.41", ts_new),
            ("D4", "A2", "ACC2", "222", "c@d.com", "B2", "11-01-2026 17:00", "$3,000.00", ts_new),
        ]
        bronze_df = spark.createDataFrame(data, schema=sample_schema)

        # Step 1: Filter latest batch
        from pyspark.sql.functions import max as spark_max
        latest_ts = bronze_df.select(spark_max(col("ingestion_timestamp"))).collect()[0][0]
        latest_batch_df = bronze_df.filter(col("ingestion_timestamp") == latest_ts)

        # Step 2: Deduplicate
        deduplicated_df = latest_batch_df.dropDuplicates(["account_id"])

        # Step 3: Transform
        transformed_df = deduplicated_df.withColumn(
            "deposit_date",
            to_date(col("deposit_timestamp"), "dd-MM-yyyy HH:mm")
        ).withColumn(
            "amount",
            regexp_replace(col("amount"), "[$,]", "").cast(DecimalType(15, 2))
        )

        # Step 4: Select
        silver_df = transformed_df.select(
            col("deposit_id").cast(StringType()).alias("deposit_id"),
            col("account_id").cast(StringType()).alias("account_id"),
            col("account_number").cast(StringType()).alias("account_number"),
            col("phone_number").cast(StringType()).alias("phone_number"),
            col("email_address").cast(StringType()).alias("email"),
            col("branch_id").cast(StringType()).alias("branch_id"),
            col("deposit_date").cast(DateType()).alias("deposit_date"),
            col("amount")
        )

        # Should have 2 records (latest batch has 3, but A1 and A2 are deduplicated)
        assert silver_df.count() == 2

        rows = silver_df.orderBy("account_id").collect()
        assert rows[0]["account_id"] == "A1"
        assert rows[1]["account_id"] == "A2"

        # Verify amounts are properly cleaned
        for row in rows:
            assert row["amount"] is not None
            assert row["deposit_date"] is not None