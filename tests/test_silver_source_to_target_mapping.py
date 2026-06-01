# This source file is a PySpark script that executes transformations procedurally
# (no reusable functions or methods). However, we can test the core transformation
# logic by recreating the key business rules in isolated test scenarios using
# PySpark's local mode.

import pytest
from decimal import Decimal
from datetime import date, datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, regexp_replace, to_date, when, lit
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
def bronze_schema():
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
    """Test conversion of deposit_timestamp string to date."""

    def test_valid_deposit_timestamp_converts_to_date(self, spark, bronze_schema):
        data = [("D001", "A001", "ACC001", "555-0100", "a@b.com", "B001",
                 "08-01-2026 14:00", "$1000.00", datetime(2026, 1, 8, 10, 0, 0))]
        df = spark.createDataFrame(data, schema=bronze_schema)

        result_df = df.withColumn(
            "deposit_date",
            to_date(col("deposit_timestamp"), "dd-MM-yyyy HH:mm")
        )

        result = result_df.select("deposit_date").collect()[0][0]
        assert result == date(2026, 1, 8)

    def test_different_date_format_converts_correctly(self, spark, bronze_schema):
        data = [("D002", "A002", "ACC002", "555-0200", "b@c.com", "B002",
                 "25-12-2025 09:30", "$500.00", datetime(2025, 12, 25, 10, 0, 0))]
        df = spark.createDataFrame(data, schema=bronze_schema)

        result_df = df.withColumn(
            "deposit_date",
            to_date(col("deposit_timestamp"), "dd-MM-yyyy HH:mm")
        )

        result = result_df.select("deposit_date").collect()[0][0]
        assert result == date(2025, 12, 25)

    def test_null_deposit_timestamp_returns_null_date(self, spark, bronze_schema):
        data = [("D003", "A003", "ACC003", "555-0300", "c@d.com", "B003",
                 None, "$200.00", datetime(2026, 1, 1, 10, 0, 0))]
        df = spark.createDataFrame(data, schema=bronze_schema)

        result_df = df.withColumn(
            "deposit_date",
            to_date(col("deposit_timestamp"), "dd-MM-yyyy HH:mm")
        )

        result = result_df.select("deposit_date").collect()[0][0]
        assert result is None

    def test_invalid_deposit_timestamp_returns_null_date(self, spark, bronze_schema):
        data = [("D004", "A004", "ACC004", "555-0400", "d@e.com", "B004",
                 "invalid-date-string", "$300.00", datetime(2026, 1, 1, 10, 0, 0))]
        df = spark.createDataFrame(data, schema=bronze_schema)

        result_df = df.withColumn(
            "deposit_date",
            to_date(col("deposit_timestamp"), "dd-MM-yyyy HH:mm")
        )

        result = result_df.select("deposit_date").collect()[0][0]
        assert result is None

    def test_time_portion_is_stripped_only_date_remains(self, spark, bronze_schema):
        data = [("D005", "A005", "ACC005", "555-0500", "e@f.com", "B005",
                 "15-06-2026 23:59", "$100.00", datetime(2026, 6, 15, 10, 0, 0))]
        df = spark.createDataFrame(data, schema=bronze_schema)

        result_df = df.withColumn(
            "deposit_date",
            to_date(col("deposit_timestamp"), "dd-MM-yyyy HH:mm")
        )

        result = result_df.select("deposit_date").collect()[0][0]
        assert result == date(2026, 6, 15)


class TestAmountTransformation:
    """Test stripping '$' and commas from amount and casting to decimal."""

    def test_amount_with_dollar_and_comma(self, spark):
        df = spark.createDataFrame([("$44,196.98",)], ["amount"])

        result_df = df.withColumn(
            "amount",
            regexp_replace(col("amount"), "[$,]", "").cast(DecimalType(15, 2))
        )

        result = result_df.select("amount").collect()[0][0]
        assert result == Decimal("44196.98")

    def test_amount_with_dollar_no_comma(self, spark):
        df = spark.createDataFrame([("$48604.41",)], ["amount"])

        result_df = df.withColumn(
            "amount",
            regexp_replace(col("amount"), "[$,]", "").cast(DecimalType(15, 2))
        )

        result = result_df.select("amount").collect()[0][0]
        assert result == Decimal("48604.41")

    def test_amount_with_multiple_commas(self, spark):
        df = spark.createDataFrame([("$1,234,567.89",)], ["amount"])

        result_df = df.withColumn(
            "amount",
            regexp_replace(col("amount"), "[$,]", "").cast(DecimalType(15, 2))
        )

        result = result_df.select("amount").collect()[0][0]
        assert result == Decimal("1234567.89")

    def test_amount_zero(self, spark):
        df = spark.createDataFrame([("$0.00",)], ["amount"])

        result_df = df.withColumn(
            "amount",
            regexp_replace(col("amount"), "[$,]", "").cast(DecimalType(15, 2))
        )

        result = result_df.select("amount").collect()[0][0]
        assert result == Decimal("0.00")

    def test_amount_null_returns_null(self, spark):
        df = spark.createDataFrame([(None,)], ["amount"])

        result_df = df.withColumn(
            "amount",
            regexp_replace(col("amount"), "[$,]", "").cast(DecimalType(15, 2))
        )

        result = result_df.select("amount").collect()[0][0]
        assert result is None

    def test_amount_without_dollar_sign(self, spark):
        """Amount that has no dollar sign but has commas."""
        df = spark.createDataFrame([("1,000.50",)], ["amount"])

        result_df = df.withColumn(
            "amount",
            regexp_replace(col("amount"), "[$,]", "").cast(DecimalType(15, 2))
        )

        result = result_df.select("amount").collect()[0][0]
        assert result == Decimal("1000.50")

    def test_amount_non_numeric_returns_null(self, spark):
        """Non-numeric string after stripping should cast to null."""
        df = spark.createDataFrame([("$abc",)], ["amount"])

        result_df = df.withColumn(
            "amount",
            regexp_replace(col("amount"), "[$,]", "").cast(DecimalType(15, 2))
        )

        result = result_df.select("amount").collect()[0][0]
        assert result is None


class TestDeduplication:
    """Test deduplication logic on account_id."""

    def test_duplicate_account_ids_are_removed(self, spark, bronze_schema):
        data = [
            ("D001", "A001", "ACC001", "555-0100", "a@b.com", "B001",
             "08-01-2026 14:00", "$1000.00", datetime(2026, 1, 8, 10, 0, 0)),
            ("D002", "A001", "ACC001", "555-0100", "a@b.com", "B001",
             "09-01-2026 14:00", "$2000.00", datetime(2026, 1, 8, 10, 0, 0)),
        ]
        df = spark.createDataFrame(data, schema=bronze_schema)

        deduplicated_df = df.dropDuplicates(["account_id"])

        assert deduplicated_df.count() == 1

    def test_unique_account_ids_all_retained(self, spark, bronze_schema):
        data = [
            ("D001", "A001", "ACC001", "555-0100", "a@b.com", "B001",
             "08-01-2026 14:00", "$1000.00", datetime(2026, 1, 8, 10, 0, 0)),
            ("D002", "A002", "ACC002", "555-0200", "b@c.com", "B002",
             "09-01-2026 14:00", "$2000.00", datetime(2026, 1, 8, 10, 0, 0)),
            ("D003", "A003", "ACC003", "555-0300", "c@d.com", "B003",
             "10-01-2026 14:00", "$3000.00", datetime(2026, 1, 8, 10, 0, 0)),
        ]
        df = spark.createDataFrame(data, schema=bronze_schema)

        deduplicated_df = df.dropDuplicates(["account_id"])

        assert deduplicated_df.count() == 3

    def test_empty_dataframe_deduplication(self, spark, bronze_schema):
        df = spark.createDataFrame([], schema=bronze_schema)

        deduplicated_df = df.dropDuplicates(["account_id"])

        assert deduplicated_df.count() == 0


class TestLatestIngestionFilter:
    """Test filtering by latest ingestion_timestamp."""

    def test_only_latest_batch_records_returned(self, spark, bronze_schema):
        ts_old = datetime(2026, 1, 7, 10, 0, 0)
        ts_new = datetime(2026, 1, 8, 10, 0, 0)

        data = [
            ("D001", "A001", "ACC001", "555-0100", "a@b.com", "B001",
             "07-01-2026 14:00", "$1000.00", ts_old),
            ("D002", "A002", "ACC002", "555-0200", "b@c.com", "B002",
             "08-01-2026 14:00", "$2000.00", ts_new),
            ("D003", "A003", "ACC003", "555-0300", "c@d.com", "B003",
             "08-01-2026 15:00", "$3000.00", ts_new),
        ]
        df = spark.createDataFrame(data, schema=bronze_schema)

        from pyspark.sql.functions import max as spark_max
        latest_ts = df.select(spark_max(col("ingestion_timestamp"))).collect()[0][0]
        latest_batch_df = df.filter(col("ingestion_timestamp") == latest_ts)

        assert latest_batch_df.count() == 2
        # Verify all returned records have the latest timestamp
        timestamps = [row["ingestion_timestamp"] for row in latest_batch_df.collect()]
        assert all(ts == ts_new for ts in timestamps)

    def test_single_batch_returns_all_records(self, spark, bronze_schema):
        ts = datetime(2026, 1, 8, 10, 0, 0)

        data = [
            ("D001", "A001", "ACC001", "555-0100", "a@b.com", "B001",
             "08-01-2026 14:00", "$1000.00", ts),
            ("D002", "A002", "ACC002", "555-0200", "b@c.com", "B002",
             "08-01-2026 15:00", "$2000.00", ts),
        ]
        df = spark.createDataFrame(data, schema=bronze_schema)

        from pyspark.sql.functions import max as spark_max
        latest_ts = df.select(spark_max(col("ingestion_timestamp"))).collect()[0][0]
        latest_batch_df = df.filter(col("ingestion_timestamp") == latest_ts)

        assert latest_batch_df.count() == 2


class TestSilverSchemaSelection:
    """Test the final column selection and renaming for silver schema."""

    def test_email_address_renamed_to_email(self, spark, bronze_schema):
        data = [("D001", "A001", "ACC001", "555-0100", "user@example.com", "B001",
                 "08-01-2026 14:00", "$1000.00", datetime(2026, 1, 8, 10, 0, 0))]
        df = spark.createDataFrame(data, schema=bronze_schema)

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
        assert "email" in silver_df.columns
        assert "email_address" not in silver_df.columns

        # Verify email value
        result = silver_df.select("email").collect()[0][0]
        assert result == "user@example.com"

    def test_silver_schema_has_correct_columns(self, spark, bronze_schema):
        data = [("D001", "A001", "ACC001", "555-0100", "user@example.com", "B001",
                 "08-01-2026 14:00", "$1000.00", datetime(2026, 1, 8, 10, 0, 0))]
        df = spark.createDataFrame(data, schema=bronze_schema)

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

        expected_columns = [
            "deposit_id", "account_id", "account_number",
            "phone_number", "email", "branch_id", "deposit_date", "amount"
        ]
        assert silver_df.columns == expected_columns

    def test_ingestion_timestamp_not_in_silver_output(self, spark, bronze_schema):
        data = [("D001", "A001", "ACC001", "555-0100", "user@example.com", "B001",
                 "08-01-2026 14:00", "$1000.00", datetime(2026, 1, 8, 10, 0, 0))]
        df = spark.createDataFrame(data, schema=bronze_schema)

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

        assert "ingestion_timestamp" not in silver_df.columns
        assert "deposit_timestamp" not in silver_df.columns


class TestEndToEndTransformation:
    """Test the full transformation pipeline logic end-to-end."""

    def test_full_pipeline_produces_correct_output(self, spark, bronze_schema):
        ts = datetime(2026, 1, 8, 10, 0, 0)
        data = [
            ("D001", "A001", "ACC001", "555-0100", "alice@bank.com", "B001",
             "08-01-2026 14:00", "$44,196.98", ts),
            ("D002", "A002", "ACC002", "555-0200", "bob@bank.com", "B002",
             "09-01-2026 09:30", "$48604.41", ts),
        ]
        df = spark.createDataFrame(data, schema=bronze_schema)

        # Simulate full pipeline
        from pyspark.sql.functions import max as spark_max
        latest_ts = df.select(spark_max(col("ingestion_timestamp"))).collect()[0][0]
        latest_batch_df = df.filter(col("ingestion_timestamp") == latest_ts)
        deduplicated_df = latest_batch_df.dropDuplicates(["account_id"])

        transformed_df = deduplicated_df.withColumn(
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

        assert silver_df.count() == 2

        rows = {row["deposit_id"]: row for row in silver_df.collect()}

        # Verify D001
        assert rows["D001"]["account_id"] == "A001"
        assert rows["D001"]["email"] == "alice@bank.com"
        assert rows["D001"]["deposit_date"] == date(2026, 1, 8)
        assert rows["D001"]["amount"] == Decimal("44196.98")

        # Verify D002
        assert rows["D002"]["account_id"] == "A002"
        assert rows["D002"]["email"] == "bob@bank.com"
        assert rows["D002"]["deposit_date"] == date(2026, 1, 9)
        assert rows["D002"]["amount"] == Decimal("48604.41")

    def test_pipeline_with_duplicates_and_multiple_batches(self, spark, bronze_schema):
        ts_old = datetime(2026, 1, 7, 10, 0, 0)
        ts_new = datetime(2026, 1, 8, 10, 0, 0)

        data = [
            # Old batch - should be filtered out
            ("D001", "A001", "ACC001", "555-0100", "old@bank.com", "B001",
             "07-01-2026 14:00", "$500.00", ts_old),
            # New batch - duplicate account_id A001
            ("D002", "A001", "ACC001", "555-0100", "new@bank.com", "B001",
             "08-01-2026 14:00", "$1000.00", ts_new),
            ("D003", "A001", "ACC001", "555-0100", "new2@bank.com", "B001",
             "08-01-2026 15:00", "$1500.00", ts_new),
            # New batch - unique account
            ("D004", "A002", "ACC002", "555-0200", "bob@bank.com", "B002",
             "08-01-2026 16:00", "$2,000.00", ts_new),
        ]
        df = spark.createDataFrame(data, schema=bronze_schema)

        from pyspark.sql.functions import max as spark_max
        latest_ts = df.select(spark_max(col("ingestion_timestamp"))).collect()[0][0]
        latest_batch_df = df.filter(col("ingestion_timestamp") == latest_ts)

        # Should have 3 records from latest batch
        assert latest_batch_df.count() == 3

        deduplicated_df = latest_batch_df.dropDuplicates(["account_id"])

        # After dedup on account_id, should have 2 unique accounts
        assert deduplicated_df.count() == 2

        transformed_df = deduplicated_df.withColumn(
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

        assert silver_df.count() == 2

        # Verify A002 record
        a002_row = silver_df.filter(col("account_id") == "A002").collect()[0]
        assert a002_row["amount"] == Decimal("2000.00")
        assert a002_row["deposit_date"] == date(2026, 1, 8)