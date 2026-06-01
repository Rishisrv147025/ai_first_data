"""
silver_deposits.py
Silver Layer — Reads from finance.deposits (clean validated records, latest batch),
applies business transformations, and appends to finance.deposits_silver.

Pipeline order: Bronze → Data Validation → Silver (this script) → Gold
Source: finance.deposits  (validated staging table from raw_data_validation.py)
Target: finance.deposits_silver
Write mode: append — accumulates monthly loads to preserve history

Notes on transformations:
- The source-to-target mapping document specifies: customer_id → CUSTOMER_ID (cast to integer)
- The Silver layer template mandates two additional standard transformations:
    a. deposit_timestamp (string) → deposit_date (date) via to_date()
    b. amount (string with '$') → amount (decimal) via regexp_replace and cast
  These are required by the Silver layer business rules even though the provided
  mapping document only lists the customer_id transformation explicitly.
- CUSTOMER_ID is included in the select list per the mapping document requirement,
  extending the base 8-column schema with this additional mapped column.
"""

import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, max as spark_max,
    regexp_replace, to_date
)
from pyspark.sql.types import DecimalType, StringType, DateType, IntegerType

# ---------------------------------------------------------------------------
# Initialise SparkSession
# ---------------------------------------------------------------------------
spark = SparkSession.builder \
    .appName("SilverDepositsTransformation") \
    .getOrCreate()

# ---------------------------------------------------------------------------
# Step 1: Read clean validated records from finance.deposits
# This table contains only records that passed all 7 Data Validation rules.
# Silver must read from finance.deposits — NOT from finance.deposits_raw.
# Pipeline order enforced: Validation happens before Transformation.
# ---------------------------------------------------------------------------
validated_df = spark.table("finance.deposits")

# ---------------------------------------------------------------------------
# Step 2: Filter to the latest ingestion batch only
# Ensures only the most recently loaded month's records are processed.
# ingestion_timestamp was added at Bronze layer and carried through validation.
# ---------------------------------------------------------------------------
latest_ts = validated_df.select(spark_max("ingestion_timestamp")).first()[0]

if latest_ts is None:
    print("WARNING: No records found in finance.deposits. Exiting.")
    spark.stop()
    sys.exit(0)

latest_df = validated_df.filter(col("ingestion_timestamp") == latest_ts)

# ---------------------------------------------------------------------------
# Step 3: Apply business transformation (a) — mandated by Silver layer rules
# Source column name: deposit_timestamp (string type in finance.deposits)
# The source column is ALWAYS "deposit_timestamp" — never "deposit_date",
# "Deposit_date", "Deposit_timestamp" or any other variation.
# Convert deposit_timestamp string into date type → new column "deposit_date"
# Note: This transformation is required by the Silver layer business rules
# even though the provided mapping document does not list it explicitly.
# ---------------------------------------------------------------------------
transformed_df = latest_df.withColumn(
    "deposit_date",
    to_date(col("deposit_timestamp"), "yyyy-MM-dd HH:mm:ss")
)

# ---------------------------------------------------------------------------
# Step 4: Apply business transformation (b) — mandated by Silver layer rules
# Trim '$' character from amount field of type string.
# Remove commas, convert to DecimalType(18,2) → populate "amount" column.
# Note: This transformation is required by the Silver layer business rules
# even though the provided mapping document does not list it explicitly.
# ---------------------------------------------------------------------------
transformed_df = transformed_df.withColumn(
    "amount",
    regexp_replace(col("amount"), r"[$,]", "").cast(DecimalType(18, 2))
)

# ---------------------------------------------------------------------------
# Step 5: Apply mapping document transformation
# Cast column 'customer_id' to IntegerType() and rename to 'CUSTOMER_ID'
# as specified in the source-to-target mapping document.
# This extends the base 8-column Silver schema per the mapping requirement.
# ---------------------------------------------------------------------------
transformed_df = transformed_df.withColumn(
    "customer_id",
    col("customer_id").cast(IntegerType())
).withColumnRenamed("customer_id", "CUSTOMER_ID")

# ---------------------------------------------------------------------------
# Step 6: Select and enforce target schema
# Base Silver schema (8 columns): deposit_id, account_id, account_number,
#   phone_number, email, branch_id, deposit_date, amount
# Additional column per mapping document: CUSTOMER_ID (cast to integer)
# Note: ingestion_timestamp is NOT included in the target schema — it is
# only used for batch filtering in Step 2 and not carried to Silver output.
# ---------------------------------------------------------------------------
silver_df = transformed_df.select(
    col("deposit_id").cast(StringType()),
    col("account_id").cast(StringType()),
    col("account_number").cast(StringType()),
    col("phone_number").cast(StringType()),
    col("email_address").cast(StringType()).alias("email"),
    col("branch_id").cast(StringType()),
    col("deposit_date").cast(DateType()),
    col("amount"),
    col("CUSTOMER_ID")
)

# ---------------------------------------------------------------------------
# Step 7: Capture record count before writing to avoid redundant Spark action
# Calling count() after write would recompute the DataFrame inefficiently.
# ---------------------------------------------------------------------------
record_count = silver_df.count()

# ---------------------------------------------------------------------------
# Step 8: Append transformed data to finance.deposits_silver
# Mode is "append" — Silver accumulates month-on-month, matching Bronze.
# Overwrite is FORBIDDEN — it would erase all previous months' history.
# Partitioned by deposit_date for query performance.
# ---------------------------------------------------------------------------
silver_df.write \
    .mode("append") \
    .partitionBy("deposit_date") \
    .saveAsTable("finance.deposits_silver")

print(f"Silver transformation complete. Records appended: {record_count}")

spark.stop()