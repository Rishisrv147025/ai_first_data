```python
# silver_deposits.py
# Silver layer transformation script for deposits data
# Reads from bronze table, applies business transformations, and writes to silver table

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, trim, regexp_replace, to_date, cast, current_timestamp,
    max as spark_max, lit
)
from pyspark.sql.types import DecimalType, DateType, StringType

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("Silver_Deposits_Transformation") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

# ---------------------------------------------------------------------------
# Step 1: Read all records from bronze table with the latest ingestion timestamp
# ---------------------------------------------------------------------------
bronze_df = spark.table("finance.deposits_raw")

# Identify the latest ingestion_timestamp to filter only the most recent batch
latest_ingestion_ts = bronze_df.select(spark_max(col("ingestion_timestamp"))).collect()[0][0]

# Filter records that belong to the latest ingestion batch
latest_batch_df = bronze_df.filter(col("ingestion_timestamp") == latest_ingestion_ts)

# ---------------------------------------------------------------------------
# Step 2: Apply data validation - filter out invalid records
# Records failing validation will be routed to error table
# ---------------------------------------------------------------------------

# Define valid email regex pattern
email_pattern = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"

# Define valid phone_number pattern (only numeric and hyphen allowed, also '+' and spaces/Ext for extensions)
# Based on the data, phone_number can contain digits, hyphens, plus sign, spaces, and "Ext" prefix
# The requirement states: "only numeric and 'hyphen'" - strict interpretation
phone_pattern = r"^[0-9\-\+\s\w]+$"

# Validation rule: phone_number has only numeric and 'hyphen'
# Strict interpretation: only digits and hyphens
phone_strict_pattern = r"^[0-9\-]+$"

# Apply validation rules
valid_records_df = latest_batch_df.filter(
    # Rule 1: deposit_id is not null
    (col("deposit_id").isNotNull()) & (trim(col("deposit_id")) != "") & (trim(col("deposit_id")) != "NaN") &
    # Rule 2: account_id is not null
    (col("account_id").isNotNull()) & (trim(col("account_id")) != "") & (trim(col("account_id")) != "NaN") &
    # Rule 3: account_number is not null
    (col("account_number").isNotNull()) & (trim(col("account_number")) != "") & (trim(col("account_number")) != "NaN") &
    # Rule 4: phone_number is not null
    (col("phone_number").isNotNull()) & (trim(col("phone_number")) != "") & (trim(col("phone_number")) != "NaN") &
    # Rule 5: phone_number has only numeric and hyphen
    (col("phone_number").rlike(r"^[0-9\-]+$")) &
    # Rule 6: email_address is a valid email
    (col("email_address").rlike(email_pattern)) &
    # Rule 7: branch_id is not null
    (col("branch_id").isNotNull()) & (trim(col("branch_id")) != "") & (trim(col("branch_id")) != "NaN")
)

# Identify invalid records for error table
invalid_records_df = latest_batch_df.subtract(valid_records_df)

# Write invalid records to error table
invalid_records_df.write \
    .mode("append") \
    .saveAsTable("finance.deposits_raw_errors")

# Write valid records to validated deposits table
valid_records_df.write \
    .mode("overwrite") \
    .saveAsTable("finance.deposits")

# ---------------------------------------------------------------------------
# Step 3: Apply business transformation rules on valid records
# ---------------------------------------------------------------------------

# Transformation a: Convert deposit_timestamp (string) to date format
# deposit_timestamp format from source: "dd-MM-yyyy HH:mm"
transformed_df = valid_records_df.withColumn(
    "deposit_date",
    to_date(col("deposit_timestamp"), "dd-MM-yyyy HH:mm")
)

# Transformation b: Trim '$' and commas from amount field, convert to decimal
# Amount field contains values like "$44,196.98", "$48604.41", "$7347.67"
transformed_df = transformed_df.withColumn(
    "amount",
    regexp_replace(col("amount"), r"[\$,]", "").cast(DecimalType(18, 2))
)

# ---------------------------------------------------------------------------
# Step 4: Select only the required columns for silver schema
# ---------------------------------------------------------------------------
silver_df = transformed_df.select(
    col("deposit_id").cast(StringType()).alias("Deposit_id"),
    col("account_id").cast(StringType()).alias("Account_id"),
    col("account_number").cast(StringType()).alias("Account_number"),
    col("phone_number").cast(StringType()).alias("Phone_number"),
    col("email_address").cast(StringType()).alias("Email"),
    col("branch_id").cast(StringType()).alias("Branch_id"),
    col("deposit_date").cast(DateType()).alias("Deposit_date"),
    col("amount").alias("Amount")
)

# ---------------------------------------------------------------------------
# Step 5: Apply deduplication on account_id before writing to Delta
# Keep the first occurrence (or latest based on deposit_date) per account_id
# ---------------------------------------------------------------------------
deduplicated_df = silver_df.dropDuplicates(["Account_id"])

# ---------------------------------------------------------------------------
# Step 6: Write transformed and deduplicated data to silver table
# Partition by deposit_date for optimized querying
# ---------------------------------------------------------------------------
deduplicated_df.write \
    .format("delta") \
    .mode("overwrite") \
    .partitionBy("Deposit_date") \
    .saveAsTable("finance.deposits_silver")

# Log completion
print("Silver layer transformation completed successfully.")
print(f"Total records processed from bronze: {latest_batch_df.count()}")
print(f"Valid records after validation: {valid_records_df.count()}")
print(f"Invalid records routed to error table: {invalid_records_df.count()}")
print(f"Records written to silver after deduplication: {deduplicated_df.count()}")

# Stop SparkSession
spark.stop()
```