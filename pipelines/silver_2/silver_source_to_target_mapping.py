```python
# silver_deposits.py
# Silver layer transformation script for deposits data
# Reads from bronze table, applies business transformations, and writes to silver table

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, trim, regexp_replace, to_date, cast, current_timestamp,
    max as spark_max, when, lit
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
# Step 2: Apply deduplication on account_id (customer_id equivalent) before transformation
# Keep the first occurrence based on deposit_id ordering
# ---------------------------------------------------------------------------
deduplicated_df = latest_batch_df.dropDuplicates(["account_id"])

# ---------------------------------------------------------------------------
# Step 3: Apply business transformation rules
# ---------------------------------------------------------------------------

# Transformation (a): Convert deposit_timestamp string to date format
# The source format is "dd-MM-yyyy HH:mm" (e.g., "08-01-2026 14:00")
# Extract only the date part and populate into deposit_date column
transformed_df = deduplicated_df.withColumn(
    "deposit_date",
    to_date(col("deposit_timestamp"), "dd-MM-yyyy HH:mm")
)

# Transformation (b): Trim '$' and commas from amount field, then cast to decimal
# The amount field contains values like "$44,196.98" or "$48604.41"
transformed_df = transformed_df.withColumn(
    "amount",
    regexp_replace(col("amount"), "[$,]", "").cast(DecimalType(15, 2))
)

# ---------------------------------------------------------------------------
# Step 4: Select and cast columns to match the target silver schema
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Step 5: Write transformed dataset to silver table partitioned by deposit_date
# ---------------------------------------------------------------------------
silver_df.write \
    .format("delta") \
    .mode("overwrite") \
    .partitionBy("deposit_date") \
    .saveAsTable("finance.deposits_silver")

# Log completion
print("Silver layer transformation completed successfully.")
print(f"Total records written to finance.deposits_silver: {silver_df.count()}")

# Stop SparkSession
spark.stop()
```