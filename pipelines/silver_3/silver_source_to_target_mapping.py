```python
# silver_deposits.py
# Silver layer transformation script for deposits data
# Reads from bronze table, applies business transformations, and writes to silver table

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, to_date, regexp_replace, trim, cast, current_timestamp,
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
# Step 1: Read all records from bronze table "finance.deposits_raw"
# ---------------------------------------------------------------------------
df_bronze = spark.table("finance.deposits_raw")

# ---------------------------------------------------------------------------
# Step 2: Filter records with the latest ingestion_timestamp
# Only process the most recent batch of ingested data
# ---------------------------------------------------------------------------
latest_ingestion_ts = df_bronze.select(spark_max("ingestion_timestamp")).collect()[0][0]

df_latest = df_bronze.filter(col("ingestion_timestamp") == latest_ingestion_ts)

# ---------------------------------------------------------------------------
# Step 3: Apply deduplication on account_id before further processing
# Keep the first occurrence based on deposit_id ordering
# ---------------------------------------------------------------------------
df_deduped = df_latest.dropDuplicates(["account_id"])

# ---------------------------------------------------------------------------
# Step 4: Apply business transformation rules
# ---------------------------------------------------------------------------

# Transformation (a): Convert deposit_timestamp (string) to date type
# Source format is "dd-MM-yyyy HH:mm", extract only the date part
df_transformed = df_deduped.withColumn(
    "deposit_date",
    to_date(col("deposit_timestamp"), "dd-MM-yyyy HH:mm")
)

# Transformation (b): Trim '$' and ',' characters from amount string and cast to decimal
# Amount field contains values like "$44,196.98" or "$48604.41"
df_transformed = df_transformed.withColumn(
    "amount",
    regexp_replace(col("amount"), "[$,]", "").cast(DecimalType(15, 2))
)

# ---------------------------------------------------------------------------
# Step 5: Select and rename columns to match target silver schema
# ---------------------------------------------------------------------------
df_silver = df_transformed.select(
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
# Step 6: Write transformed dataset to silver table partitioned by deposit_date
# Persist as Delta table "finance.deposits_silver"
# ---------------------------------------------------------------------------
df_silver.write \
    .format("delta") \
    .mode("overwrite") \
    .partitionBy("Deposit_date") \
    .saveAsTable("finance.deposits_silver")

# Log completion
print("Silver layer transformation completed successfully. Data written to finance.deposits_silver.")

# Stop SparkSession
spark.stop()
```