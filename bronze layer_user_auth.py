"""
bronze_deposits.py
Bronze Layer — Reads deposits.csv from ADLS and ingests raw data into finance.deposits_raw.
No transformations applied. Data persists as-is from source.
"""

import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp

# ---------------------------------------------------------------------------
# Runtime parameter: monthly folder name, e.g. "jan_2026", "feb_2026"
# ---------------------------------------------------------------------------
mon_yyyy = sys.argv[1]

# ---------------------------------------------------------------------------
# Initialise SparkSession
# ---------------------------------------------------------------------------
spark = SparkSession.builder \
    .appName("BronzeDepositsIngestion") \
    .getOrCreate()

# ---------------------------------------------------------------------------
# Step 1: Construct ADLS source path from runtime parameter
# Source file is placed monthly under folder named mon_yyyy (e.g. jan_2026)
# ---------------------------------------------------------------------------
source_path = (
    f"abfss://dev@depositaccount.dfs.core.windows.net"
    f"/deposits/monthly/{mon_yyyy}/deposits.csv"
)

# ---------------------------------------------------------------------------
# Step 2: Read CSV from ADLS cloud location
# All columns are ingested as-is — no type casting, no transformations
# inferSchema is set to false to preserve all values as strings (no implicit casting)
# Source columns (as expected by downstream layers):
#   deposit_id, account_id, account_number, phone_number,
#   email_address, branch_id, deposit_timestamp, amount
# ---------------------------------------------------------------------------
raw_df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "false") \
    .csv(source_path)

# ---------------------------------------------------------------------------
# Step 3: Add ingestion_timestamp system column
# Records when the data was loaded into the Bronze table
# This is the ONLY withColumn allowed at Bronze layer
# ---------------------------------------------------------------------------
raw_df = raw_df.withColumn("ingestion_timestamp", current_timestamp())

# ---------------------------------------------------------------------------
# Step 4: Capture record count before write for logging purposes
# Avoids triggering a second full scan after the write operation
# ---------------------------------------------------------------------------
record_count = raw_df.count()

# ---------------------------------------------------------------------------
# Step 5: Append raw records into finance.deposits_raw
# Mode is "append" — every monthly load accumulates in the Bronze table
# No partitioning, no deduplication, no filtering — raw landing zone
# ---------------------------------------------------------------------------
raw_df.write \
    .mode("append") \
    .saveAsTable("finance.deposits_raw")

print(f"Bronze ingestion complete. Records loaded: {record_count}")

spark.stop()