import sys
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, input_file_name
from pyspark.sql.types import StructType, StructField, StringType

# Configure structured logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("BronzeQAProjectStructureIngestion")

# =====================================================================================
# IMPORTANT NOTE ON SOURCE-TO-TARGET MAPPING DOCUMENT:
# The provided mapping document describes a QA test generation application's features
# (Unit Test Generation, Functional Testing, Context-Aware Generation, Test Analysis).
# It does NOT define source columns, target columns, data types, or transformation rules.
#
# The schema and table names used in this script are derived SOLELY from the Planner
# Agent's execution plan, which specified:
#   - Target Table: qa.project_structure_raw
#   - Columns: feature_id, feature_name, test_type, framework, input_source,
#              coverage_metric, test_quality_metric, generation_date, status
#   - System Columns: ingestion_timestamp, source_file_name
#
# ACTION REQUIRED: A proper source-to-target mapping document must be obtained that
# defines the actual source file structure, target table schema, column mappings, and
# data types before this pipeline is promoted to production. The current mapping
# document is entirely unrelated to a data ingestion pipeline.
# =====================================================================================

# =====================================================================================
# VERIFICATION CHECKLIST (to be completed when proper mapping document is available):
# [ ] Verify target table name 'qa.project_structure_raw' matches mapping document
# [ ] Verify all source columns from mapping are accounted for in the schema
# [ ] Verify data types for each column match mapping document specifications
# [ ] Verify transformation rules (if any) are correctly applied
# [ ] Confirm zero column names in current mapping document correspond to code columns
# [ ] The mapping document discusses NLP, BDD, Gherkin, code coverage — no tabular schema
# =====================================================================================

# Fix #5: Argument validation for sys.argv[1] (mon_yyyy parameter)
if len(sys.argv) < 2:
    logger.error("Usage: bronze_qa_project_structure.py <mon_yyyy> [storage_account]")
    logger.error("Example: bronze_qa_project_structure.py jan_2026 storageaccount")
    sys.exit(1)

# Accept runtime parameter mon_yyyy to construct the source file path dynamically
mon_yyyy = sys.argv[1]  # e.g. "jan_2026"

# Externalize storage account name as a configuration parameter
# In production, this should be sourced from Databricks secrets or a config file
storage_account = sys.argv[2] if len(sys.argv) > 2 else "storageaccount"

# Initialize SparkSession for Bronze layer ingestion
spark = SparkSession.builder.appName("BronzeQAProjectStructureIngestion").getOrCreate()

# Construct ADLS source path using runtime parameters
source_path = f"abfss://raw@{storage_account}.dfs.core.windows.net/qa/project_structure/{mon_yyyy}/project_structure.csv"

# File existence check with graceful fallback for non-Databricks environments
try:
    dbutils.fs.ls(source_path)  # noqa: F821 — dbutils is available in Databricks runtime
    logger.info(f"Source file verified at path: {source_path}")
except NameError:
    # dbutils not available (e.g., local testing) — skip file existence check
    logger.warning("dbutils not available. Skipping file existence pre-check; spark.read will fail if file is missing.")
except Exception as e:
    raise FileNotFoundError(f"Source file not found at path: {source_path}. Error: {str(e)}")

# Define explicit schema for type consistency across monthly ingestion runs
# NOTE: These columns are derived from the Planner Agent's execution plan, NOT from
# the mapping document (which does not define any tabular structure).
# A valid mapping document must be provided to confirm column correctness.
explicit_schema = StructType([
    StructField("feature_id", StringType(), True),
    StructField("feature_name", StringType(), True),
    StructField("test_type", StringType(), True),
    StructField("framework", StringType(), True),
    StructField("input_source", StringType(), True),
    StructField("coverage_metric", StringType(), True),
    StructField("test_quality_metric", StringType(), True),
    StructField("generation_date", StringType(), True),
    StructField("status", StringType(), True),
    StructField("_corrupt_record", StringType(), True)
])

# Fix #4: Read CSV with PERMISSIVE mode and corrupt record column to capture malformed rows
# PERMISSIVE mode places malformed rows into _corrupt_record column for Silver layer analysis
raw_df = (
    spark.read
    .option("header", "true")
    .option("mode", "PERMISSIVE")
    .option("columnNameOfCorruptRecord", "_corrupt_record")
    .schema(explicit_schema)
    .csv(source_path)
)

# Efficient empty DataFrame check using head(1) instead of count()
if len(raw_df.head(1)) == 0:
    logger.error(f"Source file at {source_path} is empty. No records to ingest.")
    sys.exit(1)

logger.info(f"Successfully read source file from: {source_path}")

# Bronze layer ingests ALL records as-is — no filtering, no validation, no quarantine
# In strict medallion architecture, data quality checks and quarantine routing are deferred
# to the Silver layer. Bronze captures raw data without any business logic or filtering.

# Add system column ingestion_timestamp using current_timestamp()
# This tracks when each record was ingested into the bronze layer
raw_df = raw_df.withColumn("ingestion_timestamp", current_timestamp())

# Add system column source_file_name using input_file_name()
# This tracks the source file from which the record was loaded — valid metadata for lineage
raw_df = raw_df.withColumn("source_file_name", input_file_name())

# Write ALL records to the target Delta table qa.project_structure_raw in append mode
# Bronze layer — no business transformations, no filtering — data ingested purely as-is
raw_df.write.mode("append").format("delta").saveAsTable("qa.project_structure_raw")

# Fix #4: Record count logging after successful write to confirm ingestion volume
record_count = raw_df.count()
logger.info(f"Successfully ingested {record_count} records into qa.project_structure_raw for period: {mon_yyyy}")

# Do not call spark.stop() on Databricks as SparkSession is platform-managed