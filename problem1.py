#!/usr/bin/env python3
"""
Problem 1: Log level distribution
"""

import logging
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructType, StructField
from pyspark.sql.functions import (
    regexp_extract, col, count
)
import pandas as pd
import re
import sys

# configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the log level to INFO
    # Define log message format
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)



def create_spark_session(master_url):
    """Create a Spark session optimized for cluster execution."""

    spark = (
        SparkSession.builder
        .appName("Problem1_DailySummaries_Cluster")

        # Cluster Configuration
        .master(master_url)  # Connect to Spark cluster

        # Memory Configuration
        .config("spark.executor.memory", "4g")
        .config("spark.driver.memory", "4g")
        .config("spark.driver.maxResultSize", "2g")

        # Executor Configuration
        .config("spark.executor.cores", "2")
        .config("spark.cores.max", "6")  # Use all available cores across cluster

        # S3 Configuration - Use S3A for AWS S3 access
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.InstanceProfileCredentialsProvider")

        # Performance settings for cluster execution
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")

        # Serialization
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

        # Add configuration to ignor metadata for batch job
        .config("spark.sql.streaming.ignoreMetadata", "true")

        # Arrow optimization for Pandas conversion
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")

        # Hide progress to aid in printed output
        .config("spark.ui.showConsoleProgress", "false")

        .getOrCreate()
    )

    logger.info("Spark session created successfully for cluster execution")
    return spark


# created with help of LLM
def group_log_entries(lines_iterator):
    """
    This function serves as an iterator that identifies log entries, implemented
    to handle log entries that span multiple lines to prevent null values in dataset
    and capture complete log entries
    """
    current_entry = []
    # regex pattern identifies new log entry
    start_pattern = re.compile(r"^\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}") 

    for line in lines_iterator:
        if start_pattern.match(line) and current_entry:
            yield "".join(current_entry)
            current_entry = [line]
        else:
            current_entry.append(line)
    if current_entry:
        yield "".join(current_entry)


# create spark session on cluster
logger.info("Creating spark session on cluster...")
master_url = sys.argv[1]
spark = create_spark_session(master_url)

# read all files into RDD from s3 bucket
logger.info("Loading data from s3 bucket into RDD...")
log_rdd = spark.sparkContext.textFile("s3a://ima35-assignment-spark-cluster-logs/data/*/*.log")
logger.info("Successfully loaded data from s3 bucket into RDD")

# get RDD with individual log entries (consider log entries that span multiple lines)
logger.info("Grouping RDD...")
grouped_rdd = log_rdd.mapPartitions(group_log_entries)
logger.info("RDD grouped successfully")

# create dataframe from RDD
logger.info("Crearing dataframe from RDD...")
logs_df = grouped_rdd.map(lambda x: (x,)).toDF(["value"])
logger.info("Successfully created dataframe")

# parse logs dataframe
logger.info("Parsing logs dataframe...")
logs_parsed = logs_df.select(
    regexp_extract('value', r'^(\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})', 1).alias('timestamp'),
    regexp_extract('value', r'(INFO|WARN|ERROR|DEBUG)', 1).alias('log_level'),
    regexp_extract('value', r'(INFO|WARN|ERROR|DEBUG)\s+([^:]+):', 2).alias('component'),
    col('value').alias('message')
).filter(col('log_level') != "")
logger.info("Dataframe parsed successfully")

# perform analysis
logger.info("Performing analysis...")

# get and save level counts
logger.info("Getting level counts...")
level_counts = (logs_parsed
    .groupby("log_level")
    .agg(count("*").alias("count"))
)
level_counts.toPandas().to_csv("problem1_counts.csv", index=False)
logger.info("Level count analysis complete")

# get sample
logger.info("Getting sample...")
sampled_df = logs_parsed.sample(fraction=0.1, withReplacement=False).limit(10)
sampled_df = sampled_df.select(
    col("message").alias("log_entry"),
    "log_level"
)
sampled_df.toPandas().to_csv("problem1_sample.csv", index=False)
logger.info("Sample analysis complete")

# get summary stats
logger.info("Getting summary statistics...")

total_lines = log_rdd.count()
level_lines = logs_parsed.count()
unique_levels_df = logs_parsed.select("log_level").distinct()
unique_levels = [row["log_level"] for row in unique_levels_df.collect()]

with open("problem1_summary.txt", "w") as f:
    print(f"Total log lines processed: {total_lines}", file=f)
    print(f"Total lines with log levels: {level_lines}", file=f)
    print(f"Unique log levels found: {unique_levels}\n", file=f)
    print("Log level distribution:", file=f)
    for level in unique_levels:
        num_level = logs_parsed.filter(col("log_level") == level).count()
        perc_level = (num_level / level_lines) * 100
        print(f"   {level}: {num_level} ({perc_level:.2f}%)", file=f)

logger.info("Summary stastic analysis complete")


# terminate spark session
logger.info("Terminating spark session")
spark.stop()
