#!/usr/bin/env python3
"""
Problem 2: Cluster and application analysis
"""

import logging
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructType, StructField
from pyspark.sql.functions import (
    regexp_extract, split, col, count, to_timestamp,
    max, min, mean, desc, unix_timestamp
)
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
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



# create spark session on cluster
logger.info("Creating spark session on cluster...")
master_url = sys.argv[1]
spark = create_spark_session(master_url)

# read all files into RDD from s3 bucket
logger.info("Loading data from s3 bucket into RDD...")
files_rdd = spark.sparkContext.wholeTextFiles("s3a://ima35-assignment-spark-cluster-logs/data/*/*.log")
lines_rdd = files_rdd.flatMapValues(lambda content: content.splitlines())
logger.info("Successfully loaded data from s3 bucket into RDD")

# convert to df and prepare for analysis
logger.info("Converting to dataframe...")
lines_df = lines_rdd.toDF(["file_path", "value"])
logger.info("Converted to dataframe")

logger.info("Preparing dataframe for analysis...")
lines_df = lines_df.withColumn('application_id',
    regexp_extract('file_path', r'application_(\d+_\d+)', 0))
lines_df = lines_df.withColumn("app_number", split(col("application_id"), "_").getItem(2))
lines_df = lines_df.withColumn("cluster_id", split(col("application_id"), "_").getItem(1))
lines_df = lines_df.select("cluster_id", "application_id", "app_number",
    regexp_extract('value', r'^(\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})', 1).alias('timestamp')
).filter(col("timestamp") != "")
lines_df = lines_df.withColumn("timestamp",
    to_timestamp('timestamp', 'yy/MM/dd HH:mm:ss'))
logger.info("Dataframe prepared for analysis")

# aggregate to application level
logger.info("Aggregating to application level...")
agg_df = (lines_df
    .groupby("cluster_id", "application_id", "app_number")
    .agg(
        min("timestamp").alias("start_time"),
        max("timestamp").alias("end_time")
    )
)
agg_df.toPandas().to_csv("problem2_timeline.csv", index=False)
logger.info("Aggregated to application level")

# aggregate to cluster level
logger.info("Aggregating to application level...")
cluster_df = (agg_df
    .groupby("cluster_id")
    .agg(
        count("*").alias("num_applications"),
        min("start_time").alias("cluster_first_app"),
        max("end_time").alias("cluster_last_app")
    )
)
cluster_df.toPandas().to_csv("problem2_cluster_summary.csv", index=False)
logger.info("Aggregated to application level")

# get summary stats
logger.info("Getting summary stats...")
total_unique_clusters = cluster_df.select("cluster_id").distinct().count()
total_apps = agg_df.select("application_id").distinct().count()
avg_apps_per_cluster = cluster_df.agg(mean("num_applications")).collect()[0][0]
sorted_df = cluster_df.orderBy(desc("num_applications"))

with open("problem2_stats.txt", "w") as f:
    print(f"Total unique clusters: {total_unique_clusters}", file=f)
    print(f"Total unique applications: {total_apps}", file=f)
    print(f"Average applications per cluster: {avg_apps_per_cluster}\n", file=f)
    print("Most heavily used clusters:", file=f)
    for row in sorted_df.collect():
        print(f"   Cluster {row.cluster_id}: {row.num_applications} applications", file=f)
logger.info("Completed analysis of summary stats")

# make bar chart
logger.info("Creating bar chart...")
cluster_pandas_df = cluster_df.toPandas()

fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x="cluster_id", y="num_applications", data=cluster_pandas_df, ax=ax, hue="cluster_id")
for bars in ax.containers:
        ax.bar_label(bars)
ax.set_title("Number of Applications by Cluster")

plt.savefig("problem2_bar_chart.png")
logger.info("Created bar chart")

# make histogram
logger.info("Creating histogram...")
top_cluster = sorted_df.head(1)[0][0]
duration_df = agg_df.filter(col("cluster_id") == top_cluster)
duration_df = duration_df.withColumn(
    "DurationSeconds", 
    unix_timestamp("end_time", format="yy/MM/dd HH:mm:ss") - unix_timestamp("start_time", format="yy/MM/dd HH:mm:ss")
)
duration_pandas_df = duration_df.toPandas()

fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(data=duration_pandas_df, x="DurationSeconds", kde=True)
plt.xscale("log")
ax.set_title(f"Histogram of {top_cluster} app durations (n={len(duration_pandas_df)})")

plt.savefig("problem2_density_plot.png")
logger.info("Created histogram")

# terminate spark session
logger.info("Terminating spark session")
spark.stop()
