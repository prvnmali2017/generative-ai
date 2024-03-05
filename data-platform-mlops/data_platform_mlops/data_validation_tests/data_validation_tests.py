# Databricks notebook source

import datetime

import pyspark.sql.functions as F
from pyspark.sql.session import SparkSession
from databricks.connect import DatabricksSession
import os

# COMMAND ----------


dbutils.widgets.text("catalog", "pilot", label="Catalog Name")
dbutils.widgets.text("schema", "gold_nyc_flights", label="Schema")
dbutils.widgets.text("feature_table", "nyc_flights_arrival_delay_features", label="Feature Table")
dbutils.widgets.text("training_table", "nyc_flights_training", label="Training Table")
dbutils.widgets.text("inference_table", "nyc_flights_inference_prediction", label="Inference Table")
dbutils.widgets.text("date_column", "time_hour", label="Date column")


catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
feature_table = dbutils.widgets.get("feature_table")
training_table = dbutils.widgets.get("training_table")
inference_table = dbutils.widgets.get("inference_table")
date_column = dbutils.widgets.get("date_column")
# COMMAND ----------

features_count = spark.read.table(f"{catalog}.{schema}.{feature_table}").count()
raw_count = spark.read.table(f"{catalog}.{schema}.{training_table}").count()

try:
    assert features_count == raw_count
except Exception as e:
    print("feature counts don't match the training data's count")


# COMMAND ----------


inference_table_df = spark.read.table(f"{catalog}.{schema}.{inference_table}")

try:
    assert (
        inference_table_df.select(F.to_date(F.max(F.col("time_hour")))).collect()[0][0]
        == datetime.date.today()
    )
except Exception as e:
    print("feature counts don't match the training data's count")

# COMMAND ----------

dbutils.notebook.exit(0)