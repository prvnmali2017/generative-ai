# Databricks notebook source
import os
notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path
%cd ..
%pip install -r ../requirements.txt

# COMMAND ----------

# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "pilot", label="Catalog Name")
dbutils.widgets.text("input_schema", "gold_nyc_flights", label="Input Schema")
dbutils.widgets.text("output_schema", "gold_nyc_flights", label="Output Schema")
dbutils.widgets.text("input_table", "nyc_flights_training", label="Input Table")
dbutils.widgets.text("output_table", "nyc_flights_arrival_delay_features", label="Output Table")
dbutils.widgets.text("features_module", "arrival_delay_features", label="Features Module")
dbutils.widgets.text("time_window_length", "86400", label="Time Window Length")
dbutils.widgets.text(
    "primary_keys",
    "time_hour,origin,dest",
    label="Primary keys columns for the feature table, comma separated.",
)
dbutils.widgets.text("timestamp_column", "time_hour", label="Timestamp column")

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
input_schema = dbutils.widgets.get("input_schema")
output_schema = dbutils.widgets.get("output_schema")
input_table = dbutils.widgets.get("input_table")
output_table = dbutils.widgets.get("output_table")
features_module = dbutils.widgets.get("features_module")
time_window_length = dbutils.widgets.get("time_window_length")
primary_keys = dbutils.widgets.get("primary_keys")
timestamp_column = dbutils.widgets.get("timestamp_column")

# COMMAND ----------

import os
notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path
%cd ../features


# COMMAND ----------

input_table_ref = f"{catalog}.{input_schema}.{input_table}"
output_table_ref = f"{catalog}.{output_schema}.{output_table}"

output_schema_ref = f"{catalog}.{output_schema}"

spark.sql("CREATE DATABASE IF NOT EXISTS " + output_schema_ref)


# COMMAND ----------

input_df = spark.read.table(input_table_ref)


# COMMAND ----------

# Compute the features. This is done by dynamically loading the features module.
from importlib import import_module

mod = import_module(features_module)
compute_features_fn = getattr(mod, "compute_features_fn")

features_df = compute_features_fn(
    input_df=input_df,
    time_window_length=int(time_window_length),
)

# COMMAND ----------

# DBTITLE 1, Write computed features.
from databricks import feature_store

fs = feature_store.FeatureStoreClient()


# Create the feature table if it does not exist first.
# Note that this is a no-op if a table with the same name and schema already exists.
fs.create_table(
    name=output_table_ref,
    primary_keys=[x.strip() for x in primary_keys.split(",")],
    timestamp_keys=[timestamp_column],
    df=features_df,
)

# Write the computed features dataframe.
fs.write_table(
    name=output_table_ref,
    df=features_df,
    mode="merge",
)

dbutils.notebook.exit(0)
