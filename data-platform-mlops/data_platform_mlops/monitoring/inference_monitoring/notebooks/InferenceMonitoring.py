# Databricks notebook source

# COMMAND ----------

import os

notebook_path = "/Workspace/" + os.path.dirname(
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .notebookPath()
    .get()
)
%cd $notebook_path

# COMMAND ----------

%pip install "https://ml-team-public-read.s3.amazonaws.com/wheels/data-monitoring/a4050ef7-b183-47a1-a145-e614628e3146/databricks_lakehouse_monitoring-0.4.4-py3-none-any.whl"

# COMMAND ----------

# MAGIC %pip install -r ../../../requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "pilot", label="Catalog Name")
dbutils.widgets.text("schema", "gold_nyc_flights", label="Schema")
dbutils.widgets.text("inference_table", "nyc_flights_inference_prediction", label="Input Table")
dbutils.widgets.text("training_table", "nyc_flights_training_prediction", label="Input Table")

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
inference_table = dbutils.widgets.get("inference_table")
training_table = dbutils.widgets.get("training_table")

inference_table_ref = f"{catalog}.{schema}.{inference_table}"
training_table_ref = f"{catalog}.{schema}.{training_table}"

# COMMAND ----------

from databricks import lakehouse_monitoring as lm
import time

# Window sizes to analyze data over
GRANULARITIES = ["1 day"]

try:
  lm.delete_monitor(table_name=inference_table_ref)
except Exception as e:
  print(f"the monitor table doesn't exist or another issue ocurred: {e}")
finally:
  inference_monitor = lm.create_monitor(
    table_name=inference_table_ref,
    profile_type=lm.InferenceLog(
      timestamp_col="time_hour",
      granularities=GRANULARITIES,
      model_id_col="model_version",
      problem_type="regression",
      prediction_col="prediction",
      label_col="arr_delay"
    ),
    baseline_table_name=training_table_ref,
    output_schema_name=f"{catalog}.{schema}"
  )

# COMMAND ----------

while inference_monitor.status == lm.MonitorStatus.PENDING:
    inference_monitor = lm.get_monitor(table_name=inference_table_ref)
    time.sleep(10)