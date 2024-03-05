# Databricks notebook source
# COMMAND ----------

%pip install "https://ml-team-public-read.s3.amazonaws.com/wheels/data-monitoring/a4050ef7-b183-47a1-a145-e614628e3146/databricks_lakehouse_monitoring-0.4.4-py3-none-any.whl"

dbutils.library.restartPython()

# COMMAND ----------

nyc_raw = spark.read.table("pilot.demo_nyc_flights.nyc_flights")

nyc_train = nyc_raw.filter(nyc_raw.month < 12)
nyc_inference = nyc_raw.filter(nyc_raw.month >= 12)

# COMMAND ----------
spark.sql("USE CATALOG pilot")
spark.sql("CREATE SCHEMA IF NOT EXISTS brnz_nyc_flights")

(
    nyc_train.write.mode("overwrite")
    .option("overWriteSchema", True)
    .saveAsTable("pilot.brnz_nyc_flights.nyc_flights")
)



# COMMAND ----------
from databricks import lakehouse_monitoring as lm
import time

catalog = "pilot"
schema = "brnz_nyc_flights"
table_name = "nyc_flights"

TABLE_REF = f"{catalog}.{schema}.{table_name}"

try:
    lm.delete_monitor(table_name=TABLE_REF)
except Exception as e:
    print(f"Couldn't delete monitor: {e}")

snapshot_monitor = lm.create_monitor(
    table_name=TABLE_REF,
    profile_type=lm.Snapshot(),
    output_schema_name=f"{catalog}.{schema}",
)

while snapshot_monitor.status == lm.MonitorStatus.PENDING:
    snapshot_monitor = lm.get_monitor(table_name=TABLE_REF)
    time.sleep(10)

# COMMAND ----------

refreshes = lm.list_refreshes(table_name=TABLE_REF)
assert len(refreshes) > 0