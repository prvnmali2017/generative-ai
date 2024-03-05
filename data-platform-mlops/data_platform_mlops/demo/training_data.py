# Databricks notebook source

from datetime import date

import pyspark.sql.functions as F

nyc_raw = spark.read.table("pilot.demo_nyc_flights.nyc_flights")

nyc_train = nyc_raw.filter(F.col("time_hour") < date.today())

spark.sql("USE CATALOG pilot")

# COMMAND ----------

spark.sql("CREATE SCHEMA IF NOT EXISTS brnz_nyc_flights")

(
    nyc_train.write.mode("overwrite")
    .option("overWriteSchema", True)
    .saveAsTable("pilot.brnz_nyc_flights.nyc_flights_training")
)

# COMMAND ----------

spark.sql("CREATE SCHEMA IF NOT EXISTS slvr_nyc_flights")

(
    nyc_train.write.mode("overwrite")
    .option("overWriteSchema", True)
    .saveAsTable("pilot.slvr_nyc_flights.nyc_flights_training")
)

# COMMAND ----------

spark.sql("CREATE SCHEMA IF NOT EXISTS gold_nyc_flights")

(
    nyc_train.write.mode("overwrite")
    .option("overWriteSchema", True)
    .saveAsTable("pilot.gold_nyc_flights.nyc_flights_training")
)
