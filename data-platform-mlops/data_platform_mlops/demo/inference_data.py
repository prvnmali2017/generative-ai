# Databricks notebook source

import pyspark.sql.functions as F
from pyspark.sql.dataframe import DataFrame

nyc_raw = spark.read.table("pilot.demo_nyc_flights.nyc_flights")
nyc_inference_today = nyc_raw.filter(F.col("time_hour") == date.today())

# COMMAND ----------
from datetime import date


spark.sql("CREATE SCHEMA IF NOT EXISTS brnz_nyc_flights")
spark.sql("CREATE DATABASE IF NOT EXISTS pilot.brnz_nyc_flights.nyc_flights_inference")

nyc_inference_today.write.mode("append").saveAsTable("pilot.brnz_nyc_flights.nyc_flights_inference")


# COMMAND ----------

spark.sql("CREATE SCHEMA IF NOT EXISTS slvr_nyc_flights")
spark.sql("CREATE DATABASE IF NOT EXISTS pilot.slvr_nyc_flights.nyc_flights_inference")

nyc_inference_today.write.mode("append").saveAsTable("pilot.slvr_nyc_flights.nyc_flights_inference")




# COMMAND ----------

spark.sql("CREATE SCHEMA IF NOT EXISTS gold_nyc_flights")
spark.sql("CREATE DATABASE IF NOT EXISTS pilot.gold_nyc_flights.nyc_flights_inference")

nyc_inference_today.write.mode("append").saveAsTable("pilot.gold_nyc_flights.nyc_flights_inference")

