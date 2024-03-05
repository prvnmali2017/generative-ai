import pyspark.sql.functions as F
from pyspark.sql import dataframe
from pyspark.sql.window import Window


def compute_features_fn(
    input_df: dataframe, time_window_length: int = 86400
) -> dataframe:
    weekly_window = (
        Window.partitionBy(F.col("origin"), F.col("dest"))
        .orderBy(F.col("time_hour").cast("long"))
        .rangeBetween(-time_window_length, 0)
    )

    arrival_delay_features = (
        input_df.withColumn(
            "weekly_arr_delay", F.mean(F.col("arr_delay")).over(weekly_window)
        )
        .select(
            F.col("time_hour"),
            F.col("origin"),
            F.col("dest"),
            F.col("weekly_arr_delay"),
        )
        .distinct()
        .orderBy(F.col("time_hour"), F.col("origin"), F.col("dest"))
    )

    return arrival_delay_features



