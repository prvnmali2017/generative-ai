import pyspark.sql
import pytest
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql import Row

from data_platform_mlops.feature_engineering.features.arrival_delay_features import (
    compute_features_fn,
)


@pytest.fixture(scope="session")
def spark(request):
    """fixture for creating a spark session
    Args:
        request: pytest.FixtureRequest object
    """
    spark = (
        SparkSession.builder.master("local[1]")
        .appName("pytest-pyspark-local-testing")
        .getOrCreate()
    )
    request.addfinalizer(lambda: spark.stop())

    return spark


@pytest.mark.usefixtures("spark")
def test_arrival_delay_fn(spark):
    # Sample data
    data = [
            Row(
                dest='IAH',
                origin='EWR',
                time_hour=datetime(2022, 1, 10),
                arr_delay=10
            )
        ]

    schema = "dest STRING, origin STRING, time_hour TIMESTAMP, arr_delay INT"
    spark_df = spark.createDataFrame(data, schema=schema)
    
    
    output_df = compute_features_fn(spark_df)
    
    assert isinstance(output_df, pyspark.sql.DataFrame)
    assert output_df.count() == 1 