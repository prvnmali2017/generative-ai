# Databricks notebook source

import os

notebook_path = "/Workspace/" + os.path.dirname(
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .notebookPath()
    .get()
)
%cd $notebook_path
%cd ..
# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Notebook Environment
dbutils.widgets.dropdown("env", "staging", ["staging", "prod"], "Environment Name")
env = dbutils.widgets.get("env")

dbutils.widgets.text(
    "experiment_name",
    "/dev-nyc-flights-experiment",
    label="MLflow experiment name",
)

dbutils.widgets.text(
    "model_name",
    "pilot.gold_nyc_flights.nyc_flights_regression_model",
    label="Full (Three-Level) Model Name",
)

dbutils.widgets.text(
    "arrival_delay_features_table",
    "pilot.gold_nyc_flights.nyc_flights_arrival_delay_features",
    label="Arrival Delay Features Table",
)

dbutils.widgets.text("catalog", "pilot", label="Catalog Name")
dbutils.widgets.text("input_schema", "gold_nyc_flights", label="Input Schema")
dbutils.widgets.text("input_table", "nyc_flights_training", label="Input Table")


# COMMAND ----------


experiment_name = dbutils.widgets.get("experiment_name")
model_name = dbutils.widgets.get("model_name")
catalog = dbutils.widgets.get("catalog")
input_schema = dbutils.widgets.get("input_schema")
input_table = dbutils.widgets.get("input_table")

input_table_ref = f"{catalog}.{input_schema}.{input_table}"


# COMMAND ----------

import mlflow

mlflow.set_experiment(experiment_name)
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

nyc_flights_train = spark.read.table(input_table_ref)
nyc_flights_train.count()

# COMMAND ----------


import mlflow.pyfunc


def get_latest_model_version(model_name):
    latest_version = 1
    mlflow_client = MlflowClient()
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version


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
%cd ..

# COMMAND ----------

import mlflow
from databricks.feature_store import FeatureLookup

arrival_delay_features_table = dbutils.widgets.get("arrival_delay_features_table")

arrival_delay_lookup = [
    FeatureLookup(
        table_name=arrival_delay_features_table,
        feature_names=[
            "weekly_arr_delay",
        ],
        lookup_key=["origin", "dest"],
        timestamp_lookup_key=["time_hour"],
    ),
]

# COMMAND ----------

from databricks import feature_store
import transform
import hyperparameter_tuning
from sklearn.linear_model import LinearRegression
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from hyperopt import fmin

fs = feature_store.FeatureStoreClient()

training_set = fs.create_training_set(
    nyc_flights_train, 
    feature_lookups=arrival_delay_lookup, 
    label="arr_delay",
    exclude_columns=["flight", "talinum"]
)

training_df = training_set.load_df()
features_and_label = training_df.columns
data = (training_df.toPandas()[features_and_label]).dropna()

X_train = data.drop(["arr_delay"], axis=1)
y_train = data["arr_delay"]


# COMMAND ----------
from sklearn.model_selection import KFold, cross_val_score

NUM_FOLDS = 5
RANDOM_STATE = 123

def train(parameters):

    mlflow.sklearn.autolog()

    with mlflow.start_run(nested=True):
        
        model = transform.transformer_fn()
        model.steps.append(["regressor", LinearRegression(**parameters)])
        kfolds = KFold(n_splits = NUM_FOLDS, random_state = RANDOM_STATE, shuffle = True)
        validation_scorer_name, validation_scorer_fn = hyperparameter_tuning.load_validation_scorer_fn()
        avg_cross_val_score = cross_val_score(
            model, 
            X_train, 
            y_train, 
            cv = kfolds, 
            scoring = validation_scorer_fn).mean()

        mlflow.log_metric(f"cv_{validation_scorer_name}", avg_cross_val_score)

        return avg_cross_val_score

load_search_space = hyperparameter_tuning.load_search_space_fn()

with mlflow.start_run(run_name = "hyperopt"):
    best_param = fmin(
        fn = train,
        space = load_search_space(),
        max_evals = 4
    )


# COMMAND ----------

validation_metric_name, _ = hyperparameter_tuning.load_validation_scorer_fn()

best_run = mlflow.search_runs(
    order_by=[f"metrics.cv_{validation_metric_name}"],
    max_results=3
).iloc[0]

best_model = mlflow.sklearn.load_model(
    f"runs://{best_run.run_id}/model"
)

mlflow.end_run()


# COMMAND ----------

model = LinearRegression(X_train, y_train, positive = best_model["positive"], fit_intercept = best_model["fit_intercept"])


fs.log_model(
    best_model,
    artifact_path="model_packaged",
    flavor=mlflow.sklearn,
    training_set=training_set,
    registered_model_name=model_name,
)


model_version = get_latest_model_version(model_name)
model_uri = f"models:/{model_name}/{model_version}"
dbutils.jobs.taskValues.set("model_uri", model_uri)
dbutils.jobs.taskValues.set("model_name", model_name)
dbutils.jobs.taskValues.set("model_version", model_version)
dbutils.notebook.exit(model_uri)

# COMMAND ----------

