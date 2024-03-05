# Databricks notebook source

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import os
notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path
%cd ..
# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt
# MAGIC dbutils.library.restartPython()


# COMMAND ----------

dbutils.widgets.text(
    "experiment_name",
    "/dev-nyc-flights-experiment",
    "Experiment Name",
)
dbutils.widgets.dropdown("run_mode", "dry_run", ["disabled", "dry_run", "enabled"], "Run Mode")
dbutils.widgets.dropdown("enable_baseline_comparison", "true", ["true", "false"], "Enable Baseline Comparison")
dbutils.widgets.text("validation_input", "pilot.gold_nyc_flights.nyc_flights_inference", "Validation Input")

dbutils.widgets.text("model_type", "regressor", "Model Type")
dbutils.widgets.text("targets", "arr_delay", "Targets")
dbutils.widgets.text("custom_metrics_loader_function", "custom_metrics", "Custom Metrics Loader Function")
dbutils.widgets.text("validation_thresholds_loader_function", "validation_thresholds", "Validation Thresholds Loader Function")
dbutils.widgets.text("evaluator_config_loader_function", "evaluator_config", "Evaluator Config Loader Function")
dbutils.widgets.text("model_name", "pilot.brnz_nyc_flights.nyc_flights_regression_model", "Full (Three-Level) Model Name")
dbutils.widgets.text("model_version", "2", "Candidate Model Version")

# COMMAND ----------

print(
    "Currently model validation is not supported for models registered with feature store. Please refer to "
    "issue https://github.com/databricks/mlops-stacks/issues/70 for more details."
)
# dbutils.notebook.exit(0)
run_mode = dbutils.widgets.get("run_mode").lower()
assert run_mode == "disabled" or run_mode == "dry_run" or run_mode == "enabled"

if run_mode == "disabled":
    print(
        "Model validation is in DISABLED mode. Exit model validation without blocking model deployment."
    )
    dbutils.notebook.exit(0)
dry_run = run_mode == "dry_run"

if dry_run:
    print(
        "Model validation is in DRY_RUN mode. Validation threshold validation failures will not block model deployment."
    )
else:
    print(
        "Model validation is in ENABLED mode. Validation threshold validation failures will block model deployment."
    )

# COMMAND ----------

import importlib
import mlflow
import os
import tempfile
import traceback

from mlflow.tracking.client import MlflowClient

client = MlflowClient(registry_uri="databricks-uc")

# set experiment
experiment_name = dbutils.widgets.get("experiment_name")
mlflow.set_experiment(experiment_name)

# set model evaluation parameters that can be inferred from the job
model_uri = dbutils.jobs.taskValues.get("Train", "model_uri", debugValue="")
model_name = dbutils.jobs.taskValues.get("Train", "model_name", debugValue="")
model_version = dbutils.jobs.taskValues.get("Train", "model_version", debugValue="")

if model_uri == "":
    model_name = dbutils.widgets.get("model_name")
    model_version = dbutils.widgets.get("model_version")
    model_uri = f"models:/{model_name}/{model_version}"

baseline_model_uri = f"models:/{model_name}@Champion"

evaluators = "default"
assert model_uri != "", "model_uri notebook parameter must be specified"
assert model_name != "", "model_name notebook parameter must be specified"
assert model_version != "", "model_version notebook parameter must be specified"

# COMMAND ----------

# take input
enable_baseline_comparison = dbutils.widgets.get("enable_baseline_comparison")
assert enable_baseline_comparison == "true" or enable_baseline_comparison == "false"
enable_baseline_comparison = enable_baseline_comparison == "true"

validation_input = dbutils.widgets.get("validation_input")
assert validation_input is not None

validation_input_table = spark.read.table(validation_input)


from databricks.feature_store import FeatureStoreClient

mlflow.set_registry_uri("databricks-uc")
fs_client = FeatureStoreClient()


candidate_predictions = fs_client.score_batch(
    model_uri,
    validation_input_table)

candidate_predictions = candidate_predictions.toPandas()
candidate_predictions = candidate_predictions.dropna()

if enable_baseline_comparison:
    
    baseline_predictions = fs_client.score_batch(
        model_uri,
        validation_input_table)
    
    baseline_predictions = baseline_predictions.toPandas()
    baseline_predictions = baseline_predictions.dropna()

    assert baseline_predictions is not None

# COMMAND ----------

import os
notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path
%cd ..

# COMMAND ----------

model_type = dbutils.widgets.get("model_type")
targets = dbutils.widgets.get("targets")

assert model_type
assert targets

custom_metrics_loader_function_name = dbutils.widgets.get("custom_metrics_loader_function")
validation_thresholds_loader_function_name = dbutils.widgets.get("validation_thresholds_loader_function")
evaluator_config_loader_function_name = dbutils.widgets.get("evaluator_config_loader_function")
assert custom_metrics_loader_function_name
assert validation_thresholds_loader_function_name
assert evaluator_config_loader_function_name
custom_metrics_loader_function = getattr(
    importlib.import_module("validation"), custom_metrics_loader_function_name
)
validation_thresholds_loader_function = getattr(
    importlib.import_module("validation"), validation_thresholds_loader_function_name
)
evaluator_config_loader_function = getattr(
    importlib.import_module("validation"), evaluator_config_loader_function_name
)
custom_metrics = custom_metrics_loader_function()
validation_thresholds = validation_thresholds_loader_function()
evaluator_config = evaluator_config_loader_function()

# COMMAND ----------

# helper methods
def get_run_link(run_info):
    return f"[Run](#mlflow/experiments/{run_info.experiment_id}/runs/{run_info.run_id})"


def get_training_run(model_name, model_version):
    version = client.get_model_version(model_name, model_version)
    return mlflow.get_run(run_id=version.run_id)


def generate_run_name(training_run):
    return None if not training_run else training_run.info.run_name + "-validation"


def generate_description(training_run):
    return (
        None
        if not training_run
        else f"Model Training Details: {get_run_link(training_run.info)}\n"
    )


def log_to_model_description(run, success):
    run_link = get_run_link(run.info)
    description = client.get_model_version(model_name, model_version).description
    status = "SUCCESS" if success else "FAILURE"
    if description != "":
        description += "\n\n---\n\n"
    description += f"Model Validation Status: {status}\nValidation Details: {run_link}"
    client.update_model_version(
        name=model_name, version=model_version, description=description
    )

# COMMAND ----------

training_run = get_training_run(model_name, model_version)

# run evaluate
with mlflow.start_run(
    run_name=generate_run_name(training_run),
    description=generate_description(training_run),
) as run, tempfile.TemporaryDirectory() as tmp_dir:
    validation_thresholds_file = os.path.join(tmp_dir, "validation_thresholds.txt")
    with open(validation_thresholds_file, "w") as f:
        if validation_thresholds:
            for metric_name in validation_thresholds:
                f.write(f"{metric_name:30}  {validation_thresholds[metric_name]}\n")
    mlflow.log_artifact(validation_thresholds_file)


    try:
        baseline_results = mlflow.evaluate(
            data=baseline_predictions,
            targets=targets,
            model_type=model_type,
            evaluators=evaluators,
            custom_metrics=custom_metrics,
            evaluator_config=evaluator_config,
            predictions="prediction"
        )

    except Exception as err:
        print(f"Baseline Model validation failed. This will not block model validation or block model deployment. {err}") 

    try:
        candidate_results = mlflow.evaluate(
            data=candidate_predictions,
            targets=targets,
            model_type=model_type,
            evaluators=evaluators,
            validation_thresholds=validation_thresholds,
            custom_metrics=custom_metrics,
            evaluator_config=evaluator_config,
            predictions="prediction"
        )
        
        metrics_file = os.path.join(tmp_dir, "metrics.txt")
        with open(metrics_file, "w") as f:
            f.write(
                f"{'metric_name' :<30} {'candidate':30}  {'baseline'}\n"
            )
            for metric in candidate_results.metrics:
                validation_metric_value = str(candidate_results.metrics[metric])
                baseline_metric_value = "N/A"
                if enable_baseline_comparison and metric in baseline_results.metrics:
                    baseline_metric_value = str(
                        baseline_results.metrics[metric]
                    )
                    mlflow.log_metric(
                        f"baseline_{metric}", baseline_metric_value
                    )
                f.write(
                    f"{metric:<30}  {validation_metric_value:<30}  {baseline_metric_value}\n"
                )
        mlflow.log_artifact(metrics_file)
        log_to_model_description(run, True)
        
        # Assign "Challenger" alias to indicate model version has passed validation checks
        print("Validation checks passed. Assigning 'Challenger' alias to model version.")
        client.set_registered_model_alias(model_name, "Challenger", model_version)
        
    except Exception as err:
        log_to_model_description(run, False)
        error_file = os.path.join(tmp_dir, "error.txt")
        with open(error_file, "w") as f:
            f.write("Validation failed : " + str(err) + "\n")
            f.write(traceback.format_exc())
        mlflow.log_artifact(error_file)
        if not dry_run:
            raise err
        else:
            print(
                "Model validation failed in DRY_RUN. It will not block model deployment."
            )
