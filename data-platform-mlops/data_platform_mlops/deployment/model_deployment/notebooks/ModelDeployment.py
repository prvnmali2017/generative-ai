# Databricks notebook source


dbutils.widgets.dropdown("env", "None", ["None", "staging", "prod"], "Environment Name")

# COMMAND ----------

import os
import sys
notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path
%cd ..
sys.path.append("../..")
%ls
%pwd
%pip install -r ../../requirements.txt


# COMMAND ----------

from deploy import deploy

model_uri = dbutils.jobs.taskValues.get("Train", "model_uri", debugValue="")
env = dbutils.widgets.get("env")
assert env != "None", "env notebook parameter must be specified"
assert model_uri != "", "model_uri notebook parameter must be specified"
deploy(model_uri, env)

# COMMAND ----------
print(
    f"Successfully completed model deployment for {model_uri}"
)
