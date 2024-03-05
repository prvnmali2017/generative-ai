
import pathlib
import sys


sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.resolve()))

from mlflow.tracking import MlflowClient


def deploy(model_uri, env):

    print(f"Deployment running in env: {env}")

    _, model_name, version = model_uri.split("/")
    client = MlflowClient(registry_uri="databricks-uc")
    mv = client.get_model_version(model_name, version)

    target_alias = "Champion"

    if target_alias not in mv.aliases:
        client.set_registered_model_alias(
            name=model_name, alias=target_alias, version=version
        )
        print(f"Assigned alias '{target_alias}' to model version {model_uri}.")

        # remove "Challenger" alias if assigning "Champion" alias
        if target_alias == "Champion" and "Challenger" in mv.aliases:
            print(f"Removing 'Challenger' alias from model version {model_uri}.")
            client.delete_registered_model_alias(name=model_name, alias="Challenger")


if __name__ == "__main__":
    deploy(model_uri=sys.argv[1], env=sys.argv[2])
