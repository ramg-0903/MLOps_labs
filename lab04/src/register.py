import mlflow
import time
from mlflow.tracking import MlflowClient


def register_model(run_name, model_name):

    run_id = mlflow.search_runs(
        filter_string=f'tags.mlflow.runName = "{run_name}"'
    ).iloc[0].run_id

    model_version = mlflow.register_model(
        f"runs:/{run_id}/model",
        model_name
    )

    time.sleep(10)

    client = MlflowClient()

    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Production"
    )