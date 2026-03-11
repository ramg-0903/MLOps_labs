from mlflow.tracking import MlflowClient
import mlflow
import time

MODEL_NAME = "titanic_survival_model"

def register_model(run_name, artifact_path):
    client = MlflowClient()

    # Get the latest run for the given run_name
    run = mlflow.search_runs(filter_string=f"tags.mlflow.runName = '{run_name}'").iloc[0]
    run_id = run.run_id

    # Register model
    model_version = mlflow.register_model(f"runs:/{run_id}/{artifact_path}", MODEL_NAME)
    print(f"Registered model from run {run_name} as version {model_version.version}")

    # Sleep to ensure registration completes
    time.sleep(5)

    # Promote to Production
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=model_version.version,
        stage="Production"
    )
    print(f"Model version {model_version.version} promoted to Production.")

if __name__ == "__main__":
    # Register v1
    register_model("titanic_v1", "titanic_model_v1")

    # Register v2
    register_model("titanic_v2", "titanic_model_v2")