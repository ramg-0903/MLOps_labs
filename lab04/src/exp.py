from mlflow.tracking import MlflowClient

client = MlflowClient()
experiments = client.list_experiments()  # Only ACTIVE experiments by default

for exp in experiments:
    print(f"ID: {exp.experiment_id}, Name: {exp.name}, Status: {exp.lifecycle_stage}")