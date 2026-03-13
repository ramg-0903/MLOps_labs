import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from mlflow.models.signature import infer_signature


def train_baseline(X_train, y_train, X_test, y_test):

    with mlflow.start_run(run_name="logit_v1_baseline"):

        model = LogisticRegression(max_iter=1000)

        model.fit(X_train, y_train)

        preds = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, preds)

        # log parameters
        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_param("max_iter", 1000)

        # log metrics
        mlflow.log_metric("auc", auc)

        # infer model signature
        signature = infer_signature(X_train, preds)

        # log model with signature
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature
        )

    return model