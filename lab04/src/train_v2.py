import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from mlflow.models.signature import infer_signature


def train_tuned(X_train, y_train, X_test, y_test):

    with mlflow.start_run(run_name="logit_v2_tuned"):

        model = LogisticRegression(
            C=0.5,
            penalty="l2",
            max_iter=2000
        )

        model.fit(X_train, y_train)

        preds = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, preds)

        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_param("C", 0.5)
        mlflow.log_param("penalty", "l2")
        mlflow.log_param("max_iter", 2000)

        mlflow.log_metric("auc", auc)

        signature = infer_signature(X_train, preds)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature
        )

    return model