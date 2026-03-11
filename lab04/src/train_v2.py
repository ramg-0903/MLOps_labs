import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from mlflow.models.signature import infer_signature

DATA_PATH = "data/processed/processed_titanic.csv"

ITERATIONS_V2 = [10, 100, 200, 300]

EXPERIMENT_NAME = "titanic_experiment"
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    mlflow.create_experiment(EXPERIMENT_NAME)
mlflow.set_experiment(EXPERIMENT_NAME)

def train_v2():
    df = pd.read_csv(DATA_PATH)
    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_accuracy = 0
    best_model = None
    best_iter = None

    with mlflow.start_run(run_name="titanic_v2"):
        for max_iter in ITERATIONS_V2:
            model = LogisticRegression(max_iter=max_iter)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])

            mlflow.log_metric(f"accuracy_iter_{max_iter}", acc)
            mlflow.log_metric(f"auc_iter_{max_iter}", auc)

            if acc > best_accuracy:  # pick best by accuracy
                best_accuracy = acc
                best_model = model
                best_iter = max_iter

        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(best_model, artifact_path="titanic_model_v2", signature=signature)
        mlflow.log_param("best_max_iter_v2", best_iter)

        print(f"v2 Best Iteration: max_iter={best_iter}, Accuracy={best_accuracy:.4f}")

if __name__ == "__main__":
    train_v2()