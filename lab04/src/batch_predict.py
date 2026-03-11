import mlflow.pyfunc
import pandas as pd

MODEL_NAME = "titanic_survival_model"
DATA_PATH = "data/processed/processed_titanic.csv"


def batch_predict():

    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Production")
    df = pd.read_csv(DATA_PATH)
    X = df.drop("Survived", axis=1)

    predictions = model.predict(X)
    df["prediction"] = predictions

    df.to_csv("data/predictions.csv", index=False)
    print("Batch predictions saved to data/predictions.csv")


if __name__ == "__main__":
    batch_predict()