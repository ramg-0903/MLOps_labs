import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os
import base64

def load_data():
    # loading our training data and serializing it 

    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/train.csv"))
    serialized_data = pickle.dumps(df)
    print("loading data")
    return base64.b64encode(serialized_data).decode("ascii")

def data_preprocessing(data_b64: str):
    # this function pre-processes the data and returns serialised output

    # decode
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    # handling missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # feature selection
    df = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]]

    # encode categoricals
    df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    serialized = pickle.dumps((X_scaled, y, scaler, X.columns.tolist()))
    print("pre-processed data")
    return base64.b64encode(serialized).decode("ascii")


def build_save_model(data_b64: str, filename: str):
    # to train our logistic regression model

    data_bytes = base64.b64decode(data_b64)
    X_scaled, y, scaler, columns = pickle.loads(data_bytes)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, y)

    predictions = model.predict(X_scaled)
    accuracy = accuracy_score(y, predictions)

    # Save model + scaler + columns
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    with open(output_path, "wb") as f:
        pickle.dump((model, scaler, columns), f)
    print("built and saved model")
    return float(accuracy)  # JSON safe


def load_model(filename: str, metric):
    # loads our logit model and evaluates it on the test set

    print(f"Training Accuracy: {metric}")

    model_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    model, scaler, columns = pickle.load(open(model_path, "rb"))

    df_test = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/test.csv"))

    # same preprocessing as training
    df_test["Age"] = df_test["Age"].fillna(df_test["Age"].median())
    df_test["Fare"] = df_test["Fare"].fillna(df_test["Fare"].median())
    df_test["Embarked"] = df_test["Embarked"].fillna(df_test["Embarked"].mode()[0])

    df_test = df_test[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
    df_test = pd.get_dummies(df_test, columns=["Sex", "Embarked"], drop_first=True)

    # aligning columns with our training data
    for col in columns:
        if col not in df_test:
            df_test[col] = 0

    df_test = df_test[columns]

    X_test_scaled = scaler.transform(df_test)

    predictions = model.predict(X_test_scaled)

    print("tested the model on the test set")
    # returns first prediction
    return int(predictions[0])
