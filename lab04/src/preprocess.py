import pandas as pd
import os

RAW_PATH = "data/raw/titanic.csv"
PROCESSED_PATH = "data/processed/processed_titanic.csv"


def preprocess():
    df = pd.read_csv(RAW_PATH)

    # missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Encoding
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    df = pd.get_dummies(df, columns=["Embarked"])

    features = [
        "Pclass",
        "Sex",
        "Age",
        "Fare",
        "Embarked_C",
        "Embarked_Q",
        "Embarked_S",
        "Survived",
    ]

    df = df[features]

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    print("Preprocessed data saved to:", PROCESSED_PATH)


if __name__ == "__main__":
    preprocess()