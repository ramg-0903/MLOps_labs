import pandas as pd
import os
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw/titanic.csv"
PROCESSED_PATH = "data/processed/processed_titanic.csv"


def preprocess():
    df = pd.read_csv(RAW_PATH)

    # missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # categorical variables
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df = pd.get_dummies(df, columns=["Embarked"])

    # Select relevant features
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

    # Save processed dataset
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    print("Preprocessed data saved to:", PROCESSED_PATH)

    # Split features and target
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test