import pandas as pd
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_dataset():
    dataset = load_breast_cancer()

    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target, name="target")

    print("Dataset loaded successfully")
    print("Feature shape:", X.shape)
    print("Target shape:", y.shape)

    return X, y


def preprocess_data(X, y):
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Data preprocessing completed")
    print("Training samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])

    return X_train, X_test, y_train, y_test, scaler


def save_model(model, scaler):
    joblib.dump(model, "svm_breast_cancer_model.pkl")
    print("Model saved successfully.")


def train_model(X_train, X_test, y_train, y_test, scaler):
    print("Training the Support Vector Machine model...")

    model = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\nModel Accuracy:", accuracy)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    save_model(model, scaler)


if __name__ == "__main__":

    X, y = load_dataset()

    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)

    train_model(X_train, X_test, y_train, y_test, scaler)