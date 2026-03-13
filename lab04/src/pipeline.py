from preprocess import preprocess
from train_v1 import train_baseline
from train_v2 import train_tuned
from register import register_model


MODEL_NAME = "titanic_survival_model"


def run_pipeline():

    print("Starting pipeline")

    print("Running preprocessing")
    X_train, X_test, y_train, y_test = preprocess()

    print("Training baseline model (v1)")
    train_baseline(X_train, y_train, X_test, y_test)

    print("Registering baseline model")
    register_model(
        run_name="logit_v1_baseline",
        model_name=MODEL_NAME
    )

    print("Training tuned model (v2)")
    train_tuned(X_train, y_train, X_test, y_test)

    print("Registering tuned model")
    register_model(
        run_name="logit_v2_tuned",
        model_name=MODEL_NAME
    )

    print("Pipeline completed!")


if __name__ == "__main__":
    run_pipeline()