import requests
import pandas as pd

URL = "http://localhost:5001/invocations"

sample = pd.DataFrame(
    [
        [1, 1, 25, 80, 0, 0, 1]
    ],
    columns=[
        "Pclass",
        "Sex",
        "Age",
        "Fare",
        "Embarked_C",
        "Embarked_Q",
        "Embarked_S",
    ],
)

data = {
    "dataframe_split": sample.to_dict(orient="split")
}

response = requests.post(URL, json=data)

print("Prediction:", response.json())