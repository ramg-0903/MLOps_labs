# MLOps lab-3
This is the repo for docker lab-1. This counts towards the third MLOps lab submission for the IE 7374 course

# Breast Cancer SVM Classifier with Docker

This project demonstrates training a Support Vector Machine (SVM) model on the **Breast Cancer Wisconsin dataset** using `scikit-learn` and packaging it with **Docker** for reproducible deployment.

---

## Project Structure


- **`src/train.py`** – Python script used for the project. It has the following functions:
    - load_model: To load the breast cancer dataset
    - preprocess_data: To applying basic pre-processing like standard scaler
    - train_model: To train the SVM classifier and evaluate test performance
    - save_model: To save the model as a pickle file

- **`requirements.txt`** – Python dependencies required to run the project (`pandas`, `scikit-learn`, `joblib`).
- **`Dockerfile`** – Defines the Docker image for this project.
- **`.dockerignore`** – Excludes unnecessary files from the Docker build context.

---


## Docker Setup

### `requirements.txt`:
- pandas
- scikit-learn
- joblib

### Dockerfile

- **docker build -t lab1:v1 .** : This command builds our docker image
- **docker save lab1:v1 > my_image.tar** : it exports the docker image as a .tar file
- **docker run lab1:v1** : It starts running a docker image and runs our main.py function


