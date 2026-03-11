from src.train_v1 import train_v1
from src.train_v2 import train_v2
from src.register import register_model
from src.batch_predict import batch_predict
from src.preprocess import preprocess

def main():

    preprocess()
    train_v1()
    train_v2()  
    register_model("titanic_v1", "titanic_model_v1")
    register_model("titanic_v2", "titanic_model_v2")
    batch_predict() 

if __name__ == "__main__":
    main()