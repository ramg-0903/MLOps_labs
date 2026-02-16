from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from src.lab import load_data, data_preprocessing, build_save_model, load_model


# default arguments for your DAG
default_args = {
    'owner': 'Ramgopal Reddy',
    'start_date': datetime(2026, 1, 15),
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

# DAG instance
dag = DAG(
    'Airflow_Lab1',
    default_args=default_args,
    description='Titanic Logistic Regression ML Pipeline',
    schedule_interval=None,
    catchup=False,
)

# load data
load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data,
    dag=dag,
)

# data preprocessing
data_preprocessing_task = PythonOperator(
    task_id='data_preprocessing_task',
    python_callable=data_preprocessing,
    op_args=[load_data_task.output],
    dag=dag,
)

# build and save model
build_save_model_task = PythonOperator(
    task_id='build_save_model_task',
    python_callable=build_save_model,
    op_args=[data_preprocessing_task.output, "titanic_model.sav"],
    provide_context=True,
    dag=dag,
)

# load model
load_model_task = PythonOperator(
    task_id='load_model_task',
    python_callable=load_model,
    op_args=["titanic_model.sav", build_save_model_task.output],
    dag=dag,
)

# task dependencies
load_data_task >> data_preprocessing_task >> build_save_model_task >> load_model_task

# main function 
if __name__ == "__main__":
    dag.cli()
