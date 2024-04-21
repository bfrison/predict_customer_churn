# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This script will predict the customer churn based on bank data. It uses two different models to predict the results: a Random Forest classifier and a Logistic Regression classifier

## Files and data description
- `churn_library.py` contains all the function to be executed
- `churn_script_logging_and_tests.py` contains all the unit tests
- `data/bank_data.csv` contains the data which is used as input for both models

## Running Files
The required Python library are all written in the `requirements_py3.8.txt` file.  
A new environment can be created by executing:  
```shell
conda create -n <env_name> python=3.8.13 pip
conda activate <env_name>
pip install -r requirements_py3.8.txt
```
To run the main script, execute
```shell
python churn_library.py
```
To run the unit tests, execute:
```shell
pytest -v churn_script_logging_and_tests.py
```