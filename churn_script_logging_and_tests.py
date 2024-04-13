import churn_library as cl
import logging
import os
import pytest


logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

eda_dir = './images/eda'

@pytest.fixture
def data_path():
    return './data/bank_data.csv'

def test_import(data_path):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = cl.import_data(data_path)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err

@pytest.mark.parametrize('eda_path', [
    'churn_hist.png',
    'customer_age_hist.png',
    'marital_status_bar.png',
    'total_trans_ct_density.png',
    'corr_heatmap.png',
])
def test_eda(data_path, eda_path):
    '''
    test perform eda function
    '''
    df = cl.import_data(data_path)
    try:
        cl.perform_eda(df)
        logging.info('Testing perform_eda: SUCCESS')
        assert os.path.exists(os.path.join(eda_dir, eda_path))
    except AssertionError as err:
        logging.error(f'File save error: {eda_path} was not saved properly')


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''


def test_train_models(train_models):
    '''
    test train_models
    '''


if __name__ == "__main__":
    pass








