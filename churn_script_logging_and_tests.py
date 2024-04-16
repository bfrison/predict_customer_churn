import churn_library as cl
import logging
import os
import pytest


logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


@pytest.fixture
def data_path():
    return './data/bank_data.csv'

@pytest.fixture
def eda_dir():
    if not os.path.exists('/tmp/churn_tests'):
        os.mkdir('/tmp/churn_tests')
    return '/tmp/churn_tests'

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

eda_paths = [
    'churn_hist.png',
    'customer_age_hist.png',
    'marital_status_bar.png',
    'total_trans_ct_density.png',
    'corr_heatmap.png',
]

@pytest.mark.parametrize('eda_path', eda_paths)
def test_eda(data_path, eda_dir, eda_path):
    '''
    test perform eda function
    '''
    df = cl.import_data(data_path)
    try:
        cl.perform_eda(df, eda_dir)
        assert os.path.exists(os.path.join(eda_dir, eda_path))
        logging.info(f'Testing perform_eda: successfully saved {eda_path}')
    except AssertionError as err:
        logging.error(f'Testing perform_eda: {eda_path} was not saved properly')

@pytest.fixture
def category_lst():
    return [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category',
    ]
        
def test_encoder_helper(data_path, category_lst):
    '''
    test encoder helper
    '''
    df = cl.import_data(data_path)
    df_churn = cl.encoder_helper(df, category_lst)
    expected_columns = [f'{cat}_Churn' for cat in category_lst]
    try:
        assert set(df.columns).issuperset(expected_columns)
        logging.info('Testing encoder_helper: SUCCESS')
    except AssertionError:
        logging.error(f'Testing encoder_helper error: {set(expected_columns) - set(df.columns)} are missing')

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








