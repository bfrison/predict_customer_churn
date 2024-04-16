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

@pytest.fixture
def df(data_path):
    return cl.import_data(data_path)

eda_paths = [
    'churn_hist.png',
    'customer_age_hist.png',
    'marital_status_bar.png',
    'total_trans_ct_density.png',
    'corr_heatmap.png',
]

def test_eda(df, tmp_path):
    '''
    test perform eda function
    '''
    try:
        cl.perform_eda(df, str(tmp_path.absolute()))
        logging.info(f'Testing perform_eda: SUCCESS')
    except Error as err:
        logging.error(f'Testing perform_eda: ERROR')
        raise err

@pytest.mark.parametrize('eda_path', eda_paths)
def test_eda_paths(tmp_path, eda_path):
    try:
        assert os.path.exists(os.path.join(tmp_path.absolute(), eda_path))
        logging.info(f'Testing perform_eda: successfully saved {eda_path}')
    except AssertionError as err:
        logging.error(f'Testing perform_eda: {eda_path} was not saved properly')

category_lst = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category',
]

@pytest.fixture
def category_lst_fix():
    return category_lst
        
def test_encoder_helper(df, category_lst_fix):
    '''
    test encoder helper
    '''
    try:
        df_churn = cl.encoder_helper(df, category_lst_fix)
        logging.info('Testing encoder_helper: SUCCESS')
    except Error as err:
        logging.error(f'Testing encoder_helper: ERROR')
        raise err

@pytest.fixture
def df_churn(df):
    return cl.encoder_helper(df, category_lst)

@pytest.mark.parametrize('cat', category_lst)
def test_encoder_helper_columns(df_churn, cat):
    '''
    test encoder helper
    '''
    column_name = f'{cat}_Churn'
    try:
        assert column_name in df_churn.columns
        logging.info(f'Testing encoder_helper: {column_name} found in dataframe')
    except AssertionError as err:
        logging.error(f'Testing encoder_helper error: {column_name} is missing')
        raise err

def test_perform_feature_engineering(df_churn):
    '''
    test perform_feature_engineering
    '''
    try:
        X_train, X_test, y_train, y_test = cl.perform_feature_engineering(df_churn)
        logging.info('Testing perform_feature_engineering: SUCCESS')
    except Error as err:
        logging.error('Testing perform_feature_engineering: ERROR')

@pytest.fixture
def split_dfs(df_churn):
    return cl.perform_feature_engineering(df_churn)

dfs_shapes = [
    (0, 0, 7088, 'X_train number of rows'),
    (0, 1, 19, 'X_train number of columns'),
    (1, 0, 3039, 'X_test number of rows'),
    (1, 1, 19, 'X_test number of columns'),
    (2, 0, 7088, 'y_train number of rows'),
    (3, 0, 3039, 'y_test number of rows'),
]

@pytest.mark.parametrize('df_index,shape_index,expected_val,assertion_message', dfs_shapes)
def test_perform_feature_engineering_shapes(split_dfs, df_index, shape_index, expected_val, assertion_message):
    val = split_dfs[df_index].shape[shape_index]
    try:
        assert val == expected_val
        logging.info(f'Testing perform_feature_engineering: {assertion_message} is correct')
    except AssertionError as err:
        logging.error(f'Testing perform_feature_engineering error: {assertion_message} is incorrect, expected {expected_val:d} instead of {val:d}')
        raise err

def test_train_models(train_models):
    '''
    test train_models
    '''


if __name__ == "__main__":
    pass








