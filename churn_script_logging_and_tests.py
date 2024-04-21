'''
This modules contains all the unit tests to be run against churn_library
'''

import logging
import os
import pytest

import joblib

import churn_library as cl


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


@pytest.fixture
def data_path():
    '''
    provides the path of the data csv file
    '''
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
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


@pytest.fixture
def df_data(data_path):
    '''
    returns the dataframe
    '''
    return cl.import_data(data_path)


eda_paths = [
    'churn_hist.png',
    'customer_age_hist.png',
    'marital_status_bar.png',
    'total_trans_ct_density.png',
    'corr_heatmap.png',
]


@pytest.fixture(scope='module')
def mod_tmp_path(tmp_path_factory):
    '''
    Creates a single temparory directory that will presist throughout the whole module
    '''
    return tmp_path_factory.mktemp('data')


def test_eda(df_data, mod_tmp_path):
    '''
    test perform eda function
    '''
    try:
        cl.perform_eda(df_data, mod_tmp_path)
        logging.info('Testing perform_eda: SUCCESS')
    except Exception as err:
        logging.error('Testing perform_eda: ERROR')
        raise err


@pytest.mark.parametrize('eda_path', eda_paths)
def test_eda_paths(mod_tmp_path, eda_path):
    '''
    test that all eda images are properly saved
    '''
    try:
        assert os.path.exists(os.path.join(mod_tmp_path, eda_path))
        logging.info('Testing perform_eda: successfully saved %s', eda_path)
    except AssertionError as err:
        logging.error(
            'Testing perform_eda: %s was not saved properly', eda_path)
        raise err


category_lst = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category',
]


@pytest.fixture
def category_lst_fix():
    '''
    Returns the list of categorical columns
    '''
    return category_lst


def test_encoder_helper(df_data, category_lst_fix):
    '''
    test encoder helper
    '''
    try:
        cl.encoder_helper(df_data, category_lst_fix)
        logging.info('Testing encoder_helper: SUCCESS')
    except Exception as err:
        logging.error('Testing encoder_helper: ERROR')
        raise err


@pytest.fixture
def df_churn(df_data):
    '''
    returns a dataframe where categorical columns have been encoded
    '''
    return cl.encoder_helper(df_data, category_lst)


@pytest.mark.parametrize('cat', category_lst)
def test_encoder_helper_columns(df_churn, cat):
    '''
    test whether all columns have been properly encoded
    '''
    column_name = f'{cat}_Churn'
    try:
        assert column_name in df_churn.columns
        logging.info(
            'Testing encoder_helper: %s found in dataframe', column_name)
    except AssertionError as err:
        logging.error(
            'Testing encoder_helper error: %s is missing', column_name)
        raise err


def test_perform_feature_engineering(df_churn):
    '''
    test perform_feature_engineering
    '''
    try:
        cl.perform_feature_engineering(
            df_churn)
        logging.info('Testing perform_feature_engineering: SUCCESS')
    except Exception as err:
        logging.error('Testing perform_feature_engineering: ERROR')
        raise err


@pytest.fixture
def split_dfs(df_churn):
    '''
    Returns of tuples where the data has been split in train and test data sets
    '''
    return cl.perform_feature_engineering(df_churn)


dfs_shapes = [
    (0, 0, 7088, 'X_train number of rows'),
    (0, 1, 19, 'X_train number of columns'),
    (1, 0, 3039, 'X_test number of rows'),
    (1, 1, 19, 'X_test number of columns'),
    (2, 0, 7088, 'y_train number of rows'),
    (3, 0, 3039, 'y_test number of rows'),
]


@pytest.mark.parametrize(
    'df_index,shape_index,expected_val,assertion_message',
    dfs_shapes)
def test_perform_feature_engineering_shapes(
        split_dfs,
        df_index,
        shape_index,
        expected_val,
        assertion_message):
    '''
    Test whether the dataframe is of the proper shape after feature engineering
    '''
    val = split_dfs[df_index].shape[shape_index]
    try:
        assert val == expected_val
        logging.info(
            'Testing perform_feature_engineering: %s is correct', assertion_message)
    except AssertionError as err:
        logging.error(
            'Testing perform_feature_engineering error: \
            %s is incorrect, expected %d \
            instead of %d', assertion_message, expected_val, val)
        raise err


def test_train_models(split_dfs, mod_tmp_path):
    '''
    test train_models
    '''
    X_train, X_test, y_train, y_test = split_dfs
    try:
        cl.train_models(X_train, X_test, y_train, y_test, mod_tmp_path)
        logging.info('Testing train_models: SUCCESS')
    except Exception as err:
        logging.error('Testing train_models: ERROR')
        raise err


@pytest.fixture
def rfc(mod_tmp_path):
    '''
    loads the random forest classifier model from its pickle file
    '''
    return joblib.load(os.path.join(mod_tmp_path, 'rfc_model.pkl'))


@pytest.fixture
def lrc(mod_tmp_path):
    '''
    loads the logistic regression classifier model from its pickle file
    '''
    return joblib.load(os.path.join(mod_tmp_path, 'logistic_model.pkl'))


def test_feature_importance_plot(rfc, split_dfs, mod_tmp_path):
    '''
    test feature importance plot
    '''
    model = rfc
    X_train = split_dfs[0]
    model_name = str(model).split('(')[0]
    try:
        cl.feature_importance_plot(model, X_train, mod_tmp_path)
        logging.info(
            'Testing feature_importance_plot for %s: SUCCESS', model_name)
    except Exception as err:
        raise err


def test_feature_importance_plot_path(rfc, mod_tmp_path):
    '''
    test whether all plots are properly saved
    '''
    model = rfc
    model_name = str(model).split('(')[0]
    file_name = f'{model_name}_feature_importances.png'
    try:
        assert os.path.exists(os.path.join(mod_tmp_path, file_name))
        logging.info(
            'Testing feature_importance_plot for %s: %s successfully saved', model_name, file_name)
    except AssertionError as err:
        logging.error(
            'Testing feature_importance_plot for %s: %s was not saved properly',
            model_name,
            file_name)
        raise err


def test_classification_report_image(split_dfs, rfc, lrc, mod_tmp_path):
    '''
    test classificaiton report image
    '''
    X_train, X_test, y_train, y_test = split_dfs
    y_train_preds_rf = rfc.predict(X_train)
    y_test_preds_rf = rfc.predict(X_test)
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)
    try:
        cl.classification_report_image(
            y_train,
            y_test,
            y_train_preds_lr,
            y_train_preds_rf,
            y_test_preds_lr,
            y_test_preds_rf,
            mod_tmp_path)
        logging.info('Testing classification_report_image: SUCCESS')
    except Exception as err:
        logging.error('Testing classification_report_image: ERROR')
        raise err


@pytest.mark.parametrize('file_name',
                         ['random_forest_classification.png',
                          'logistic_regression_classification.png'])
def test_classification_report_image_path(file_name, mod_tmp_path):
    '''
    test whether all classification repot images are properly saved
    '''
    try:
        assert os.path.exists(os.path.join(mod_tmp_path, file_name))
        logging.info(
            'Testing classification_image_report: %s successfully saved', file_name)
    except AssertionError as err:
        logging.error(
            'Testing classification_image_report error: %s was not saved properly', file_name)
        raise err


def test_performance_curves(lrc, rfc, split_dfs, mod_tmp_path):
    '''
    test performance curves
    '''
    _, X_test, _, y_test = split_dfs
    try:
        cl.performance_curves(lrc, rfc, X_test, y_test, mod_tmp_path)
        logging.info('Testing performance_curves: SUCCESS')
    except Exception as err:
        logging.info('Testing performance_curves: ERROR')
        raise err


def test_performance_curves_path(mod_tmp_path):
    '''
    test whether all performance curves files are properly saved
    '''
    try:
        assert os.path.exists(os.path.join(mod_tmp_path, 'roc_curves.png'))
        logging.info(
            'Testing performance_curves: roc_curves.png successfully saved')
    except AssertionError as err:
        logging.error(
            'Testing performance_curves: roc_curves.png was not saved properly')
        raise err


if __name__ == "__main__":
    pass
