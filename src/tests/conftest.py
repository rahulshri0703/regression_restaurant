import pytest
from boosting_model.config.core import config
from boosting_model.processing.load_data import load_dataset


# in fixute can turn autouse=True for automatic request
# scope can be 'module' also
@pytest.fixture(scope='session')
def pipeline_input():

    x_train = load_dataset(filename=config.app_config.training_x_file)
    y_train = load_dataset(filename=config.app_config.training_y_file)

    return x_train, y_train


'''

@pytest.fixture()
def raw_training_data():
    # For larger datasets, here we would use a testing sub-sample.
    return load_dataset(file_name=config.app_config.training_data_file)


@pytest.fixture()
def sample_input_data():
    return load_dataset(file_name=config.app_config.test_data_file)
'''
