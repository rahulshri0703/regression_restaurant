from pathlib import Path
from boosting_model.config.core import create_and_validate_config, fetch_config_from_yaml


import pytest
from pydantic import ValidationError


TEST_CONFIG_FILE = '''

package_name: boosting_model

training_x_file: x_train.csv
training_y_file: y_train.csv

test_x_file: x_test.csv
test_y_file: y_test.csv

saved_model_name: xgboost

pipeline_name: final_pipeline2.pkl

feature_to_drop: 
    - 'location'
    - 'name'
    
count_vector: 
    - 'cuisines'
    
tfidf:
    - 'reviews_list'
    
categorical:
    - 'listed_in_type'
    - 'rest_type'
    - 'listed_in_city'
    
numerical:
    - 'votes'
    - 'approx_cost'
ordinal: 
    - 'online_order'
    - 'book_table'
target:
    - 'rate'

rename:
    approx_cost(for two people) : 'approx_cost'
    listed_in(type) : 'listed_in_type'
    listed_in(city) : 'listed_in_city'

features:
    - 'cuisines'
    - 'reviews_list'
    - 'listed_in_type'
    - 'rest_type'
    - 'listed_in_city'
    - 'votes'
    - 'approx_cost'
    - 'online_order'
    - 'book_table'
    - 'approx_cost'
    - 'listed_in_type'
    - 'listed_in_city'

'''

INVALID_TEST_CONFIG_FILE = '''

package_name: boosting_model

training_x_file: x_train.csv
training_y_file: y_train.csv

test_x_file: x_test.csv
test_y_file: y_test.csv

saved_model_name: xgboost

pipeline_name: final_pipeline2.pkl

feature_to_drop: 
    - 'location'
    - 'name'
    
count_vector: 
    - 'cuisines'
    
tfidf:
    - 'reviews_list'
    
categorical:
    - 'listed_in_type'
    - 'rest_type'
    - 'listed_in_city'
    
numerical:
    - 'votes'
    - 'approx_cost'
ordinal: 
    - 'online_order'
    - 'book_table'
target:
    - 'rate'


features:
    - 'cuisines'
    - 'reviews_list'
    - 'listed_in_type'
    - 'rest_type'
    - 'listed_in_city'
    - 'votes'
    - 'approx_cost'
    - 'online_order'
    - 'book_table'
    - 'approx_cost'
    - 'listed_in_type'
    - 'listed_in_city'

'''
# tmpdir is used by pytest to test


def test_fetch_config_structure(tmpdir):

    config_dir = Path(tmpdir)
    config_1 = config_dir / 'sample_config.yaml'
    config_1.write_text(TEST_CONFIG_FILE)
    parsed_config = fetch_config_from_yaml(cfg_path=config_1)

    config = create_and_validate_config(parsed_config=parsed_config)

    assert config.app_config
    assert config.model_config


def test_config_validation_raise_error_for_invalid_config(tmpdir):

    config_dir = Path(tmpdir)
    config_1 = config_dir / 'sample_config.yaml'

    config_1.write_text(INVALID_TEST_CONFIG_FILE)
    parsed_config = fetch_config_from_yaml(cfg_path=config_1)

    with pytest.raises(ValidationError) as excinfo:
        create_and_validate_config(parsed_config=parsed_config)

    assert 'value_error.missing' in str(excinfo.value)


def test_missing_config_field_raises_validation_error(tmpdir):
    # Given
    # We make use of the pytest built-in tmpdir fixture
    configs_dir = Path(tmpdir)
    config_1 = configs_dir / "sample_config.yml"
    TEST_CONFIG_TEXT = """package_name: boosting_model"""
    config_1.write_text(TEST_CONFIG_TEXT)
    parsed_config = fetch_config_from_yaml(cfg_path=config_1)

    # When
    with pytest.raises(ValidationError) as excinfo:
        create_and_validate_config(parsed_config=parsed_config)

    # Then
    assert "field required" in str(excinfo.value)
    assert "pipeline_name" in str(excinfo.value)
