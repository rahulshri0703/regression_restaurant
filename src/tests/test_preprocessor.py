from boosting_model.processing import preprocessor
from boosting_model.config.core import config


def test_transformer_drop_feature(pipeline_input):
    x_train, y_train = pipeline_input

    assert config.model_config.feature_to_drop[0] in list(x_train.columns)
    assert config.model_config.feature_to_drop[1] in list(x_train.columns)

    transformer = preprocessor.featuresDrop(
        variables=config.model_config.feature_to_drop)
    xx = transformer.transform(x_train)

    assert config.model_config.feature_to_drop[0] not in list(xx.columns)
    assert config.model_config.feature_to_drop[1] not in list(xx.columns)


def test_count_vectorizer(pipeline_input):
    x_train, y_train = pipeline_input

    assert config.model_config.count_vector[0] in list(x_train.columns)

    transformer = preprocessor.sklearnCountVector(
        variable=config.model_config.count_vector)

    x = transformer.fit_transform(x_train)
    assert x is not None
