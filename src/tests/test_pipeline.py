from boosting_model import pipeline
from boosting_model.config.core import config
from boosting_model.processing.validation import validate_input
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline


def test_pipeline_drops_unnecessary_features(pipeline_input):

    x_train, y_train = pipeline_input

    assert config.model_config.feature_to_drop[0] in list(x_train.columns)
    assert config.model_config.feature_to_drop[1] in list(x_train.columns)

    pipeline.pipe[:-1].fit(x_train)

    xx = pipeline.pipe[:-1].transform(x_train)

    assert config.model_config.feature_to_drop[0] in list(x_train.columns)
    assert config.model_config.feature_to_drop[1] in list(x_train.columns)

    assert config.model_config.feature_to_drop[0] not in list(xx.columns)
    assert config.model_config.feature_to_drop[1] not in list(xx.columns)


def test_pipeline_predict_takes_valid_input(pipeline_input):

    x_train, y_train = pipeline_input
    input_data, error = validate_input(input_data=x_train)
    input_data = input_data[config.model_config.features]

    p = pipeline.pipe
    model = pipeline.model_maker(model=XGBRegressor(max_depth=3))

    p.fit(input_data)

    xx = p.transform(input_data)

    model.fit(xx.to_numpy(), y_train)

    final_pipe = make_pipeline(p, model)

    prediction = final_pipe.predict(input_data)

    assert prediction is not None
    assert error is None
