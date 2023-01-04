

'''
y range: {'min': 1.8, 'max': 4.9}

'''


from boosting_model.predict import make_prediction
from boosting_model.config.core import config

from sklearn.metrics import mean_squared_error
import numpy as np

#from regression_model.predict import make_prediction as alt_make_prediction


def test_prediction_quality_against_benchmark(pipeline_input):
    # Given
    x_train, y_train = pipeline_input

    # Generate rough benchmarks (you would tweak depending on your model)
    benchmark_flexibility = 0.05
    # setting ndigits to -4 will round the value to the nearest 10,000 i.e. 210,000
    benchmark_lower_boundary = (
        1.8 - benchmark_flexibility
    )
    benchmark_upper_boundary = (
        5 + benchmark_flexibility
    )

    # When
    subject = make_prediction(input_data=x_train.iloc[0].to_dict())

    # Then
    assert subject is not None
    prediction = subject.get("predictions")[0]
    assert isinstance(prediction, np.float32)
    assert prediction > benchmark_lower_boundary
    assert prediction < benchmark_upper_boundary


# def test_prediction_quality_against_another_model(raw_training_data, sample_input_data):
#     # Given
#     input_df = raw_training_data.drop(config.model_config.target, axis=1)
#     output_df = raw_training_data[config.model_config.target]
#     current_predictions = make_prediction(input_data=input_df)

#     # the older model has these variable names reversed
#     input_df.rename(
#         columns={
#             "FirstFlrSF": "1stFlrSF",
#             "SecondFlrSF": "2ndFlrSF",
#             "ThreeSsnPortch": "3SsnPorch",
#         },
#         inplace=True,
#     )
#     alternative_predictions = alt_make_prediction(input_data=input_df)

#     # When
#     current_mse = mean_squared_error(
#         y_true=output_df.values, y_pred=current_predictions["predictions"]
#     )

#     alternative_mse = mean_squared_error(
#         y_true=output_df.values, y_pred=alternative_predictions["predictions"]
#     )

#     # Then
#     assert current_mse < alternative_mse
