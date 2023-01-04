import pandas as pd
from boosting_model.config.core import config
from boosting_model.processing.load_data import load_dataset, load_pipeline
from boosting_model.processing.validation import validate_input
from boosting_model import __version__ as _version
import typing as t
import logging

_logger = logging.getLogger(__name__)
'''

{'name': 'Sri Banashankari Donne Biriyani',
 'online_order': 1,
 'book_table': 0,
 'votes': 80,
 'location': 'Banaswadi',
 'rest_type': 'Quick Bites',
 'cuisines': 'Biryani',
 'approx_cost': 150,
 'reviews_list': '',
 'listed_in_type': 'Dine-out',
 'listed_in_city': 'Kammanahalli'}

'''

pipeline_final = load_pipeline(filename=config.app_config.pipeline_name)


def make_prediction(*, input_data: t.Union[pd.DataFrame, dict]) -> dict:

    try:
        data = pd.DataFrame(input_data)
    except:
        data = pd.DataFrame(input_data, index=[0])

    validated_data, errors = validate_input(input_data=data)
    results = dict(prediction=None, version=_version, error=errors)

    if not errors:
        prediction = pipeline_final.predict(
            validated_data[config.model_config.features])

        _logger.info(
            f"Making predictions with model version: {_version} "
            f"Predictions: {prediction}"
        )
        results = {"predictions": list(
            prediction), "version": _version, "errors": errors}

    return results
