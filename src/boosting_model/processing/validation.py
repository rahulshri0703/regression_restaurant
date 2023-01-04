import pandas as pd
import numpy as np
import typing as t
from boosting_model.config.core import config
from marshmallow import Schema, fields, ValidationError


class InputSchema(Schema):

    location = fields.Str(allow_none=True)

    name = fields.Str(allow_none=True)

    cuisines = fields.Str(allow_none=False)

    reviews_list = fields.Str(allow_none=False)

    listed_in_type = fields.Str(allow_none=False)
    rest_type = fields.Str(allow_none=False)
    listed_in_city = fields.Str(allow_none=False)

    votes = fields.Integer()
    approx_cost = fields.Integer()
    online_order = fields.Integer()
    book_table = fields.Integer()


def validate_input(*,
                   input_data=pd.DataFrame) -> t.Tuple[pd.DataFrame, t.Optional[dict]]:

    input_data.rename(columns=config.model_config.rename, inplace=True)

    schema = InputSchema(many=True)
    errors = None

    try:
        schema.load(input_data.replace(
            {np.nan: None}).to_dict(orient='record'))

    except ValidationError as exc:
        errors = exc.messages

    return input_data, errors
