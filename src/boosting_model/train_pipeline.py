from boosting_model.processing.load_data import load_dataset, save_pipeline
#from boosting_model.processing.validation import validate_input
from boosting_model import pipeline
from boosting_model.config.core import config
import joblib
import logging

import logging


_logger = logging.getLogger(__name__)


def run_training() -> None:
    """Train the model."""

    # read training data
    x = load_dataset(filename=config.app_config.training_x_file)
    x.rename(columns=config.model_config.rename, inplace=True)

    pipeline.pipe.fit(x.iloc[:10])

   # _logger.warning(f"saving model version: {_version}")
    save_pipeline(pipeline_to_persist=pipeline.pipe)


if __name__ == "__main__":
    run_training()
