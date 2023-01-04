import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from boosting_model.config.core import config, DATASET_DIR, TRAINED_MODEL_DIR
from boosting_model import __version__ as _version
import logging
import typing as t
from xgboost import XGBRegressor
import joblib

_logger = logging.getLogger(__name__)


def load_dataset(*, filename: str) -> pd.DataFrame:

    df = pd.read_csv(f"{DATASET_DIR}/{filename}", index_col=0)

    return df


def load_pipeline(*, filename: str) -> Pipeline:

    path = f"{TRAINED_MODEL_DIR}/{filename}"

    pipeline = joblib.load(path)

    return pipeline


def load_model(*, filename: str) -> XGBRegressor:

    path = TRAINED_MODEL_DIR/filename

    model = XGBRegressor()
    model.load_model(path)

    return model


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.

    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config.pipeline_name}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    # remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)
    _logger.info(f"saved pipeline: {save_file_name}")


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.

    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()


def save_model(*, model: XGBRegressor) -> None:
    """Persist the pipeline.

    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config.saved_model_name}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    # remove_old_pipelines(files_to_keep=[save_file_name])
    model.save_model(save_file_name)
    _logger.info(f"saved model: {save_file_name}")
