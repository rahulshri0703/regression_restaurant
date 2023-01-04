from pathlib import Path
import typing as t
from pydantic import BaseModel, validator
from strictyaml import load, YAML


# os.path.dirname(__file__)
PACKAGE_DIR = Path(__file__).resolve().parent.parent
ROOT = PACKAGE_DIR.parent
CONFIG_FILE_PATH = PACKAGE_DIR / 'config.yaml'
TRAINED_MODEL_DIR = PACKAGE_DIR / 'best_model'
DATASET_DIR = PACKAGE_DIR / 'data'


class AppConfig(BaseModel):

    package_name: str
    pipeline_name: str

    training_x_file: str
    training_y_file: str

    test_x_file: str
    test_y_file: str

    saved_model_name: str


class ModelConfig(BaseModel):

    feature_to_drop: t.Sequence[str]
    count_vector: t.Sequence[str]
    tfidf: t.Sequence[str]
    categorical: t.Sequence[str]
    numerical: t.Sequence[str]
    target: t.Sequence[str]
    ordinal: t.Sequence[str]
    rename: t.Dict
    features: t.Sequence[str]


'''
t.Dict for dictionary
t.Sequence[float]
float
int

if we have a loss funtion in config.yaml:

    allowed_loss_functions: t.Tuple[str, ...]
    loss: str

    @validator("loss")
    def allowed_loss_function(cls, value, values):
        """
        Loss function to be optimized.

        `ls` refers to least squares regression.
        `lad` (least absolute deviation)
        `huber` is a combination of the two.
        `quantile` allows quantile regression.

        Following the research phase, loss is restricted to
        `ls` and `huber` for this model.
        """

        allowed_loss_functions = values.get("allowed_loss_functions")
        if value in allowed_loss_functions:
            return value
        raise ValueError(
            f"the loss parameter specified: {value}, "
            f"is not in the allowed set: {allowed_loss_functions}"
        )
'''


class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig
    model_config: ModelConfig


def find_config_file() -> Path:
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:

    if not cfg_path:
        cfg_path = find_config_file()

    with open(cfg_path, 'r') as f:
        parsed_config = load(f.read())
        return parsed_config

    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:

    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_config=AppConfig(**parsed_config.data),
        model_config=ModelConfig(**parsed_config.data),
    )

    return _config


config = create_and_validate_config()
