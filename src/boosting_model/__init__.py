from boosting_model.config import core
#from boosting_model.processing import load_data, validation, preprocessor, error

import logging
from .pipeline import pipe
from boosting_model.config.core import config, PACKAGE_DIR

# It is strongly advised that you do not add any handlers other than
# NullHandler to your library’s loggers. This is because the configuration
# of handlers is the prerogative of the application developer who uses your
# library. The application developer knows their target audience and what
# handlers are most appropriate for their application: if you add handlers
# ‘under the hood’, you might well interfere with their ability to carry out
# unit tests and deliver logs which suit their requirements.
# https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library

logging.getLogger(config.app_config.package_name).addHandler(
    logging.NullHandler())


with open(PACKAGE_DIR / "VERSION") as version_file:
    __version__ = version_file.read().strip()
