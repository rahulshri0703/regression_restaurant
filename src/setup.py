from pathlib import Path
from setuptools import setup, find_packages

'''
include MANIFEST.in file:

include *.yaml
include *.pkl
recursive-include ./boosting_model/*

include boosting_model/config.yaml
include boosting_model/data/x_train.csv
include boosting_model/data/y_train.csv
include boosting_model/best_model/xgboost
include boosting_model/VERSION
include boosting_model/best_model/final_pipeline2.pkl


'''

# Package meta-data.
NAME = 'boosting-model'
DESCRIPTION = "Gradient boosting regression model from Train In Data."

AUTHOR = "ChristopherGS"
REQUIRES_PYTHON = ">=3.6.0"


# ROOT_DIR = Path(__file__).resolve().parent
# PACKAGE_DIR = ROOT_DIR / 'boosting_model'


setup(
    name='boosting_model',

    description=DESCRIPTION,

    long_description_content_type="text/markdown",
    author=AUTHOR,

    python_requires=REQUIRES_PYTHON,

    packages=find_packages(),
    #package_dir={" ": "boosting_model"},

    package_data={"boosting_model": [
        '*.yaml', 'data/*.csv', 'best_model/xgboost',
        'best_model/final_pipeline2.pkl']},

    include_package_data=True,
    license="BSD-3",

)

# install = pip install path_to_dir where setup.py is located
# or go to dir where setup.py is located: python setup.py install
# python3 setup.py sdist bdist_wheel
