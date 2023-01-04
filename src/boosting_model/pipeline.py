import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import yaml
import boosting_model.processing.preprocessor as pp
from boosting_model.config.core import config
from sklearn.base import BaseEstimator, TransformerMixin


def transform_log(x):
    return np.log(x+1)


pipe = Pipeline([

    ('drop feature',
     pp.featuresDrop(variables=config.model_config.feature_to_drop)),
    ('functional',
     pp.sklearnTransformerWrapper(
         variable=config.model_config.numerical,
         transformer=FunctionTransformer(func=transform_log))
     ),
    ('scale',
     pp.sklearnTransformerWrapper(
         variable=config.model_config.numerical,
         transformer=StandardScaler())
     ),
    ('onehot',
     pp.sklearnOne(
         variable=config.model_config.categorical,
         transformer=OneHotEncoder(sparse=False, handle_unknown='ignore'))
     ),
    ('countVector',
     pp.sklearnCountVector(variable=config.model_config.count_vector)

     ),
    ('tfidf',
     pp.reviewsTransform(variable=config.model_config.tfidf)
     )

])


class model_maker(BaseEstimator, TransformerMixin):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def fit(self, X, y):
        x = X.copy()
        y = y.copy()
        self.model.fit(x, y)
        return self

    def transform(self, X):
        x = X.copy()

        yhat = self.model.predict(x)

        return yhat

    def predict(self, X):
        x = X.copy()

        yhat = self.model.predict(x)

        return yhat

    def load_model(self, path):

        self.model.load_model(path)
        return "loaded model"
