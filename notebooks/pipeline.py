import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
import preprocessor as pp
from sklearn.pipeline import Pipeline
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)


pipe = Pipeline([

    ('drop feature',
     pp.featuresDrop(variables=config['feature_to_drop'])),
    ('functional',
     pp.sklearnTransformerWrapper(
         variable=config['numerical'],
         transformer=FunctionTransformer(func=lambda x: np.log(x+1)))
     ),
    ('scale',
     pp.sklearnTransformerWrapper(
         variable=config['numerical'],
         transformer=StandardScaler())
     ),
    ('onehot',
     pp.sklearnOne(
         variable=config['categorical'],
         transformer=OneHotEncoder(sparse=False, handle_unknown='ignore'))
     ),
    ('countVector',
     pp.sklearnCountVector(variable=config['count_vector'])

     ),
    ('tfidf',
     pp.reviewsTransform(variable=config['tfidf'])
     )

])
