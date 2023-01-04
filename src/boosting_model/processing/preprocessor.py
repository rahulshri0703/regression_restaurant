import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import re


class featuresDrop(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):

        self.variables = variables

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        X = X.copy()

        X = X.drop(columns=self.variables, axis=1)

        return X


class sklearnTransformerWrapper(BaseEstimator, TransformerMixin):

    def __init__(self, variable=None, transformer=None):

        if not isinstance(variable, list):
            self.variable = [variable]
        else:
            self.variable = variable

        self.transformer = transformer

    def fit(self, X, y=None):

        self.transformer.fit(X[self.variable])
        return self

    def transform(self, X):
        X = X.copy()

        X[self.variable] = self.transformer.transform(X[self.variable])
        return X



class sklearnCountVector(BaseEstimator, TransformerMixin):

    def __init__(self, variable=None):

        self.variable = variable[0]
        

        self.transformer = CountVectorizer()

    def fit(self, X, y=None):
        
        X = X.copy()
        X[self.variable] = X[self.variable].apply(lambda x: x.lower())
        X[self.variable] = X[self.variable].apply(lambda x: x.replace(' ',''))
        self.transformer.fit(X[self.variable])
        return self

    def transform(self, X):
        
        X = X.copy()
        X[self.variable] = X[self.variable].apply(lambda x: x.lower())
        X[self.variable] = X[self.variable].apply(lambda x: x.replace(' ',''))
        x = X[self.variable]

        xx = self.transformer.transform(x)
        xx = pd.DataFrame(
             xx.toarray(), columns=self.transformer.
                    get_feature_names_out())
        
        xx.index = X.index
        xx = pd.concat([X,xx],axis=1)
        xx = xx.drop(columns=[self.variable],axis=1)
        return xx
    
    def get_features(self):
        return self.transformer.get_feature_names_out()


        
class sklearnOne(BaseEstimator, TransformerMixin):

    def __init__(self, variable=None, transformer=None):

        if not isinstance(variable, list):
            self.variable = [variable]
        else:
            self.variable = variable

        self.transformer = transformer

    def fit(self, X, y=None):

        self.transformer.fit(X[self.variable])
        return self

    def transform(self, X):
        X = X.copy()

        xx = self.transformer.transform(X[self.variable])
        xx = pd.DataFrame(xx,columns = self.transformer.get_feature_names_out())
        xx.index = X.index
        xx = pd.concat([X,xx],axis=1)
        xx = xx.drop(columns=self.variable,axis=1)
        return xx     
    
class reviewsTransform(BaseEstimator,TransformerMixin):
    
    def __init__(self,variable=None):
        
        self.variable = variable[0]
        self.c = re.compile('[(\\\.,)(?\]\[\\\']|(RATED)|(Rated)|(\d)')
        self.transformer = TfidfVectorizer(max_features=100,stop_words='english')
        self
        
    def fit(self,X,y=None):
        
        X = X.copy()
        
        X[self.variable] = X[self.variable].apply(lambda x:self.c.sub('',x))
        self.transformer.fit(X[self.variable])
        
        return self
    
    def transform(self,X):
        
        X = X.copy()
        
        X[self.variable] = X[self.variable].apply(lambda x:self.c.sub('',x))
        xx = self.transformer.transform(X[self.variable])
        xx = pd.DataFrame(xx.toarray(),columns= self.transformer.get_feature_names_out())
        xx.index = X.index
        xx = pd.concat([X,xx],axis=1)
        xx = xx.drop(columns=[self.variable],axis=1)
        
        return xx
        
            