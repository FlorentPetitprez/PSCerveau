# We need to reduce the number of features to very few relevant voxels
# f_classif works with one target, but not several

from sklearn.base import BaseEstimator
from sklearn.feature_selection import f_classif
import numpy as np


class MultiSelectKBest(BaseEstimator):

    def __init__(self, classif_func=f_classif, k=10,
                 pooling_function=np.min):
        self.classif_func = classif_func
        self.k = k
        self.pooling_function = pooling_function

    def fit(self, X, y):
        scores = []
        if y.ndim == 1:
            y = y[:, np.newaxis]

        for yy in y.T:
            score, _ = self.classif_func(X, yy)
            scores.append(score)

        self.scores = self.pooling_function(np.array(scores), axis=0)

        return self

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def get_support(self):
        if self.scores is not None:
            res = np.zeros(self.scores.shape, dtype=np.bool)
            res[self.scores.argsort()[::-1][:self.k]] = True
            return res

    def transform(self, X):
        supp = self.get_support()
        return X[:, supp]
