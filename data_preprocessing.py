# -*- coding: utf-8 -*-
"""
Contains
a preprocessing class for selecting columns that are below a correlation thresholds
a preprocessing class that drops unwanted columns
"""
from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
# data preprocessing script

class SelectedColumns(TransformerMixin, BaseEstimator):
    """
    performs data preprocessing by selecting columns that do not meet the threshold
    correlation thresholds. ie ignores/removes columns that have correlation coefficient
    values above the specified upper limit threshold and below the specified lower limit
    correlation coefficient threshold.
    """

    def __init__(self, upper_limit, lower_limit=None, target=None, collist=[]):
        self.limit = upper_limit
        self.lower = lower_limit
        self.cols = collist
        self.target = target

    def fit(self, X):
        corr_df = X.corr()
        columns = np.full(corr_df.shape[0], True, dtype=bool)
        for i in range(corr_df.shape[0]):
            for j in range(i+1, corr_df.shape[0]):
                if corr_df.iloc[i, j] > self.limit or corr_df.iloc[i,j] < -self.lower:
                    if columns[j]:
                        columns[j] = False
        if self.target:
            selected_cols = X.drop(self.target, axis=1).columns[columns]
        else:
            selected_cols = X.columns[columns]
        self.cols.extend(selected_cols)
        if self.target: self.cols.append(self.target)
        return self
    def transform(self, X):
        return X[self.cols]

class Drop_cols(TransformerMixin, BaseEstimator):
    """
    drops specified columns
    """
    def __init__(self, cols_to_drop):
        self.cols_drop = cols_to_drop
    def fit(self, X):
        return self
    def transform(self, X):
        return X.drop(self.cols_drop, axis=1)
