#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# First solution for the Mercari Price Suggestion Challenge                   #
#                                                                             #
# This file contains the code needed for creating a transformer that encodes  #
# text using bag of words or TF-IDF representation. It is compatible with the #
# Scikit-Learn framework and uses sparse matrices for reducing memory         #
# consumption.                                                                #
# Developed using Python 3.6.                                                 #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2017-12-31                                                            #
# Version: 1.0.0                                                              #
#                                                                             #
###############################################################################

import numpy as np
import pandas as pd
import time

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from wordbatch import WordBatch
from wordbatch.extractors import WordBag, WordHash
from wordbatch.models import FTRL, FM_FTRL

from sklearn.base import BaseEstimator, TransformerMixin

from scipy.sparse import csr_matrix, hstack

class SparseTextEncoder(BaseEstimator, TransformerMixin):
    """
    This class defines a Scikit-Learn transformer that implements a text encoder using bag of words or TF-IDF representation.
    """

    def __init__(self, columns_names_lst, encoders_lst):
        """
        This is the class' constructor.

        Parameters
        ----------
        columns_names_lst : list
                This contains the names of the columns we want to transform.

        encoders_lst : list
                This contains the encoders chosen for each column of the columns_names_lst list.
                                
        Returns
        -------
        None
        """

        self.__columns_names_lst = columns_names_lst
        self.__min_df = 10
        self.__encoders_lst = encoders_lst
        self.__encoders_masks_lst = [None for i in encoders_lst]

    def fit(self, X, y = None):
        """
        This method is called to fit the transformer on the training data.

        Parameters
        ----------
        X : pd.DataFrame
                This is a data frame containing the data used to fit the transformer.

        y : pd.Series (optional)
                This is the target associated with the X data.

        Returns
        -------
        None
        """

        print("Method not implemented! Please call fit_transform() instead.")

        return self

    def fit_transform(self, X, y = None):
        """
        This method is called to fit the transformer on the training data and then transform the associated data.

        Parameters
        ----------
        X : pd.DataFrame
                This is a data frame containing the data used to fit the transformer.

        y : pd.Series (optional)
                This is the target associated with the X data.

        Returns
        -------
        None
        """

        start_time = time.time()

        standard_columns_lst = list(set(X.columns.tolist()) - set(self.__columns_names_lst))
        arrays_lst = [csr_matrix(X[standard_columns_lst].values)]
        columns_names_lst = [c for c in standard_columns_lst]

        print("Regular columns shape:", arrays_lst[0].shape)
        print("Regular columns names length:", len(columns_names_lst))

        nnz_threshold = 2

        for idx, column in enumerate(self.__columns_names_lst):
            X[column].fillna("NaN", inplace = True)
            
            if type(self.__encoders_lst[idx]) == CountVectorizer or type(self.__encoders_lst[idx]) == TfidfVectorizer:
                encoder_features_csr = self.__encoders_lst[idx].fit_transform(X[column])
                self.__encoders_masks_lst[idx] = np.array(np.clip(encoder_features_csr.getnnz(axis = 0) - nnz_threshold, 0, 1), dtype = bool)
                encoder_features_csr = encoder_features_csr[:, self.__encoders_masks_lst[idx]]

                encoder_columns_names_lst = [column + "_" + w for w in self.__encoders_lst[idx].get_feature_names()]
            elif type(self.__encoders_lst[idx]) == LabelBinarizer:
                encoder_features_csr = self.__encoders_lst[idx].fit_transform(X[column])
                self.__encoders_masks_lst[idx] = np.array(np.clip(encoder_features_csr.getnnz(axis = 0) - nnz_threshold, 0, 1), dtype = bool)
                encoder_features_csr = encoder_features_csr[:, self.__encoders_masks_lst[idx]]

                encoder_columns_names_lst = [column + "_LabelBinarizer_" + str(w + 1) for w in range(encoder_features_csr.shape[1])]
            elif type(self.__encoders_lst[idx]) == WordBatch:
                self.__encoders_lst[idx].dictionary_freeze = True
                encoder_features_csr = self.__encoders_lst[idx].fit_transform(X[column])
                self.__encoders_masks_lst[idx] = np.array(np.clip(encoder_features_csr.getnnz(axis = 0) - nnz_threshold, 0, 1), dtype = bool)
                encoder_features_csr = encoder_features_csr[:, self.__encoders_masks_lst[idx]]

                encoder_columns_names_lst = [column + "_WordBatch_" + str(w + 1) for w in range(encoder_features_csr.shape[1])]

            print(column, ": shape:", encoder_features_csr.shape)
            print(column, ": columns names length:", len(encoder_columns_names_lst))

            arrays_lst.append(encoder_features_csr)
            columns_names_lst.extend(encoder_columns_names_lst)

        sparse_merge = hstack(arrays_lst).tocsr()

        print("sparse_merge: shape:", sparse_merge.shape)
        print("*** Sparse text encoder: transform in ", round(time.time() - start_time, 3), "seconds ***")

        return sparse_merge

    def transform(self, X):
        """
        This method is called to fit the transformer on the training data.

        Parameters
        ----------
        X : pd.DataFrame
                This is a data frame containing the data that will be transformed.
                
        Returns
        -------
        X : pd.DataFrame
                This is a data frame containing the data that will be transformed.
        """

        standard_columns_lst = list(set(X.columns.tolist()) - set(self.__columns_names_lst))
        arrays_lst = [csr_matrix(X[standard_columns_lst].values)]
        
        for idx, column in enumerate(self.__columns_names_lst):
            X[column].fillna("NaN", inplace = True)

            encoder_features_csr = self.__encoders_lst[idx].transform(X[column])
            encoder_features_csr = encoder_features_csr[:, self.__encoders_masks_lst[idx]]

            arrays_lst.append(encoder_features_csr)
            
        sparse_merge = hstack(arrays_lst).tocsr()
        
        return sparse_merge
