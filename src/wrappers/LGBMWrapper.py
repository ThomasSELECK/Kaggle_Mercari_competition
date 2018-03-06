#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# First solution for the Mercari Price Suggestion Challenge                   #
#                                                                             #
# This file contains the code needed to provide a Scikit-Learn like wrapper   #
# to LightGBM.                                                                #
# Developed using Python 3.6.                                                 #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2017-12-09                                                            #
# Version: 1.0.0                                                              #
#                                                                             #
###############################################################################

import numpy as np
import pandas as pd
import lightgbm as lgb
from operator import itemgetter
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
import pickle

class LGBMWrapper(BaseEstimator):
    """
    The purpose of this class is to provide a wrapper for a LightGBM model, with cross-validation for finding the best number of rounds.
    """
    
    def __init__(self, params, early_stopping_rounds, custom_eval_function = None, maximize = True, nrounds = -1, random_state = 0, test_size = 0.1):
        """
        Class' constructor

        Parameters
        ----------
        params : dictionary
                This contains the parameters of the XGBoost model.

        early_stopping_rounds : integer
                This indicates the number of rounds to keep before stopping training when the score doesn't increase. If negative, disable this functionality.

        verbose_eval : positive integer
                This indicates the frequency of scores printing. E.g. 50 means one score printed every 50 rounds.

        custom_eval_function : function
                This is a function XGBoost will use as loss function.

        maximize : boolean
                Indicates if the function customEvalFunction must be maximized or minimized. Not used when customEvalFunction is None.

        nrounds : integer
                Number of rounds for XGBoost training.

        random_state : zero or positive integer
                Seed used by XGBoost to ensure reproductibility.

        test_size : float between 0 and 1.
                This indicates the size of the test set.
                
        Returns
        -------
        None
        """
        
        # Class' attributes
        self.__params = params
        self.__early_stopping_rounds = early_stopping_rounds
        self.__custom_eval_function = custom_eval_function
        self.__maximize = maximize
        self.__nrounds = nrounds
        self.__random_state = random_state
        self.__test_size = test_size
        self.__params["seed"] = self.__random_state
        self.__lgb_model = None
        self.__model_name = "LightGBM"

    def get_model_name(self):
        return self.__model_name

    def get_LGB_model(self):
        return self.__lgb_model

    def get_nb_rounds(self):
        return self.__nrounds

    def get_params(self):
        return self.__params

    def PrintFeatureImportance(self, maxCount = -1):
        importance = self.__lgb_model.get_score(importance_type = "gain")
        importance = sorted(importance.items(), key = itemgetter(1), reverse = True)

        for i, item in enumerate(importance):
            if i < maxCount or maxCount < 0:
                print(i + 1, "/", len(importance), ":", item[0])

    def plot_features_importance(self, importance_type = "gain"):
        lgb.plot_importance(self.__lgb_model, importance_type = importance_type)
        plt.show()

    def get_features_importance(self, importance_type = "gain"):
        importance = self.__lgb_model.feature_importance(importance_type = "gain")
        features_names = self.__lgb_model.feature_name()

        return pd.DataFrame({"feature": features_names, "importance": importance}).sort_values(by = "importance", ascending = False).reset_index(drop = True)

    def fit(self, X, y):
        """
        This method trains the LightGBM model.

        Parameters
        ----------
        X : Pandas DataFrame
                This is the training data.

        y : Pandas Series
                This is the target related to the training data.
                
        Returns
        -------
        None
        """

        print("Preparing data for LightGBM...")
                       
        if self.__nrounds == -1:
            dtrain = lgb.Dataset(X, label = y)
            watchlist = [dtrain]

            print("    Cross-validating LightGBM classifier with seed: " + str(self.__random_state) + "...")
            cv_output = lgb.cv(self.__params, dtrain, num_boost_round = 10000, early_stopping_rounds = self.__early_stopping_rounds, show_stdv = True)

            self.__nrounds = cv_output.shape[0]

            print("    Training LightGBM classifier with seed: " + str(self.__random_state) + " and num rounds = " + str(self.__nrounds) + "...")
            self.__lgb_model = lgb.train(self.__params, dtrain, self.__nrounds, watchlist, early_stopping_rounds = self.__early_stopping_rounds, verbose_eval = 100)
        else:
            X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size = self.__test_size, random_state = self.__random_state)
            
            print("Training LightGBM...")
            dtrain = lgb.Dataset(X_train, label = y_train)
            dvalid = lgb.Dataset(X_eval, label = y_eval)
            watchlist = [dtrain, dvalid]

            self.__lgb_model = lgb.train(self.__params, dtrain, self.__nrounds, watchlist, early_stopping_rounds = self.__early_stopping_rounds, verbose_eval = 100)

        return self

    def predict(self, X):
        """
        This method makes predictions using the previously trained model.

        Parameters
        ----------
        X : Pandas DataFrame
                This is the testing data we want to make predictions on.
                
        Returns
        -------
        predictions_npa : numpy array
                Numpy array containing predictions for each sample of the testing set.
        """

        # Sanity checks
        if self.__lgb_model is None:
            raise ValueError("You MUST train the XGBoost model using fit() before attempting to do predictions!")

        print("Predicting outcome for testing set...")
        predictions_npa = self.__lgb_model.predict(X)

        return predictions_npa