#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# First solution for the Mercari Price Suggestion Challenge                   #
#                                                                             #
# This file provides everything needed to load the data.                      #
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
from sklearn.model_selection import train_test_split

def load_data(training_set_path_str, testing_set_path_str, enable_validation, target_name_str, validation_type = "random"):
    """
    This function is a wrapper for the loading of the data.

    Parameters
    ----------
    training_set_path_str : string
            A string containing the path of the training set file.

    testing_set_path_str : string
            A string containing the path of the testing set file.

    enable_validation : boolean
            A boolean indicating if we are validating our model or if we are creating a submission for Kaggle.

    target_name_str : string
            A string indicating the target column name.

    validation_type : string
            A string indicating either if the validation split is random or time based.

    Returns
    -------
    training_set_df : pd.DataFrame
            A pandas DataFrame containing the training set.

    testing_set_df : pd.DataFrame
            A pandas DataFrame containing the testing set.
            
    target_sr : pd.Series
            The target values for the training part.

    truth_sr : pd.Series
            The target values for the validation part.
    """

    # Load the data
    print("Loading the data...")
    training_set_df = pd.read_csv(training_set_path_str, sep = "\t", index_col = 0)
    testing_set_df = pd.read_csv(testing_set_path_str, sep = "\t", index_col = 0)

    # Remove rows from training set where the price is zero
    print("Removing rows from training set where the target is zero...")
    training_set_df = training_set_df.loc[training_set_df[target_name_str] > 0]
    
    # Generate a validation set if enable_validation is True
    if enable_validation:
        print("Generating validation set...")
        test_size_ratio = 0.2
        if type == "random":
            X_train, X_test = train_test_split(training_set_df, test_size = test_size_ratio, random_state = 2017)
            training_set_df = pd.DataFrame(X_train, columns = training_set_df.columns)
            testing_set_df = pd.DataFrame(X_test, columns = training_set_df.columns)
        else:
            split_threshold = int(training_set_df.shape[0] * (1 - test_size_ratio))
            testing_set_df = training_set_df.iloc[split_threshold:]
            training_set_df = training_set_df.iloc[0:split_threshold]
    
        # Extract truth / target
        truth_sr = testing_set_df[target_name_str]
        testing_set_df = testing_set_df.drop(target_name_str, axis = 1)

        # Reindex DataFrames
        training_set_df = training_set_df.reset_index(drop = True)
        testing_set_df = testing_set_df.reset_index(drop = True)
        truth_sr = truth_sr.reset_index(drop = True)

        print("Generating validation set... done")
    else:
        truth_sr = None

    # Extract target for training set
    target_sr = training_set_df[target_name_str]
    training_set_df = training_set_df.drop(target_name_str, axis = 1)

    print("Loading data... done")

    return training_set_df, testing_set_df, target_sr, truth_sr