#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# First solution for the Mercari Price Suggestion Challenge                   #
#                                                                             #
# This is the entry point of the solution.                                    #
# Developed using Python 3.6.                                                 #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2017-12-09                                                            #
# Version: 1.0.0                                                              #
#                                                                             #
###############################################################################

import os
import time
import numpy as np
import pandas as pd
import pickle
import pyximport; pyximport.install()
import gc

from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error

from sklearn.model_selection import train_test_split, cross_val_score
import lightgbm as lgb

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, StandardScaler
from processing_steps.PreprocessingStep2 import PreprocessingStep2
from processing_steps.SparseTextEncoder import SparseTextEncoder
from wrappers.LGBMWrapper import LGBMWrapper

from load_data import load_data
from files_paths import *

from wordbatch import WordBatch
from wordbatch.extractors import WordBag, WordHash
from wordbatch.models import FTRL, FM_FTRL

from nltk.corpus import stopwords
import re

def rmsle(y, y0):
     assert len(y) == len(y0)
     return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))

# Define helpers for text normalization
stopwords = {x: 1 for x in stopwords.words('english')}
non_alphanums = re.compile(u'[^A-Za-z0-9]+')

def normalize_text(text):
    return u" ".join([x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] if len(x) > 1 and x not in stopwords])
        
# Call to main
if __name__ == "__main__":
    # Start the timer
    start_time = time.time()
    
    # Set the seed of numpy's PRNG
    np.random.seed(2017)

    enable_validation = False  

    features_to_be_scaled_lst = ["reversed_item_condition_id", "category_name_mean_target", "binned_category_name_target_mean", "name_nb_chars", "item_description_nb_chars", 
                                 "nb_chars", "name_nb_tokens", "item_description_nb_tokens", "nb_tokens", "tokens_ratio", "name_nb_words", "item_description_nb_words", "nb_words", 
                                 "name_nb_numbers", "item_description_nb_numbers", "nb_numbers", "name_nb_letters", "item_description_nb_letters", "nb_letters", "name_nb_digits", 
                                 "item_description_nb_digits", "nb_digits", "contains_dust", "contains_gold", "name_contains_lularoe", "name_contains_bundle"]

    columns_to_be_encoded_lst = ["name", "item_description", "category_1", "category_2", "category_3", "brand_name", "item_condition_id"]

    coeffs_dict = {"Men": [-0.01180431, 0.6517549, 0.35800552], "Electronics": [-0.02658065, 0.70573716, 0.32172113], "Women": [-0.04936933, 0.65523746, 0.39512383], 
                   "Home": [-0.04675894, 0.72631163, 0.32075199], "Sports & Outdoors": [-0.11175634, 0.80728063, 0.30652454], "Vintage & Collectibles": [-0.06974298, 0.70257194, 0.36689862], 
                   "Beauty": [-0.00074757, 0.67291934, 0.32724406], "Other": [0.03854487, 0.69971427, 0.25985178], "Kids": [0.01197509, 0.63588165, 0.35228066], 
                   "others": [0.14820591, 0.75975425, 0.10707419], "Handmade": [-0.05670657, 0.68721659, 0.37363133]}

    encoders_lst = [WordBatch(normalize_text, extractor = (WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.5, 1.0], "hash_size": 2 ** 29, "norm": None, "tf": 'binary', "idf": None}), procs = 8),
                    WordBatch(normalize_text, extractor = (WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.0, 1.0], "hash_size": 2 ** 28, "norm": "l2", "tf": 1.0, "idf": None}), procs = 8),
                    CountVectorizer(),
                    CountVectorizer(),
                    CountVectorizer(),
                    LabelBinarizer(sparse_output = True),
                    LabelBinarizer(sparse_output = True)]

    params = {
        "learning_rate": 0.20,
        "application": "regression",
        "max_depth": 10,
        "num_leaves": 60,
        "verbosity": -1,
        "metric": "RMSE",
        "data_random_seed": 1,
        "subsample": 0.9,
        "colsample_bytree": 0.75,
        "nthread": 4
    }

    # Load the data
    train, test, target, truth = load_data(TRAINING_DATA_str, TESTING_DATA_str, enable_validation, "price") # truth is None
    
    print("Train shape: ", train.shape)
    print("Test shape: ", test.shape)
            
    ps = PreprocessingStep2()
    train = ps.fit_transform(train, target)

    sc = StandardScaler()
    train[features_to_be_scaled_lst] = sc.fit_transform(train[features_to_be_scaled_lst])

    ste = SparseTextEncoder(columns_to_be_encoded_lst, encoders_lst)
    X = ste.fit_transform(train, target)

    gc.collect()

    # Take the log of the target
    y = np.log1p(target)

    if enable_validation:
        truth_sr = np.log1p(truth)

    del train, target
    gc.collect()

    FTRL_model = FTRL(alpha = 0.01, beta = 0.1, L1 = 0.00001, L2 = 1.0, D = X.shape[1], iters = 50, inv_link = "identity", threads = 1)
    FTRL_model.fit(X, y)
    print("[{}] Train FTRL completed".format(time.time() - start_time))

    FM_FTRL_model = FM_FTRL(alpha = 0.01, beta = 0.01, L1 = 0.00001, L2 = 0.1, D = X.shape[1], alpha_fm = 0.01, L2_fm = 0.0, init_fm = 0.01, D_fm = 200, e_noise = 0.0001, iters = 17, inv_link = "identity", threads = 4)
    FM_FTRL_model.fit(X, y)
    print("[{}] Train FM FTRL completed".format(time.time() - start_time))

    # Remove features with document frequency <=100
    print("Before removing features with document frequency <=100:", X.shape)
    mask = np.array(np.clip(X.getnnz(axis = 0) - 100, 0, 1), dtype = bool)
    X = X[:, mask]
    print("After removing features with document frequency <=100:", X.shape)
    
    print("[{}] Generating LightGBM data.".format(time.time() - start_time))
    
    train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size = 0.10, random_state = 100) 
    d_train = lgb.Dataset(train_X, label = train_y)
    d_valid = lgb.Dataset(valid_X, label = valid_y)
    watchlist = [d_train, d_valid]

    lgb_model = lgb.train(params, train_set = d_train, num_boost_round = 2200, valid_sets = watchlist, early_stopping_rounds = 1000, verbose_eval = 100) 
    print("[{}] Train LGB completed.".format(time.time() - start_time))

    del X, y, train_X, valid_X, train_y, valid_y, d_train, d_valid, watchlist
    gc.collect()

    # Test part
    submission = pd.DataFrame({"test_id": test.index.values, "price": np.zeros(test.shape[0])})
    
    ## Get the name of the category level
    test["category_1"] = test["category_name"].map(lambda x: x.split("/")[0] if type(x) == str and len(x.split("/")) > 0 else None)

    ## Fill missing values
    test["category_1"].fillna("missing", inplace = True)

    # Group least occuring categories
    tmp = test["category_1"].value_counts()
    tmp = tmp.loc[tmp.index != "missing"].index[:1000]
    test.loc[~test["category_1"].isin(tmp), "category_1"] = "others"

    for category in test["category_1"].unique():
        tmp = ps.transform(test.loc[test["category_1"] == category])
        tmp[features_to_be_scaled_lst] = sc.transform(tmp[features_to_be_scaled_lst])
        X_test = ste.transform(tmp)
    
        gc.collect()

        predsF = FTRL_model.predict(X_test)
        print("predsF shape:", predsF.shape)
        print("predsF NaNs:", np.isnan(predsF).sum())
        print('[{}] Predict FTRL completed'.format(time.time() - start_time))

        predsFM = FM_FTRL_model.predict(X_test)
        print("predsFM shape:", predsFM.shape)
        print("predsFM NaNs:", np.isnan(predsFM).sum())
        print('[{}] Predict FM_FTRL completed'.format(time.time() - start_time))

        X_test = X_test[:, mask]

        predsL = lgb_model.predict(X_test)
        print('[{}] Predict LGB completed.'.format(time.time() - start_time))

        preds = coeffs_dict[category][0] * predsF + coeffs_dict[category][1] * predsFM + coeffs_dict[category][2] * predsL
        submission["price"].loc[test["category_1"] == category] = np.expm1(preds)

        del X_test, tmp
        gc.collect()
        
    # Prevent issues with RMSLE
    submission["price"] = np.abs(submission["price"])

    if enable_validation:
        print("Validation RMSLE for the ensemble:", np.sqrt(mean_squared_error(truth_sr, np.log1p(submission["price"]))))
    else:
        submission.to_csv(OUTPUT_DIR_str + "final_submission_04032018.csv", index = False)
    
    # Stop the timer and print the exectution time
    print("*** Test finished : Executed in:", time.time() - start_time, "seconds ***")