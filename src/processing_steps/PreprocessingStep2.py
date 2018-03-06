#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# First solution for the Mercari Price Suggestion Challenge                   #
#                                                                             #
# This file contains the code needed for the second preprocessing step.       #
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
import time
import re
import string
from nltk.corpus import stopwords

from sklearn.base import BaseEstimator, TransformerMixin

class PreprocessingStep2(BaseEstimator, TransformerMixin):
    """
    This class defines the first preprocessing step.
    """

    def __init__(self):
        """
        This is the class' constructor.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self.__num_brands = 4000
        self.__num_categories = 1000
        self.__min_df = 10
        self.__max_features_item_description = 50000
                      
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

        # Compute the mean target for each category found in 'category_name'
        tmp_df = X.copy()
        tmp_df["target"] = y
        tmp_df["category_name"].fillna("no_category_name", inplace = True)
        tmp2_df = tmp_df[["category_name", "target"]].groupby("category_name").mean()
        self.__mean_target_by_category_dict = tmp2_df.to_dict()["target"]

        # For categories that have less than 10 samples, group them and replace their mean by the group mean
        categories_count_df = X["category_name"].value_counts()
        least_occuring_categories_lst = categories_count_df.loc[categories_count_df < 10].index.tolist()
        mean_value = tmp_df["target"].loc[tmp_df["category_name"].isin(least_occuring_categories_lst)].mean()

        for category in categories_count_df.loc[categories_count_df < 10].index:
            self.__mean_target_by_category_dict[category] = mean_value

        ## Get the name of the category level
        X["category_1"] = X["category_name"].map(lambda x: x.split("/")[0] if type(x) == str and len(x.split("/")) > 0 else None)

        ## Fill missing values
        X["category_1"].fillna("missing", inplace = True)

        # Group least occuring categories
        tmp = X["category_1"].value_counts()
        tmp = tmp.loc[tmp.index != "missing"].index[:self.__num_categories]
        X.loc[~X["category_1"].isin(tmp), "category_1"] = "others"

        self.__brands_groups = X[["brand_name", "category_1"]].groupby(["category_1", "brand_name"]).size().reset_index()
            
        return self

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

        start_time = time.time()

        # Add indicator for missing brands
        X["is_no_name"] = (X["brand_name"].isnull()).astype(np.int8)

        # Reverse the value of the 'item_condition_id' feature to make increasing value wrt to the price (remember, 1 -> new item, 5 -> broken thing)
        X["reversed_item_condition_id"] = 5 - X["item_condition_id"]

        # Add mean target by category
        X["category_name_mean_target"] = round(X["category_name"].map(self.__mean_target_by_category_dict), 0)
        X["category_name_mean_target"].fillna(self.__mean_target_by_category_dict["no_category_name"], inplace = True)
        
        # Bin mean price by category_name
        X["category_name_target_mean_lt10"] = (X["category_name_mean_target"] < 10).astype(np.int8)
        X["category_name_target_mean_10_20"] = ((X["category_name_mean_target"] >= 10) & (X["category_name_mean_target"] < 20)).astype(np.int8)
        X["category_name_target_mean_lt20"] = (X["category_name_mean_target"] < 20).astype(np.int8)
        X["category_name_target_mean_20_40"] = ((X["category_name_mean_target"] >= 20) & (X["category_name_mean_target"] < 40)).astype(np.int8)
        X["category_name_target_mean_40_50"] = ((X["category_name_mean_target"] >= 40) & (X["category_name_mean_target"] < 50)).astype(np.int8)
        X["category_name_target_mean_50_75"] = ((X["category_name_mean_target"] >= 50) & (X["category_name_mean_target"] < 75)).astype(np.int8)
        X["category_name_target_mean_gt75"] = (X["category_name_mean_target"] >= 75).astype(np.int8)
        X["binned_category_name_target_mean"] = X["category_name_target_mean_lt20"] + 2 * X["category_name_target_mean_20_40"] + 3 * X["category_name_target_mean_40_50"] + 4 * X["category_name_target_mean_50_75"] + 5 * X["category_name_target_mean_gt75"]
        
        # Extract categories hierarchy from 'category_name' feature
        for i in range(3):
            ## Get the name of the category level
            X["category_" + str(i + 1)] = X["category_name"].map(lambda x: x.split("/")[i] if type(x) == str and len(x.split("/")) > i else None)

            ## Fill missing values
            X["category_" + str(i + 1)].fillna("missing", inplace = True)

        X.drop("category_name", axis = 1, inplace = True)
        
        # Fill missing values
        X["brand_name"].fillna(value = "missing", inplace = True)
        X["name"].fillna(value = "missing", inplace = True)
        X["item_description"].fillna(value = "missing", inplace = True)

        # Look for luxury brands: they are expensive
        luxury_brands_set = {"MCM", "MCM Worldwide", "Louis Vuitton", "Burberry", "Burberry London", "Burberry Brit", "HERMES", "Tieks", "Rolex", "Apple", "Gucci", "Valentino", 
                             "Valentino Garavani", "RED Valentino", "Cartier", "Christian Louboutin", "Yves Saint Laurent", "Saint Laurent", "YSL Yves Saint Laurent", "Georgio Armani",
                             "Armani Collezioni", "Emporio Armani"}

        X["is_luxury_brand"] = X["brand_name"].apply(lambda x: int(x in luxury_brands_set))
                
        # Look for some important keywords 
        X["joint_description"] = X["name"].str.cat(X["item_description"], sep = " ").str.lower()
        X["contains_dust"] = X["joint_description"].str.count("dust", flags = re.IGNORECASE)
        X["contains_gold"] = X["joint_description"].str.count("gold", flags = re.IGNORECASE)
        X["name_contains_lularoe"] = X["name"].str.contains("lularoe", case = False).astype(np.int8)
        X["name_contains_bundle"] = X["name"].str.count("bundle", flags = re.IGNORECASE)

        # Group least occuring brands
        tmp = X["brand_name"].value_counts()
        tmp = tmp.loc[tmp.index != "missing"].index[:self.__num_brands]
        X.loc[~X["brand_name"].isin(tmp), "brand_name"] = "missing"

        # Group least occuring categories
        for i in range(3):
            tmp = X["category_" + str(i + 1)].value_counts()
            tmp = tmp.loc[tmp.index != "missing"].index[:self.__num_categories]
            X.loc[~X["category_" + str(i + 1)].isin(tmp), "category_" + str(i + 1)] = "others"

        print("[{}] Cut completed.".format(time.time() - start_time))
        
        # Convert 'item_condition_id' to dummies
        X = pd.concat([X, pd.get_dummies(X["item_condition_id"], prefix = "item_condition_id")], axis = 1)
        print('[{}] Get dummies on `item_condition_id` completed.'.format(time.time() - start_time))

        # Compute some statistics on 'name' and 'item_description'
        X["name_nb_chars"] = X["name"].str.len()
        X["item_description_nb_chars"] = X["item_description"].str.len()
        X["nb_chars"] = X["name_nb_chars"] + X["item_description_nb_chars"]
        
        X["name_nb_tokens"] = X["name"].str.lower().str.split(" ").str.len()
        X["item_description_nb_tokens"] = X["item_description"].str.lower().str.split(" ").str.len()
        X["nb_tokens"] = X["name_nb_tokens"] + X["item_description_nb_tokens"]
        X["tokens_ratio"] = X["name_nb_tokens"] / X["item_description_nb_tokens"]

        X["name_nb_words"] = X["name"].str.count("(\s|^)[a-z]+(\s|$)")
        X["item_description_nb_words"] = X["item_description"].str.count("(\s|^)[a-z]+(\s|$)")
        X["nb_words"] = X["name_nb_words"] + X["item_description_nb_words"]

        X["name_nb_numbers"] = X["name"].str.count("(\s|^)[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?(\s|$)")
        X["item_description_nb_numbers"] = X["item_description"].str.count("(\s|^)[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?(\s|$)")
        X["nb_numbers"] = X["name_nb_numbers"] + X["item_description_nb_numbers"]

        X["name_nb_letters"] = X["name"].str.count("[a-zA-Z]")
        X["item_description_nb_letters"] = X["item_description"].str.count("[a-zA-Z]")
        X["nb_letters"] = X["name_nb_letters"] + X["item_description_nb_letters"]

        X["name_nb_digits"] = X["name"].str.count("[0-9]")
        X["item_description_nb_digits"] = X["item_description"].str.count("[0-9]")
        X["nb_digits"] = X["name_nb_digits"] + X["item_description_nb_digits"]

        for group in ["Beauty", "Electronics", "Handmade", "Home", "Other", "Sports & Outdoors"]:
            brands_lst = self.__brands_groups["brand_name"].loc[self.__brands_groups["category_1"] == group].tolist()
            X["brand_group_" + group] = (X["brand_name"].isin(brands_lst)).astype(np.int8)

        brands_lst = ["Michael Kors", "Louis Vuitton", "Lululemon", "LuLaRoe", "Kendra Scott", "Tory Burch", "Apple", "Kate Spade", "UGG Australia", "Coach", "Gucci", "Rae Dunn", "Tiffany & Co.",
                      "Rock Revival", "Adidas", "Beats", "Burberry", "Christian Louboutin", "David Yurman", "Ray-Ban", "Chanel"]
        X["is_most_expensive"] = (X["brand_name"].isin(brands_lst)).astype(np.int8)

        brands_lst = ["FOREVER 21", "Old Navy", "Carter's", "Elmers", "NYX", "Maybelline", "Disney", "American Eagle", "PopSockets", "Wet n Wild", "Hollister", "Pokemon", "Hot Topic", "Konami", 
                      "Charlotte Russe", "H&M", "e.l.f.", "Bath & Body Works", "Gap"]
        X["is_cheapest"] = (X["brand_name"].isin(brands_lst)).astype(np.int8)

        X.drop(["joint_description"], axis = 1, inplace = True)
                
        return X