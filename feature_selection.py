import pandas as pd
import numpy as np
import xgboost as xgb
import os
from sklearn.feature_selection import SelectFromModel
randomstate = 42

def selectFromXGB(mydir_, filename_, data_, label_):
    '''
    Input:
    mydir_: directory to store the output data
    filename_: output data filename
    data_: original data w/o feature selection
    label_: IS

    Output:
    data_filtered_: data w/ feature selected
    va_sorted: tuple of feature name and its importance sorted in decreasing order
    va_dict: dictionary version for va_sorted tuple
    '''
    xgb_reg = xgb.XGBRegressor(
        objective = "reg:absoluteerror", 
        n_estimators = 200, 
        max_depth = 10, 
        learning_rate = 0.02, 
        verbosity = 1, 
        booster = "gbtree", 
        tree_method = "auto", 
        n_jobs = -1, 
        subsample = 0.9, 
        random_state = randomstate, 
        importance_type = "gain"
    )
    selector = SelectFromModel(
        estimator = xgb_reg, 
        threshold = None, 
        prefit = False, 
        max_features = None, 
        importance_getter = "auto"
    )
    selector = selector.fit(data_, label_)
    s_support = selector.get_support()
    data_filtered_ = data_
    for i in range(len(data_.columns)):
        if s_support[i] == False:
            data_filtered_ = data_filtered_.drop(data_.columns[i], axis = 1)
    print("Variables nums:")
    print(len(data_filtered_.columns))
    print("Selected variables are: ")
    print(data_filtered_.columns)
    data_filtered_.to_excel(mydir_ + filename_)
    xgb_reg.fit(data_, label_)
    def get_feature_importance(estimator_, data_):
        va_dict = {}
        feature_importance_list = estimator_.feature_importances_
        for i in range(len(feature_importance_list)):
            va_dict[data_.columns[i]] = feature_importance_list[i]
        va_sorted = sorted(va_dict.items(), key = lambda item:item[1], reverse = True)
        return va_sorted, va_dict  # tuple list, dict
    va_sorted, va_dict = get_feature_importance(xgb_reg, data_)
    return data_filtered_, va_sorted, va_dict

def pick_top10(data_, va_sorted_):
    '''
    Input:
    data_: original feature df
    va_sorted_: tuple of feature name and its importance sorted in decreasing order

    Output:
    feature_all: design matrix with all features
    feature_filtered: design matrix with selected features
    feature_top10: design matrix with top10 important features
    '''
    filtered_list = [item[0] for item in va_sorted_]
    top10_list = [item[0] for item in va_sorted_[:10]]
    feature_top10 = data_[top10_list]
    feature_all = data_
    feature_filtered = data_[filtered_list]
    return feature_all, feature_filtered, feature_top10

if __name__ == "__main__":
    pass