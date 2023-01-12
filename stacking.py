import pandas as pd
import numpy as np
import scipy.stats as stats
import sklearn
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from deepforest import CascadeForestClassifier, CascadeForestRegressor
import lightgbm as lgb
import time 
import os
import re
randomstate = 42

def trainingCV_alt(estimator_, data_, label_, n_splits_):
    # Metrics: AUC
    # estimator_ should have fit / predict methods
    # data_.index = np.linspace(0, len(data_) - 1, len(data_), dtype = int)
    # label_.index = np.linspace(0, len(label_) - 1, len(label_), dtype = int)
    if type(estimator_) != CascadeForestRegressor or type(estimator_) != CascadeForestClassifier:
        randomstate = 42
        folds = KFold(n_splits = n_splits_, shuffle = True, random_state = randomstate)
        AUC_list = []
        index_list = []
        temp_original_value = []
        temp_testing_value = []
        for (train_index, test_index) in folds.split(data_):
            data_train, label_train = data_.iloc[train_index,:], label_[train_index]
            data_test, label_test = data_.iloc[test_index,:], label_[test_index] 
            estimator_.fit(data_train, label_train)
            temp_original_value.append(label_test)
            temp_testing_value.append(estimator_.predict(data_test))
            index_list.extend(list(data_test.index))
            AUC_list.append(roc_auc_score(label_test, estimator_.predict(data_test)))
        mean_AUC = np.mean(AUC_list)
        original_value = []
        testing_value = []
        for i in range(len(temp_original_value)):
            temp_original_value[i] = list(temp_original_value[i])
            temp_testing_value[i] = list(temp_testing_value[i])
            for j in range(len(temp_original_value[i])):
                original_value.append(temp_original_value[i][j])
                testing_value.append(temp_testing_value[i][j])
        return estimator_, mean_AUC, pd.Series(original_value, index = index_list), pd.Series(testing_value, index = index_list)
    else:
        data_.index = np.linspace(0, len(data_) - 1, len(data_), dtype = int)
        label_.index = np.linspace(0, len(label_) - 1, len(label_), dtype = int)
        randomstate = 42
        folds = KFold(n_splits = n_splits_, shuffle = True, random_state = randomstate)
        AUC_list = []
        temp_original_value = []
        temp_testing_value = []
        for (train_index, test_index) in folds.split(data_):
            data_train, label_train = data_.iloc[train_index,:], label_[train_index]
            data_test, label_test = data_.iloc[test_index,:], label_[test_index] 
            estimator_.fit(data_train, label_train)
            temp_original_value.append(label_test)
            temp_testing_value.append(estimator_.predict(data_test))
            AUC_list.append(roc_auc_score(label_test, estimator_.predict(data_test)))
        mean_AUC = np.mean(AUC_list)
        original_value = []
        testing_value = []
        for i in range(len(temp_original_value)):
            temp_original_value[i] = list(temp_original_value[i])
            temp_testing_value[i] = list(temp_testing_value[i])
            for j in range(len(temp_original_value[i])):
                original_value.append(temp_original_value[i][j])
                testing_value.append(temp_testing_value[i][j])
        return estimator_, mean_AUC, original_value, testing_value


def lightgbmCV(estimator_, data_, label_, n_splits_):
    # Metrics: MAE
    # estimator_ should have fit / predict methods
    data_.index = np.linspace(0, len(data_) - 1, len(data_), dtype = int)
    label_.index = np.linspace(0, len(label_) - 1, len(label_), dtype = int)
    randomstate = 42
    folds = KFold(n_splits = n_splits_, shuffle = True, random_state = randomstate)
    MAE_list = []
    booster = []
    temp_original_value = []
    temp_testing_value = []
    i = 0
    for (train_index, test_index) in folds.split(data_):
        data_train, label_train = data_.iloc[train_index,:], label_[train_index]
        data_test, label_test = data_.iloc[test_index,:], label_[test_index] 
        if i == 0:
            estimator_.fit(data_train, label_train)
            temp_original_value.append(label_test)
            temp_testing_value.append(estimator_.predict(data_test))
        else:
            estimator_.fit(data_train, label_train, init_model = booster[i - 1])
            temp_original_value.append(label_test)
            temp_testing_value.append(estimator_.predict(data_test))
        booster.append(estimator_)
        MAE_list.append(mean_absolute_error(label_test, estimator_.predict(data_test)))
    mean_MAE = np.mean(MAE_list)
    original_value = []
    testing_value = []
    for i in range(len(temp_original_value)):
        temp_original_value[i] = list(temp_original_value[i])
        temp_testing_value[i] = list(temp_testing_value[i])
        for j in range(len(temp_original_value[i])):
            original_value.append(temp_original_value[i][j])
            testing_value.append(temp_testing_value[i][j])
    return estimator_, MAE_list, mean_MAE, original_value, testing_value

def pred_oof(estimator_, data_train_, label_train_, data_test_):
    oof_train = np.zeros(len(data_train_))
    oof_test = np.zeros(len(data_test_))
    oof_test_folds = pd.DataFrame(data = np.zeros((len(data_test_), n)), columns = list(np.linspace(0, n - 1, n, dtype = int)))
    folds = KFold(n_splits = 5, shuffle = True, random_state = randomstate)
    # df21 cannot use kfolds
    # df21 uses np.array
    if type(estimator_) == CascadeForestRegressor or type(estimator_) == CascadeForestClassifier:
        ### change data_train_ sequence
        folds = KFold(n_splits = 5, shuffle = True, random_state = randomstate)
        temp_list = []
        for train_index, test_index in folds.split(data_train_):
            for item in test_index:
                temp_list.append(item)

        temp_data_train_ = pd.DataFrame(data = None)
        for i in range(len(data_train_)):
            temp_data_train_ = temp_data_train_.append([data_train_.iloc[temp_list[i]]], ignore_index = True)
        
        temp_label_train_ = pd.Series(data = None, dtype = float)
        for i in range(len(label_train_)):
            temp_label_train_ = temp_label_train_.append(pd.Series(label_train_[temp_list[i]]), ignore_index = True)
        
        estimator_.fit(np.array(temp_data_train_), np.array(temp_label_train_))
        oof_train = estimator_.predict(temp_data_train_)
        oof_test = estimator_.predict(data_test_)
        return oof_train, oof_test
    else:
        i = 0
        for (train_index, test_index) in folds.split(data_train_):
            data_tr = data_train_.iloc[train_index,:]
            label_tr = label_train_[train_index]
            label_tr = label_tr.astype(int)
            data_te = data_train_.iloc[test_index,:]
            # print(data_te.shape)

            # label_tr should be integer series
            estimator_.fit(data_tr, label_tr)

            oof_train[test_index] = estimator_.predict(data_te)
            oof_test_folds.loc[:,i] = estimator_.predict(data_test_)
            i += 1
        oof_test = oof_test_folds.mean(axis = 1)  
        return pd.Series(oof_train), pd.Series(oof_test)

def train_stacking_models(subsetname_):
    subsetname = subsetname_
    new_name = re.sub("_stacking", "", subsetname, flags = 0)
    # models = ["rf", "lgb", "df21", "dnn"]
    models = ["rf", "lgb", "df21", "dnn", "knn"]

    for model in models:  # models
        # print('-------------------------------------------------------------------------------------------------------------')
        # print("Now trainning model: " + model)
        globals()[model + "_oof_train_" + subsetname], globals()[model + "_oof_test_" + subsetname] = pred_oof(
            estimator_ = globals()[model + "_reg_" + subsetname], 
            data_train_ = globals()["data_train_" + subsetname], 
            label_train_ = globals()["label_train"], 
            data_test_ = globals()["data_test_" + subsetname]
            )
    print("Training finished")
    pass


def datatype_conv(subsetname_):
    # models = ["rf", "lgb", "df21", "dnn"]
    models = ["rf", "lgb", "df21", "dnn", "knn"]
    subsetname = subsetname_

    for model in models:  # models
        globals()[model + "_oof_test_" + subsetname] = globals()[model + "_oof_test_" + subsetname].astype(float)
        globals()[model + "_oof_train_" + subsetname] = globals()[model + "_oof_train_" + subsetname].astype(float)
    pass

def generate_data_layer_2(subsetname_):
    subsetname = subsetname_
    # models = ["rf", "lgb", "df21", "dnn"]
    models = ["rf", "lgb", "df21", "dnn", "knn"]
    # subsetname e.g. all

    globals()["data_train_final_" + subsetname] = pd.DataFrame(data = np.empty((len(globals()["rf_oof_train_" + subsetname]), len(models))), columns = models)
    globals()["data_test_final_" + subsetname] = pd.DataFrame(data = np.empty((len(globals()["rf_oof_test_" + subsetname]), len(models))), columns = models)

    for model in models: # models
        globals()["data_train_final_" + subsetname].loc[:,model] = globals()[model + "_oof_train_" + subsetname]
        globals()["data_test_final_" + subsetname].loc[:,model] = globals()[model + "_oof_test_" + subsetname]

    # combine train and test
    globals()["data_final_" + subsetname] = pd.DataFrame(data = None, columns = models, dtype = int)
    globals()["label_final_" + subsetname] = pd.DataFrame(data = None, columns = models, dtype = int)

    globals()["data_final_" + subsetname] = pd.concat([globals()["data_train_final_" + subsetname], globals()["data_test_final_" + subsetname]], axis = 0, ignore_index = True)
    globals()["label_final"] = pd.concat([globals()["label_train"], globals()["label_test"]], axis = 0, ignore_index = True)
    globals()["label_final"] = globals()["label_final"].astype(int)
    globals()["label_train_final"] = globals()["label_train"].astype(int)
    globals()["label_test_final"] = globals()["label_test"].astype(int)

    # data after concatenation  e.g.   data_final_all / label_final / data_test_final_all
    pass

def train_final_model(subsetname_):
    subsetname = subsetname_
    # XGBoost
    globals()["xgb_reg_" + subsetname] = xgb.XGBRegressor(
        objective = "reg:squarederror", 
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
    # booster e.g.   xgb_reg_all
    # estimator_, MSE_list, mean_MSE, original_value, testing_value
    # e.g. xgb_reg_all, _, xgb_all_stacking_mean_MSE, xgb_all_stacking_original, xgb_all_stacking_testing
    globals()["xgb_reg_" + subsetname], _, globals()[subsetname + "_mean_AUC"], globals()[subsetname + "_original"], globals()[subsetname + "_testing"] = trainingCV(
        estimator_ = globals()["xgb_reg_" + subsetname], 
        data_ = globals()["data_final_" + subsetname], 
        label_ = globals()["label_final"], 
        n_splits_ = 5
    )
    pass

def applyGridSearch(estimator_ , parameters_, data_, label_):
    reg = GridSearchCV(
        estimator = estimator_, 
        param_grid = parameters_, 
        n_jobs = -1, 
        verbose = 3, 
        scoring = 'neg_mean_squared_error', 
        refit = True, 
        cv = 5
    )
    reg.fit(data_, label_)
    return reg.best_estimator_, reg.best_score_, reg.best_params_, reg.cv_results_


if __name__ == "__main__":
    pass