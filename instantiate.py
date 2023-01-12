import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from deepforest import CascadeForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor

randomstate = 42

# instantiate
seq = ["all", "xgboost", "top10", "all_stacking", "xgboost_stacking", "top10_stacking"]

# # linear regression
# regname = "lin"
# for name in seq:
#     globals()[regname + "_reg_" + name] = LinearRegression(
#         fit_intercept = False, 
#         n_jobs = -1
#     )

# Random Forest
regname = "rf"
for name in seq:
    globals()[regname + "_reg_" + name] = RandomForestRegressor(
        n_estimators = 200, 
        criterion = "mae", 
        max_depth = 10, 
        n_jobs = -1, 
        min_samples_split = 3, 
        max_features = 0.9, 
        bootstrap = True, 
        random_state = randomstate, 
        warm_start = True, 
        verbose = 0
    )



# lightGBM
regname = "lgb"
for name in seq:
    globals()[regname + "_reg_" + name] = lgb.LGBMRegressor(
        objective = 'regression', 
        num_leaves = 40, 
        max_depth = 10, 
        learning_rate = 0.01, 
        n_estimators = 300, 
        subsample = 0.9, 
        min_child_samples = 2, 
        silent = True, 
        importance_type = 'gain', 
        metric = 'mae'
    )

# DF21
regname = "df21"
for name in seq:
    globals()[regname + "_reg_" + name] = CascadeForestRegressor(
        n_bins = 255, 
        bin_subsample = 200, 
        max_layers = 5, 
        criterion = "mse", 
        n_estimators = 4, 
        n_trees = 200,     # 300?
        max_depth = 10, 
        use_predictor = True, 
        backend = "sklearn", 
        n_tolerant_rounds = 2, 
        n_jobs = -1, 
        verbose = 0, 
        random_state = randomstate
    )

# dnn
regname = "dnn"
for name in seq:
    globals()[regname + "_reg_" + name] = MLPRegressor(
        hidden_layer_sizes = (256, 128, 64), 
        activation = "relu", 
        solver = "adam", 
        alpha = 0.001, 
        batch_size = 64, 
        learning_rate = "adaptive", 
        learning_rate_init = 0.0005, 
        momentum = 0.4, 
        max_iter = 1000, 
        shuffle = True, 
        random_state = randomstate, 
        verbose = False, 
        early_stopping = False, 

    )

# kNN
regname = "knn"
for name in seq:
    globals()[regname + "_reg_" + name] = KNeighborsRegressor(
        algorithm =  'auto',
        leaf_size = 2,
        metric = 'minkowski',
        n_jobs = -1,
        n_neighbors = 2,
        p = 2,
        weights = 'uniform'
    )

if __name__ == "__main__":
    pass