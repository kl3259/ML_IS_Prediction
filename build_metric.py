from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
import numpy as np
import pandas as pd

evaluation_metric_df = pd.DataFrame(
    data = None, 
    columns = ["MAE", "R2", "Epsilon_Accuracy"]
)

def eps_accuracy(y_true, y_pred, eps = 0.05):
    count = 0
    for (pred_item, true_item) in zip(y_pred, y_true):
        if (pred_item >= (1 - eps) * true_item) and (pred_item <=  (1 + eps) * true_item):
            count += 1
    eps_accu = count / len(y_true)
    return eps_accu

def eps_accuracy_abs(y_true, y_pred, eps = 0.05):
    count = 0
    for (pred_item, true_item) in zip(y_pred, y_true):
        if (pred_item >= true_item - eps) and (pred_item <=  true_item + eps):
            count += 1
    eps_accu = count / len(y_true)
    return eps_accu

def RMSE(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true = y_true, y_pred = y_pred))

scoring = {"MAE": make_scorer(mean_absolute_error), "R2": "r2", "Eps_Accu": make_scorer(eps_accuracy_abs, greater_is_better = True)}




