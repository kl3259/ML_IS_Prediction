import pandas as pd
import numpy as np
import matplotlib
from matplotlib import cm
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

def myplot(title_, label_pred_, label_original_, save = False, mydir_ = "/"):
    plt.style.use("ggplot")
    pred_all = label_pred_
    original_all = label_original_

    plt.rcParams["font.weight"] = "bold"

    fig = plt.figure(figsize = (6,6))
    ax = fig.add_subplot()
    # label_list = ["original data", "prediction"]
    # color = ['tab:blue', 'tab:orange']
    # for i in range(len(color)):
    color = ['tab:blue']
    x, y = original_all, pred_all
    scale = 10
    x_veri = np.linspace(0, 0.8, 1000)
    y_veri = np.linspace(0, 0.8, 1000)
    ax.scatter(x, y, color = [3/255,37/255,108/255], s = scale, alpha = 0.3)
    ax.plot(x_veri, y_veri, linewidth = 1, color = "darkgrey")
    ax.set_xlim(0,0.8)
    ax.set_ylim(0,0.8)
    ax.set_xlabel("Original Value", weight = "bold")
    ax.set_ylabel("Prediction Value", weight = "bold")
    # ax.legend()
    ax.grid(False)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['bottom'].set_color("black")
    ax.spines['left'].set_linewidth(1)
    ax.spines['left'].set_color("black")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.patch.set_facecolor('white') 
    ax.set_facecolor('white')
    plt.show()
    if save:
        fig.savefig(mydir_ + title_ + "_prediction_figure_chubu_new.png", dpi = 240)
    return

def get_eps_accu_abs(label_pred_, label_original_):
    temp_list_1 = [(item - 0.05) for item in label_original_] 
    temp_list_2 = [(item + 0.05) for item in label_original_]
    count = 0
    for i in range(len(label_pred_)):
        if label_pred_[i] >= temp_list_1[i] and label_pred_[i] <= temp_list_2[i]:
            count += 1
    eps_accu = count / len(label_pred_)
    return eps_accu

def train_loop(regname):
    seq = ["all", "xgboost", "top10"]
    for name in seq:
        # training
        globals()[regname + "_reg_" + name], _, globals()[regname + "_" + name + "_mean_MAE"], globals()[regname + "_" + name + "_original"], globals()[regname + "_" + name + "_testing"] = trainingCV(estimator_ = globals()[regname + "_reg_" + name], data_ = globals()["imputed_df_" + name], label_ = result_df, n_splits_ = 5) 
        # output
        print("-------------------normal-------------------")
        print(globals()[regname + "_" + name + "_mean_MAE"])
        subsetname = regname + "_" + name
        globals()[subsetname + "_accu"] = get_eps_accu_abs(label_pred_ = globals()[regname + "_" + name + "_testing"], label_original_ = globals()[regname + "_" + name + "_original"])
        print(globals()[subsetname + "_accu"])
        globals()[subsetname + "_r2"] = r2_score(y_pred = globals()[subsetname + "_testing"], y_true = globals()[subsetname + "_original"])
        print(globals()[subsetname + "_r2"])

        myplot(title_ = "prediction_" + subsetname, label_pred_ = globals()[subsetname + "_testing"], label_original_ = globals()[subsetname + "_original"])

        curr_eval_df = pd.DataFrame(
            data = [[globals()[subsetname + "_mean_MAE"] / 10000, globals()[subsetname + "_r2"], globals()[subsetname + "_accu"]]], 
            index = [subsetname], 
            columns = ["MAE", "R2", "Epsilon_Accuracy"]
        )
        evaluation_metric_df = pd.concat(
            [evaluation_metric_df, curr_eval_df], 
            axis = 0
        )
    pass

if __name__ == "__main__":
    pass
