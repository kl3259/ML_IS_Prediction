import matplotlib
from matplotlib import cm
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

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

def show_evaluation_metrics_alt(curr_comb, grid_search, best_param, curr_method = None):
    gs_result = grid_search.cv_results_
    best_index = grid_search.best_index_
    # rmse = gs_result['mean_test_RMSE'][best_index]
    mae = gs_result['mean_train_MAE'][best_index]
    r2 = gs_result['mean_train_R2'][best_index]
    eps_accu = gs_result['mean_test_Eps_Accu'][best_index]

    evaluation_metric_df = pd.DataFrame(
        data = [[ mae, r2, eps_accu]], 
        index = [curr_method + curr_comb], 
        columns = ["MAE", "R2", "Epsilon_Accuracy"]
    )
    print(f'Current candidates: {curr_comb}\nTest MAE: {mae:8.4f}\nTest Eps Accu: {eps_accu:8.4f}\nR^2 {r2:8.4f}\nBest param: {best_param}')
    print(f'-----------------------------------------------------------------------------------------------------------------------------------------')
    return evaluation_metric_df


