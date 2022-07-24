import pandas as pd
import numpy as np

def get_fpr_tpr(y_true, y_pred):
    # cutoff by the prediction
    cutoff_list = list(np.linspace(start = np.min(y_pred), stop = np.max(y_pred), num = 200))
    tpr_list, fpr_list = [], []
    for cutoff in cutoff_list:
        # assign labels according to the cutoff IS
        y_labeled_pred = [int(item > cutoff) for item in y_pred]
        y_labeled_true = [int(item > cutoff) for item in y_true]
        # calculate each tpr fpr for fixed cutoff
        tp, fn, fp, tn = 0, 0, 0, 0
        for i in np.arange(len(y_true)):
            if y_labeled_pred[i] == 1 and y_labeled_true[i] == 1:
                tp += 1
            elif y_labeled_pred[i] == 0 and y_labeled_true[i] == 1:
                fn += 1
            elif y_labeled_pred[i] == 1 and y_labeled_true[i] == 0:
                fp += 1
            elif y_labeled_pred[i] == 0 and y_labeled_true[i] == 0:
                tn += 1
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        fpr_tpr_df = pd.concat([pd.Series(fpr_list), pd.Series(tpr_list)], axis = 1)
        fpr_tpr_df.columns = ["fpr", "tpr"]
        fpr_tpr_df.sort_values(by = ["fpr"], ascending = True, inplace = True)
    return list(fpr_tpr_df["fpr"]), list(fpr_tpr_df["tpr"])

def get_auc_prob(labels, probs, cutoff = 0.3414471):
    probs = [int(item >= cutoff) for item in probs]
    labels = [int(item >= cutoff) for item in labels]
    N, P = 0, 0
    neg_prob, pos_prob = [], []
    for index, label in enumerate(labels):
        if label == 1:
            P += 1
            pos_prob.append(probs[index])
        else:
            N += 1
            neg_prob.append(probs[index])
    number = 0
    for pos in pos_prob:
        for neg in neg_prob:
            if (pos > neg):
                number += 1
            elif (pos == neg):
                number += 0.5
    return number / (N * P)

def plot_roc_curve_alt(regname, y_true_all, rf_all_testing, rf_xgboost_testing, rf_top10_testing, title, save = False, mydir_ = "/"):
    fpr_all, tpr_all = get_fpr_tpr(y_true = y_true_all, y_pred = rf_all_testing)
    fpr_xgboost, tpr_xgboost = get_fpr_tpr(y_true = y_true_all, y_pred = rf_xgboost_testing)
    fpr_top10, tpr_top10 = get_fpr_tpr(y_true = y_true_all, y_pred = rf_top10_testing)
    import pandas as pd
    import numpy as np  
    import matplotlib
    from matplotlib import pyplot as plt
    from matplotlib import cm
    plt.style.use('ggplot')
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"
    fig,ax = plt.subplots(1, 1, figsize = (6, 6), dpi = 320)
    ax.plot([0,1], [0,1], 'k--', linewidth = '2')
    ax.plot(fpr_all, tpr_all, label = 'All', linewidth = '2')
    ax.plot(fpr_xgboost, tpr_xgboost, label = 'Selected From XGBoost', linewidth = '2')
    ax.plot(fpr_top10, tpr_top10, label = 'Top10', linewidth = '2')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.legend(loc='best')
    plt.show()
    if save:
        fig.savefig(mydir_ + title + ".png", format = "png")
    pass