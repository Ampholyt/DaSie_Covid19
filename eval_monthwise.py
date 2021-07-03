import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import lightgbm as lgb
from sklearn import metrics # plots
import seaborn as sb

import shap  # package used to calculate shap values
"""
    every model against every month in the future

    Iteration over the models
    Über alle Modelle (Monate) iterieren und n+1 für testing und n für training genutzt.


"""

def model_evaluation(trained_model, test_X, test_target, onlyAUC=False, month="None"):

    #predict
    ypred = trained_model.predict(test_X) #
    # print(ypred)
    # print(max(ypred))
    # print(min(ypred))

    fpr, tpr, thresholds = metrics.roc_curve(test_target, ypred, pos_label=1)

    #########AUC#############
    AUC = metrics.auc(fpr,tpr)
    print("The AUC is {}".format(AUC))

    if not onlyAUC and AUC > 0.5:
        ########ROC_Curve########
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkblue',
                 lw=lw, label='ROC curve')
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig("images/" + month + "_ROC.png")

        ########auPRC############
        precision, recall, thresholds = metrics.precision_recall_curve(test_target, ypred, pos_label=1)
        plt.figure()
        lw = 2
        plt.plot(recall, precision, color='darkblue',
                 lw=lw, label='Precision Recall Curve')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall-Curve')
        plt.legend(loc="lower left")
        plt.savefig("images/" + month + "_PRC.png")

        #############SHAP###############
        trained_model.params["objective"] = "binary"

        explainer = shap.explainers.Exact(trained_model.predict, test_X)
        shap_values = explainer(test_X)

        plt.figure()
        shap.plots.beeswarm(shap_values, show=False)
        plt.savefig("images/" + month + "_SHAP.png")

    return AUC


def main():
    prefix = "C:/Users/victo/Desktop/"
    iwas_prefix = "C:/Users/victo/Google Drive/Uni/DataScience/python/DaSie_Covid19/"
    # iwas_prefix = ""
    # prefix = ""
    months = []
    with open("data/months/month_list.txt", "r") as month_list:
        months = month_list.read().splitlines()
        months.reverse()
    auc_df = pd.DataFrame(np.nan, index=months, columns=months)
    for i in range(len(months)):
        cur_model = lgb.Booster(model_file=iwas_prefix + f'models/mode_{months[i]}.txt')
        # fileprefix = f'data/months/preprocessed_{months[i]}_'
        fileprefix = f'{prefix}data/months/preprocessed_{months[i]}_'
        test_pd = pd.read_csv(fileprefix + 'test.csv', sep=',')
        test_X = test_pd.drop(labels="corona_result", axis=1)
        test_target = test_pd["corona_result"]
        print("im i Bereich: ", i)
        auc_i = model_evaluation(cur_model, test_X, test_target, False, months[i]) # normal eval on same month (shap and so on)
        auc_df.at[f'{months[i]}', f'{months[i]}'] = auc_i
        for j in range(len(months)):
            if(j > i):
                # fileprefix = f'data/months/preprocessed_{months[j]}_'
                fileprefix = f'{prefix}data/months/preprocessed_{months[j]}_'
                test_pd = pd.read_csv(fileprefix + 'test.csv', sep=',')
                test_X = test_pd.drop(labels="corona_result", axis=1)
                test_target = test_pd["corona_result"]
                print("im j Bereich: ", i, " - ",  j)
                auc_df.at[f'{months[i]}', f'{months[j]}'] = model_evaluation(cur_model, test_X, test_target, True) # only auc
    plt.figure()
    sb.heatmap(auc_df.where(auc_df > 0.5), annot=True) #heatmap
    plt.savefig("images/heatmap_training(row)_testing(column).png")
if __name__ == "__main__":
    main()
