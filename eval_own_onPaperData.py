import lightgbm as lgb
import numpy as np
# data: https://github.com/nshomron/covidpred/tree/master/data
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn import metrics
import shap  # package used to calculate shap values


#building models
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_score

#feature importance
import seaborn as sns

'''
Evaluate the own model on the Data from the paper
'''

# fileprefix = 'data/preprocessed_full_'
fileprefix = 'C:/Users/victo/Desktop/data/preprocessed_full_'

# load train and validation data
train_pd = pd.read_csv(fileprefix + 'train.csv', sep=',')
valid_pd = pd.read_csv(fileprefix + 'val.csv', sep=',')
test_pd = pd.read_csv(fileprefix + 'test.csv', sep=',')

train_X = train_pd.drop(labels="corona_result", axis=1)
valid_X = valid_pd.drop(labels="corona_result", axis=1)
test_X = test_pd.drop(labels="corona_result", axis=1)

# possible print
# print(train_X.dtypes)
# print(test_X.dtypes)
# print(valid_X.dtypes)

# load model from file
# todo import own model
################################################################################
bst = lgb.Booster(model_file='models/own_onFullData.txt')
################################################################################

#predict
ypred = bst.predict(test_X)
print(ypred)
print(max(ypred))
print(min(ypred))


#validation/testset
vY = test_pd["corona_result"]
vX = test_pd.drop(labels="corona_result", axis=1)
print(vX)

#roc curve:
fpr, tpr, thresholds = metrics.roc_curve(vY, ypred, pos_label=1)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkblue',
         lw=lw, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

#########AUC#############
AUC = metrics.auc(fpr,tpr)
print("The AUC is {}".format(AUC))

########auPRC############
precision, recall, thresholds = metrics.precision_recall_curve(vY, ypred, pos_label=1)
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
plt.show()


#######SHAP##############

shaping = input("Do you want to calculate the shap values?")
if (shaping == 'yes' or shaping == 'y'):
    # fix?
    bst.params["objective"] = "binary"

    explainer = shap.TreeExplainer(bst, vX)
    shap_values = explainer(vX)

    shap.plots.beeswarm(shap_values, order=shap_values.abs.max(0))




    # # Create object that can calculate shap values
    # explainer = shap.TreeExplainer(bst)
    # print(explainer.expected_value)
    # # Calculate Shap values
    # shap_values = explainer.shap_values(vX)
    #
    # print(shap_values)
    # # shap.initjs()
    # print("summary plot: ")
    # shap.summary_plot(shap_values, vX)
    #
    # print("\nsummary plot for predicting class 1: ")
    # shap.summary_plot(shap_values[1], vX)
    #
    # print("\nsummary plot for predicting class 0: ")
    # shap.summary_plot(shap_values[0], vX)
