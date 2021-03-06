import lightgbm as lgb
import numpy as np
# data: https://github.com/nshomron/covidpred/tree/master/data
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import shap  # package used to calculate shap values

'''
Evaluate the paper's (default) or a own model on the full Data
'''

# pathprefix = 'C:/Users/victo/Desktop/'
pathprefix = ''

fileprefix = pathprefix + 'data/preprocessed_full_'

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
# bst = lgb.Booster(model_file='models/lgbm_model_all_features.txt')
#  OR
# Train our own model
# training:
param_readin = {'feature_pre_filter': False} # why is this needed? (https://lightgbm.readthedocs.io/en/latest/Parameters.html search for feature_pre_filter)
train_data = lgb.Dataset(train_X, label=train_pd['corona_result'])
val_data = lgb.Dataset(valid_X, params=param_readin,  label=valid_pd['corona_result'], reference=train_data)
# train model
train_param = {'num_leaves': 34, #34.24
               'objective': 'binary',
               'min_data_in_leaf': 54, #54.56
               'min_sum_hessian_in_leaf': 40, #39.9
               'feature_fraction': 0.6758,
               'bagging_fraction': 0.9081,
               'bagging_freq': 5,
               'subsample': 0.02567,
               'learning_rate': 0.4616,
               'max_bin': 88, #88.32
               'max_depth': 28, #28.23
    }

bst = lgb.train(train_param, train_data, num_boost_round=604, early_stopping_rounds=50, valid_sets=[val_data])

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
print("The AUC is {}".format(AUC)) #0.8317182822094226 (603 rounds (break at 62)) 0.8314688168533265 (20 rounds)

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
    # Create object that can calculate shap values
    explainer = shap.TreeExplainer(bst)

    # Calculate Shap values
    shap_values = explainer.shap_values(vX)

    # shap.initjs()
    shap.summary_plot(shap_values, vX)

    # # How to use this?
    # shap.force_plot(explainer.expected_value[1], shap_values[1], vX)
    #
    # # use Kernel SHAP to explain test set predictions
    # k_explainer = shap.KernelExplainer(my_model.predict_proba, train_X)
    # k_shap_values = k_explainer.shap_values(vX)
    # shap.force_plot(k_explainer.expected_value[1], k_shap_values[1], vX)
