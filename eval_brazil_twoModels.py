import lightgbm as lgb
import numpy as np
# data: https://github.com/nshomron/covidpred/tree/master/data
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import shap  # package used to calculate shap values

'''
Evaluate the paper's (default) or a own model on the brazil's data
'''

dataPrefix = 'data/preprocessed_brazil_'
dataModel = 'models/lgbm_model_all_features.txt'

# load train and validation data
train_pd = pd.read_csv(dataPrefix + 'train.csv', sep=',')
valid_pd = pd.read_csv(dataPrefix + 'val.csv', sep=',')
test_pd = pd.read_csv(dataPrefix + 'test.csv', sep=',')

train_X = train_pd.drop(labels="Corona_Result", axis=1)
valid_X = valid_pd.drop(labels="Corona_Result", axis=1)
test_X = test_pd.drop(labels="Corona_Result", axis=1)

# possible print
print(train_pd)
print(test_pd)
print(valid_pd)

'''
 Either load a model or train a model on the train data
 (default: load the model from the paper)
'''

# load model:
bst = lgb.Booster(model_file=dataModel)
# OR
# # train model:
# param_readin = {'feature_pre_filter': False} # why is this needed? (https://lightgbm.readthedocs.io/en/latest/Parameters.html search for feature_pre_filter)
# train_data = lgb.Dataset(train_X, label=train_pd['Corona_Result'])
# val_data = lgb.Dataset(valid_X, params=param_readin,  label=valid_pd['Corona_Result'], reference=train_data)
# # train model
# train_param = {'num_leaves': 20,
#                'objective': 'binary',
#                'min_data_in_leaf': 4,
#                'feature_fraction': 0.2,
#                'bagging_fraction': 0.8,
#                'bagging_freq': 5,
#                'learning_rate': 0.05,
#                'verbose': 1,
#     }
# #num_boost_round=603
# bst = lgb.train(train_param, train_data, num_boost_round=604, early_stopping_rounds=5, valid_sets=[val_data])

#predict
ypred = bst.predict(test_X, predict_disable_shape_check=True)
print(ypred)
print(max(ypred))
print(min(ypred))


#validation/testset
vY = test_pd["Corona_Result"]
vX = test_pd.drop(labels="Corona_Result", axis=1)
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
         lw=lw, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall-Curve')
plt.legend(loc="lower left")
plt.show()


#######SHAP##############
# shap values are not possible with the paper's model
#
# shaping = input("Do you want to calculate the shap values?")
# if (shaping == 'yes' or shaping == 'y'):
#     # fix?
#     bst.params["objective"] = "binary"
#     # Create object that can calculate shap values
#     explainer = shap.TreeExplainer(bst)
#
#     # Calculate Shap values
#     shap_values = explainer.shap_values(vX)
#
#     # shap.initjs()
#     shap.summary_plot(shap_values, vX)
#
#     # How to use this?
#     shap.force_plot(explainer.expected_value[1], shap_values[1], vX)
#
#     # use Kernel SHAP to explain test set predictions
#     k_explainer = shap.KernelExplainer(my_model.predict_proba, train_X)
#     k_shap_values = k_explainer.shap_values(vX)
#     shap.force_plot(k_explainer.expected_value[1], k_shap_values[1], vX)
