import lightgbm as lgb
import numpy as np
# data: https://github.com/nshomron/covidpred/tree/master/data
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics


# load train and validation data
train_pd = pd.read_csv('data/preprocessed_github_train.csv', sep=',')
valid_pd = pd.read_csv('data/preprocessed_github_val.csv', sep=',')
test_pd = pd.read_csv('data/preprocessed_github_test.csv', sep=',')

train_X = train_pd.drop(labels="corona_result", axis=1)
valid_X = valid_pd.drop(labels="corona_result", axis=1)
test_X = test_pd.drop(labels="corona_result", axis=1)


# print(train_X.dtypes)
# print(test_X.dtypes)
# print(valid_X.dtypes)

# create datasets from lgb
# param_readin = {'feature_pre_filter': False}

# train_data = lgb.Dataset(train_X, params=param_readin, label=train_pd['corona_result']).construct()
#
# validation_data = lgb.Dataset(valid_X, params=param_readin,  label=valid_pd['corona_result'], reference=train_data).construct()

# load model from file
bst = lgb.Booster(model_file='models/lgbm_model_all_features.txt')

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


#
# #confusion matrix #### redo is broken now --> use threshold from roc curve to create mask
# mask = ypred < threshold
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(vY, mask)
# print('\nConfusion matrix\n\n', cm)
# print('\nTrue Positives(TP) = ', cm[0,0])
# print('\nTrue Negatives(TN) = ', cm[1,1])
# print('\nFalse Positives(FP) = ', cm[0,1])
# print('\nFalse Negatives(FN) = ', cm[1,0])



# import shap  # package used to calculate Shap values
#
# # Create object that can calculate shap values
# explainer = shap.TreeExplainer(bst)
#
# # Calculate Shap values
# shap_values = explainer.shap_values(vX)
#
# shap.initjs()
# shap.summary_plot(shap_values, vX)
#
# # shap.force_plot(explainer.expected_value[1], shap_values[1], vX)
# #
# # # use Kernel SHAP to explain test set predictions
# # k_explainer = shap.KernelExplainer(my_model.predict_proba, train_X)
# # k_shap_values = k_explainer.shap_values(vX)
# # shap.force_plot(k_explainer.expected_value[1], k_shap_values[1], vX)
