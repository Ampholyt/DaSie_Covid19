import lightgbm as lgb
import numpy as np
# data: https://github.com/nshomron/covidpred/tree/master/data
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import shap  # package used to calculate shap values
import seaborn as sb

'''
Evaluate the own model on the full Dataset
'''

def plotImp(model, X, num = 8, fig_size = (40, 8)):
    feature_imp = pd.DataFrame({'Value':model.feature_importance(),'Feature':X.columns})
    print(feature_imp)
    plt.figure(figsize=fig_size)
    sb.set(font_scale = 5)
    sb.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    print(feature_imp)
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances-01.png')
    plt.show()



fileprefix = 'data/preprocessed_full_'

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

# training:
param_readin = {'feature_pre_filter': False} # why is this needed? (https://lightgbm.readthedocs.io/en/latest/Parameters.html search for feature_pre_filter)
train_data = lgb.Dataset(train_X, label=train_pd['corona_result'])
val_data = lgb.Dataset(valid_X, params=param_readin,  label=valid_pd['corona_result'], reference=train_data)
# train model
train_param = {'bagging_fraction': 0.9081128350776642,
                'feature_fraction': 0.6757639514310116,
                'learning_rate': 0.46164810494204905,
                'max_bin': 88,
                'max_depth': 28,
                'min_data_in_leaf': 55,
                'min_sum_hessian_in_leaf': 39.895374699362016,
                'num_leaves': 34,
                'subsample': 0.02567451763571079,
                'objective': 'binary',
                'metric': 'auc',
                'is_unbalance': True,
                'boost_from_average': False,
                'verbose': -1}
#num_boost_round=603
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
print("max: ", max(ypred), "\nmin: ", min(ypred))
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
    # Create object that can calculate shap values
    explainer = shap.TreeExplainer(bst)
    print(explainer.expected_value)
    # Calculate Shap values
    shap_values = explainer.shap_values(vX)

    print(shap_values)
    # shap.initjs()
    print("summary plot: ")
    shap.summary_plot(shap_values, vX)

    print("\nsummary plot for predicting class 1: ")
    shap.summary_plot(shap_values[1], vX)

    print("\nsummary plot for predicting class 0: ")
    shap.summary_plot(shap_values[0], vX)




    # print(shap_values)
    # print(valid_X)
    # shap.dependence_plot("Cough", shap_values[0], valid_X[0])
    # shap.dependence_plot("Fever", shap_values[1], valid_X[1])

    # # How to use this?
    # shap.force_plot(explainer.expected_value[1], shap_values[1], vX)
    #
    # # use Kernel SHAP to explain test set predictions
    # k_explainer = shap.KernelExplainer(my_model.predict_proba, train_X)
    # k_shap_values = k_explainer.shap_values(vX)
    # shap.force_plot(k_explainer.expected_value[1], k_shap_values[1], vX)
