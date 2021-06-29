import lightgbm as lgb
import pandas as pd
import shap  # package used to calculate shap values
from matplotlib import pyplot as plt
from sklearn import metrics

# data: https://github.com/nshomron/covidpred/tree/master/data

'''
Evaluate the paper's model
'''

# fileprefix = 'C:/Users/victo/Desktop/data/preprocessed_github_'
fileprefix = 'data/preprocessed_github_'

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

    shap.plots.beeswarm(shap_values)
