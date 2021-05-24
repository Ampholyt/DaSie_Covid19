import lightgbm as lgb
import numpy as np
# data: https://github.com/nshomron/covidpred/tree/master/data
import pandas as pd
from matplotlib import pyplot as plt

#laad train and validation data
train_pd=pd.read_csv('data/preprocessed_train.csv', sep=',')
valid_pd=pd.read_csv('data/preprocessed_valid.csv', sep=',')
test_pd=pd.read_csv('data/preprocessed_test.csv', sep=',')

#create datasets from lgb
param_readin = {'feature_pre_filter': False}
train_data = lgb.Dataset(train_pd, params=param_readin, label=train_pd['corona_result']).construct()

validation_data = lgb.Dataset(valid_pd, params=param_readin,  label=valid_pd['corona_result'], reference=train_data).construct()

#training
train_param = {'num_leaves': 20, 'objective': 'binary',
'min_data_in_leaf': 4,
'feature_fraction': 0.2,
'bagging_fraction': 0.8,
'bagging_freq': 5,
'learning_rate': 0.05,
'verbose': 1}
#num_boost_round=603
bst = lgb.train(train_param, train_data, num_boost_round=80, early_stopping_rounds=5, valid_sets=[validation_data])

#predict
ypred = bst.predict(test_pd)
mask = ypred > 0.5
#validation/testset
vy = test_pd["corona_result"]
vX = test_pd.drop(labels="corona_result", axis=1)
#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(vy, mask)
print('\nConfusion matrix\n\n', cm)
print('\nTrue Positives(TP) = ', cm[0,0])
print('\nTrue Negatives(TN) = ', cm[1,1])
print('\nFalse Positives(FP) = ', cm[0,1])
print('\nFalse Negatives(FN) = ', cm[1,0])

#roc curve:
import numpy as np
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(vy, mask, pos_label=1)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
