import lightgbm as lgb
import numpy as np
# data: https://github.com/nshomron/covidpred/tree/master/data
import pandas as pd

#laad train and validation data
train_pd=pd.read_csv('data/preprocessed_train.csv', sep=',')
valid_pd=pd.read_csv('data/preprocessed_valid.csv', sep=',')

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
bst = lgb.train(train_param, train_data, num_boost_round=10, early_stopping_rounds=5, valid_sets=[validation_data])

#data eval
data_eval = bst.eval(train_data, name="training")
print(data_eval)

#predict
ypred = bst.predict(train_pd)
print(type(ypred))
print(ypred)
mask = ypred > 0.5
print(mask)
train_pd["predict"] = mask
train_pd.to_csv('data/predicted_train_17Uhr.csv', index=False)

#predict eval
eval_train = bst.eval_train()
print(eval_train)
