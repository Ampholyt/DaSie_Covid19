import lightgbm as lgb
# data: https://github.com/nshomron/covidpred/tree/master/data

fileprefix = 'data/preprocessed_full_'
# fileprefix = 'C:/Users/victo/Desktop/data/preprocessed_full_'


# load train and validation data
train_pd = pd.read_csv(fileprefix + 'train.csv', sep=',')
valid_pd = pd.read_csv(fileprefix + 'val.csv', sep=',')
test_pd = pd.read_csv(fileprefix + 'test.csv', sep=',')

train_X = train_pd.drop(labels="corona_result", axis=1)
valid_X = valid_pd.drop(labels="corona_result", axis=1)
test_X = test_pd.drop(labels="corona_result", axis=1)

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

# save one model
bst.save_model('models/own_onFullData.txt')
