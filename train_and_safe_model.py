import lightgbm as lgb
# data: https://github.com/nshomron/covidpred/tree/master/data

#training
#param
train_data = lgb.Dataset('train.bin').construct()
validation_data = lgb.Dataset('validation.bin').construct()
print("Rows of validation: ",validation_data.num_data(), "\nColumns of validation: ", validation_data.num_feature())

# param = {'num_leaves': 20, 'objective': 'binary',
# 'min_data_in_leaf': 4,
# 'feature_fraction': 0.2,
# 'bagging_fraction': 0.8,
# 'bagging_freq': 5,
# 'learning_rate': 0.05,
# 'verbose': 1}
#
# bst = lgb.train(param, train_data, num_boost_round=603, early_stopping_rounds=5, valid_sets=[validation_data])
#
#
# #save model:
# bst.save_model('model_round-603.txt')

#To load a scipy.sparse.csr_matrix array into Dataset:
# import scipy
# csr = scipy.sparse.csr_matrix((train_data_1, (500, 2)))
# train_data_2 = lgb.Dataset(csr)
#
# print(train_data_2)
