import lightgbm as lgb
import pandas as pd

#load model
print("hallo")
train_data = lgb.Dataset('train.bin')
bst = lgb.Booster(model_file='model_round-603.txt', train_set=train_data)  # init model
print("hallo")
#load train_data
# train_data=pd.read_csv('data/preprocessed_train.csv', sep=',')


# validation_data = lgb.Dataset('validation.bin')

# print(train_data)

asdf = bst.eval(train_data, name="training")
print(asdf)
# num_round = 10
# param = {}
# param['metric'] = 'auc'

# lgb.cv(param, train_data, num_round, nfold=5)
# ypred = bst.predict(train_data)
# print(type(ypred))
# print(ypred)
# mask = ypred > 0.5
# print(mask)
# train_data["predict"] = mask
# train_data.to_csv('data/predicted_train.csv', index=False)
