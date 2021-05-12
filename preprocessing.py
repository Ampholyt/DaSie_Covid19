import lightgbm as lgb
import numpy as np
# data: https://github.com/nshomron/covidpred/tree/master/data
import pandas as pd
#reading in

df=pd.read_csv('data/corona_tested_individuals_ver_0083.english.csv', sep=',')
# show nan: #11
df1 = df[df.isna().any(axis=1)]
print("All nan values:\n", df1)
# drop nan
df.dropna(inplace=True)

#drop date
del df['test_date']

# preprocessing:
# date notwendig?,
# age: Yes = 1
# gender: female = 1
# test_indication: Other = 0 was machen wir mit im Ausland (= 2)

#corona result filtering
df.loc[(df.corona_result == 'positive'),'corona_result']=True
df.loc[(df.corona_result == 'negative'),'corona_result']=False
#how many others? (9 in 2000 rows)
print("All values of corona result:\n", df.groupby('corona_result').count())


# removing others
df = df[df['corona_result'] != "other"]

df_res = df['corona_result']

df_res = df_res.squeeze()
df['corona_result'] = df_res
#age filtering
# df = df[(df.age_60_and_above != 'Yes') & (df.age_60_and_above != 'No')]
df.loc[(df.age_60_and_above == 'Yes'),'age_60_and_above']=True
df.loc[(df.age_60_and_above == 'No'),'age_60_and_above']=False
print("All values of age_60_and_above:\n", df.groupby('age_60_and_above').count())

# print('age filtered\n', df)

#gender filtering
df.loc[(df.gender == 'female'),'gender']=True
df.loc[(df.gender == 'male'),'gender']=False
# print("gender filtered:\n", df)

#test_indication filtering
df.loc[(df.test_indication == 'Other'),'test_indication']=0
df.loc[(df.test_indication == 'Contact with confirmed'),'test_indication']=1
df.loc[(df.test_indication == 'Abroad'),'test_indication']=2
# print(df.test_indication.unique())

print("finished filtering\n", df)
#change data types
df = df.astype({'corona_result': 'int64', 'age_60_and_above': 'int64', 'gender': 'int64', 'test_indication': 'int64'})
print(df.dtypes)

#spliting into test, train and validation data
# 4 - 1: train valid
mask = np.random.rand(len(df)) < 0.8
print(mask)
train = df[mask]

validation = df[~mask]

print("length of train: ", len(train), "\nlength of valid: ", len(validation))

param = {'feature_pre_filter': False}

# df.to_csv('data/preprocessed_full.csv', index=False)
train.to_csv('data/preprocessed_train.csv', index=False)
validation_data.to_csv('data/preprocessed_valid.csv', index=False)
#end

#tried to compute everthing in one file:


train_data = lgb.Dataset(train, params=param, label=train['corona_result']).construct()

validation_data = lgb.Dataset(validation, params=param,  label=validation['corona_result'], reference=train_data).construct()


param = {'num_leaves': 20, 'objective': 'binary',
'min_data_in_leaf': 4,
'feature_fraction': 0.2,
'bagging_fraction': 0.8,
'bagging_freq': 5,
'learning_rate': 0.05,
'verbose': 1}

bst = lgb.train(param, train_data, num_boost_round=603, early_stopping_rounds=5, valid_sets=[validation_data])



bst.save_model('model_round-603_test.txt') #save model:

asdf = bst.eval(validation_data, name="training")


print("Rows of validation: ",validation_data.num_data(), "\nColumns of validation: ", validation_data.num_feature())
