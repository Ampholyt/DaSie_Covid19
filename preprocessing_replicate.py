import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
#import seaborn as sn
#import matplotlib.pyplot as plt

'''
Replicate dataset used in the paper
'''


# data: https://github.com/nshomron/covidpred/tree/master/data

fileprefix = 'C:/Users/victo/Desktop/data/'

# raw Data from GitHub:
path = fileprefix + 'corona_tested_individuals_ver_006.english.csv'


# read data
# df=pd.read_csv('data/corona_tested_individuals_ver_006.english.csv', sep=',')
df = pd.read_csv(path, low_memory=False)


# drop all obs where data is not binary (mostly missing info)
df = df[df['corona_result'] != 'other']  # corona result not negative or positive
df = df[df['gender'] != 'None']
# this reproduces the exact data from the paper
# replace 'None' with np.nan so lgbm can work with it
df['cough'].replace('None', np.nan, inplace=True)
df['fever'].replace('None', np.nan, inplace=True)
df['sore_throat'].replace('None', np.nan, inplace=True)
df['shortness_of_breath'].replace('None', np.nan, inplace=True)
df['head_ache'].replace('None', np.nan, inplace=True)
df['age_60_and_above'].replace('None', np.nan, inplace=True)

# replace categories with numbers



translations = {
    'corona_result': {
        'negative': 0,  # negative
        'positive': 1,  # positive
    },
    'age_60_and_above': {
        'No': 0,
        'Yes': 1
    },
    'gender': {
        'female': 0,  # female
        'male': 1  # male
    },
    'test_indication': {
        'Contact with confirmed': 1, # as test make this binary also
        'Abroad': 0,
        'Other': 0
    }
}

df = df.replace(translations)
df.columns = ["test_date", "Cough", "Fever", "Sore_throat", "Shortness_of_breath",
              "Headache", "corona_result", "Age_60+", "Male", "Contact_with_confirmed"]

# Training data: 22.03.20 - 31.03.20 # n=51.831, corona=4769
# Test data:     01.04.20 - 07.04.20 # n=47.401, corona=3624

train_df = df[((df['test_date'] >= '2020-03-22') & (df['test_date'] <= '2020-03-31'))]
test_df = df[((df['test_date'] >= '2020-04-01') & (df['test_date'] <= '2020-04-07'))]

# Date not needed for analysis
test_df.drop(['test_date'], axis=1, inplace=True)
train_df.drop(['test_date'], axis=1, inplace=True)

test_df = test_df.astype(float)
train_df = test_df.astype(float)

print(test_df.dtypes)


# Numbers are not exactly the same but very similar to the ones used in the paper.
print(f"Train n = {len(train_df)}, corona: {sum(train_df['corona_result'] == 1)}")
print(train_df.nunique())
print(f"Test n = {len(test_df)}, corona: {sum(test_df['corona_result'] == 1)}")
print(test_df.nunique())




# split training data into training and validation data
train_df, val_df = train_test_split(train_df, test_size=0.2)


# Save data and category explanations
filename = 'preprocessed_github_'

train_df.to_csv(fileprefix + filename + 'train.csv', index=False)
val_df.to_csv(fileprefix + filename + 'val.csv', index=False)
test_df.to_csv(fileprefix + filename + 'test.csv', index=False)
