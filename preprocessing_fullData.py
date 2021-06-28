import pandas as pd
from sklearn.model_selection import train_test_split
#import numpy as np
#import seaborn as sn
#import matplotlib.pyplot as plt

'''
Preprocess raw data from Israeli Ministry of Health
Translated website: https://4ezcibkawkaklp6a2uxvwy2q6q-ac4c6men2g7xr2a-data-gov-il.translate.goog/dataset/covid-19
'''

# pathprefix = ''
pathprefix = 'C:/Users/victo/Desktop/'

path = pathprefix + 'data/corona_tested_individuals_ver_00151.csv'

# read data
df = pd.read_csv(path, low_memory=False)

# Translate from Hebrew
# Change categorical to numerical data and translate hebrew
translations = {
    'corona_result': {
        'שלילי': 0,  # negative
        'חיובי': 1,  # positive
        'אחר': 2  # other
    },
    'age_60_and_above': {
        'No': 0,
        'Yes': 1
    },
    'gender': {
        'נקבה': 0,  # female
        'זכר': 1  # male
    },
    'test_indication': {
        'Abroad': 0,
        'Contact with confirmed': 1,
        'Other': 0
    }
}

df = df.replace(translations)


# Replicate the data structure from the paper:

# drop all obs where corona result is not binary
df = df[df['corona_result'] != 2]
df = df.dropna(subset=['gender'])

# unnecessary on raw data, missing data already nan
# # replace 'None' with np.nan so lgbm can work with it
# df['cough'].replace('None', np.nan, inplace=True)
# df['fever'].replace('None', np.nan, inplace=True)
# df['sore_throat'].replace('None', np.nan, inplace=True)
# df['shortness_of_breath'].replace('None', np.nan, inplace=True)
# df['head_ache'].replace('None', np.nan, inplace=True)
# df['age_60_and_above'].replace('None', np.nan, inplace=True)

# replace categories with numbers


df.columns = ["test_date", "Cough", "Fever", "Sore_throat", "Shortness_of_breath",
              "Headache", "corona_result", "Age_60+", "Male", "Contact_with_confirmed"]


'''
Different usecases for this data set:
    1 - Evaluate model from the paper
    2 - Train and test new model
    3 - Train and Test models on single months
Comment out the case you don't need currently
'''
outfile_prefix = 'preprocessed_full_'

# 1 - Evaluate model from the paper #####################################################
# all data can be used for testing only excluding the data that was originally used for
# training/validation (week 2020-03-22 - 2020-03-31)
#
# test_zoabi_df = df[df['test_date'] >= '2020-04-01']
# test_zoabi_df = test_zoabi_df.drop(['test_date'], axis=1).astype(float)
#
# test_zoabi_df.to_csv(pathprefix + 'data/' + outfile_prefix + 'test_zoabi.csv', index=False)

# 2 - Train and Test new model ##########################################################
# data is randomly split 4:1 into training and testing sets
# training set itself is split 4:1 into training and validation sets

# 3 - Train and Test models on single months ############################################
# split data by months
# data is randomly split 4:1 into training and testing sets
# training set itself is split 4:1 into training and validation sets
print(df['test_date'])
df['test_date'] = pd.to_datetime(df['test_date'])
df['test_date'] = df['test_date'].dt.to_period('M')

for month in df.test_date.unique():
    outfile_prefix = f'preprocessed_{month}_'

    train_df, test_df = train_test_split(df[df['test_date'] == month], test_size=0.2, random_state=123)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=123)

    # remove date
    test_df = test_df.drop(['test_date'], axis=1).astype(float)
    train_df = train_df.drop(['test_date'], axis=1).astype(float)
    val_df = val_df.drop(['test_date'], axis=1).astype(float)

    test_df.to_csv(pathprefix + 'data/months/' + outfile_prefix + 'test.csv', index=False)
    train_df.to_csv(pathprefix + 'data/months/' + outfile_prefix + 'train.csv', index=False)
    val_df.to_csv(pathprefix + 'data/months/' + outfile_prefix + 'val.csv', index=False)
