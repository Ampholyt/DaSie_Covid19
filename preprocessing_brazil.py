import pandas as pd
from sklearn.model_selection import train_test_split
#import numpy as np
#import seaborn as sn
#import matplotlib.pyplot as plt

'''
Preprocess data from Brazilian Paper - Originally collected by public health agency of the city of Campina Grande
Available here: 
https://data.mendeley.com/datasets/b7zcgmmwx4/5

We are using the unbalanced dataset containing both types of tests, as this contains the most cases whil staying 
closest to the raw data. 

The data is already very clean, with all binary features and no missing values. However 0s represent positive responses
and 1s negative i.e. fever=0 notes an instance were someone has fever.
This is extremely unintuitive and we need to switch it.
'''

# pathprefix = ''
pathprefix = 'C:/Users/victo/Desktop/'
path = pathprefix + 'data/both_test_unbalanced.csv'

# read data
df = pd.read_csv(path, low_memory=False)

# assign better col names
df.columns = ["Throat_Pain", "Shortness_of_Breath", "Fever", "Cough", "Headache", "Taste_Disorder",
              "Olfactory_Disorder", "Stuffed_Nose", "Male", "Health_Professional", "Corona_Result"]

# reverse all features except Gender
for col in df.columns.difference(['Male']):
    df[col] = 1 - df[col]

# Test/Train/Validation Split
# data is randomly split 4:1 into training and testing sets
# training set itself is split 4:1 into training and validation sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=123)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=123)

outfile_prefix = 'preprocessed_brazil_'

test_df.to_csv(pathprefix + 'data/' + outfile_prefix + 'test.csv', index=False)
train_df.to_csv(pathprefix + 'data/' + outfile_prefix + 'train.csv', index=False)
val_df.to_csv(pathprefix + 'data/' + outfile_prefix + 'val.csv', index=False)
