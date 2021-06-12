import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sn
import matplotlib.pyplot as plt



# data: https://github.com/nshomron/covidpred/tree/master/data
path = 'C:/Users/victo/Downloads/corona_tested_individuals_ver_00143.csv'

# read data
# df=pd.read_csv('data/corona_tested_individuals_ver_006.english.csv', sep=',')
df = pd.read_csv(path, low_memory=False)


# show nan: #11
# df1 = df[df.isna().any(axis=1)]
# print("All nan values:\n", df1)
# drop nan
df.dropna(inplace=True)
# this drops ~1.4mio Observations out of the 5.7mio

#drop date
df.drop(['test_date'], axis=1, inplace=True)


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
        'Other': 2
    }
}

df = df.replace(translations).astype(int)

# drop all obs where corona result is "other"
# df.drop(df[df.corona_result == 'other'].index, inplace=True)
df = df[df['corona_result'] != 2]

# Split into test, train, validation data
data_train, data_test = train_test_split(df, test_size=0.2)

# split training data again into training and validation data
data_train, data_val = train_test_split(data_train, test_size=0.2)

# correlation matrix
corrMatrix = df.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()




# Save data and category explanations
filename = 'preprocessed_full_'

data_train.to_csv('data/' + filename + 'train.csv', index=False)
data_val.to_csv('data/' + filename + 'val.csv', index=False)
data_test.to_csv('data/' + filename + 'test.csv', index=False)