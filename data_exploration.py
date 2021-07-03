# pandas value counts:
import pandas as pd
from sklearn.model_selection import train_test_split

fileprefix = 'data/'

test_df = pd.read_csv(fileprefix + 'corona_tested_individuals_ver_00143.csv', sep=',')

# train_df, test_df = train_test_split(df, test_size=0.2, random_state=123)
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

test_df = test_df.replace(translations)


# Replicate the data structure from the paper:

# drop all obs where corona result is not binary
test_df = test_df[test_df['corona_result'] != 2]
test_df = test_df.dropna(subset=['gender'])

# unnecessary on raw data, missing data already nan
# # replace 'None' with np.nan so lgbm can work with it
# df['cough'].replace('None', np.nan, inplace=True)
# df['fever'].replace('None', np.nan, inplace=True)
# df['sore_throat'].replace('None', np.nan, inplace=True)
# df['shortness_of_breath'].replace('None', np.nan, inplace=True)
# df['head_ache'].replace('None', np.nan, inplace=True)
# df['age_60_and_above'].replace('None', np.nan, inplace=True)

# replace categories with numbers


test_df.columns = ["test_date", "Cough", "Fever", "Sore_throat", "Shortness_of_breath",
              "Headache", "corona_result", "Age_60+", "Male", "Contact_with_confirmed"]

# test_df = pd.read_csv('data/tmp_test.csv', sep=',')

df_2020 = test_df[test_df['test_date'].apply(lambda x: x.split("-")[0] == '2020')]
df_2021 = test_df[test_df['test_date'].apply(lambda x: x.split("-")[0] == '2021')]

print(df_2020)
print(df_2020.columns)
dict_of_lists = {
    "Cough": [], "Fever": [], "Sore_throat": [], "Shortness_of_breath": [],
    "Headache": [], "corona_result": [], "Age_60+": [], "Male": [], "Contact_with_confirmed": []
}
cols = ["Cough", "Fever", "Sore_throat", "Shortness_of_breath","Headache", "corona_result", "Age_60+", "Male", "Contact_with_confirmed"]
for i in range(3,13):
    df_2020_temp = df_2020[df_2020["test_date"].apply(lambda x: int(x.split("-")[1]) == i)]
    for column in cols:
        print("Values for month: " + str(i))
        print(df_2020_temp[column].value_counts(), "\n")
        series = df_2020_temp[column].value_counts()
        list = series.tolist()
        dict_of_lists[column].append(list)

print(dict_of_lists)

dict_df = pd.DataFrame.from_dict(dict_of_lists)

dict_df.to_csv("full_data_summary.csv", index=False)


# first date: 2020-03-11,
# last date: 2021-05-09,

# ﻿test_date,cough,fever,sore_throat,shortness_of_breath,head_ache,corona_result,age_60_and_above,gender,test_indication
# 2021-05-09,0,0,0,0,0,שלילי,NULL,NULL,Other
# 2021-05-09,0,0,0,0,0,שלילי,NULL,NULL,Other
# 2021-05-09,0,0,0,0,0,שלילי,NULL,NULL,Other
# 2021-05-09,0,0,0,0,0,שלילי,NULL,NULL,Other
# 2021-05-09,0,0,0,0,0,שלילי,NULL,NULL,Other
# 2021-05-09,0,0,0,0,0,שלילי,NULL,NULL,Other
# 2021-05-09,0,0,0,0,0,שלילי,NULL,NULL,Other
# 2021-05-09,0,0,0,0,0,שלילי,NULL,NULL,Other
# 2021-05-09,0,0,0,0,0,שלילי,NULL,NULL,Other
