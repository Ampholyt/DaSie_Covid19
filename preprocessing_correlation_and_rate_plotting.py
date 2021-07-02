import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

"""
Is able to plot correlation matrices and rate plots for every feature as well as the mean for every month
(Without saving by default)
"""
fileprefix = 'data/'
df = pd.read_csv(fileprefix + 'corona_tested_individuals_ver_00143.csv', sep=',')

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


df.columns = ["test_date", "Cough", "Fever", "Sore_throat", "Shortness_of_breath",
              "Headache", "corona_result", "Age_60+", "Male", "Contact_with_confirmed"]


# positive rate for every feature and every month
def most_common(lst):
    return max(set(lst), key=lst.count)

# df = pd.read_csv('corrected_152.csv', sep=',')

df['test_date'] = pd.to_datetime(df['test_date'])
df['test_date'] = df['test_date'].dt.to_period('M')

month_list = list(df.test_date.unique())
month_list.reverse()


#correlation matrix



for month in month_list:
    cur_month_df = df[df["test_date"] == month]
    corrMatrix = cur_month_df.corr()
    fig, ax = plt.subplots(figsize=(30, 10))
    sn.heatmap(corrMatrix, annot=True)
    plt.title(f'Correlation Matrix of {month}')
    # plt.savefig(f'C:/Users/Saloberg/Desktop/FU/Master/sem_2/Data-Science/Software-project/light-gbm/images/rates/correlation_{month}.png', dpi=100)
    plt.show()


# rate figure(s)
fig, ax = plt.subplots(figsize=(30, 10))
months = []
for i in month_list:
    tmp = i.strftime('%Y-%b')
    months.append(tmp)
columns = ["Cough", "Fever", "Sore_throat", "Shortness_of_breath","Headache", "corona_result", "Age_60+", "Male", "Contact_with_confirmed"]
for column in columns:
    if column == "Male":
        continue
    pos_rates = []
    for month in months:
        cur_month_df = df[df["test_date"] == month]
        tru = cur_month_df[cur_month_df[column] == True]
        pos = len(tru)
        sum = len(cur_month_df)
        if pos != 0:
            rate = pos / sum
        else:
            rate = 0
        pos_rates.append(rate)
    # every feature in different plot
    # fig_iter, ax_iter = plt.subplots(figsize=(30, 10))
    # ax_iter.plot(months, pos_rates, label=f'{column}')
    # plt.xlabel('Period of Time')
    # plt.ylabel('Symptom rate')
    # plt.legend()
    # plt.show()
    # plt.savefig(f'C:/Users/Saloberg/Desktop/FU/Master/sem_2/Data-Science/Software-project/light-gbm/images/rates/feature_{column}_pos_rates.png')
    # every line in one plot
    ax.plot(months, pos_rates, label=f'{column}')

# every line in one plot
plt.xlabel('Period of Time')
plt.ylabel('Symptom rate')
plt.legend()
# plt.savefig(f'C:/Users/Saloberg/Desktop/FU/Master/sem_2/Data-Science/Software-project/light-gbm/images/rates/lines_all_pos_rates.png')
plt.show()

# # mean_rates_per_month:
mean_rates_per_month = []
control_rate = []
for month in months:
    pos_rates = []
    pos_vals = 0
    sum_vals = 0
    for column in columns:
        cur_month_df = df[df["test_date"] == month] # num of testings in the month
        tru = cur_month_df[cur_month_df[column] == True] # num of true values from this feature
        pos = len(tru)
        sum = len(cur_month_df)
        pos_vals += pos # sum all true values and all testings
        sum_vals += sum
    print(f'{month}: mean positive value over all features: \n', pos_vals/sum_vals)
    mean_rates_per_month.append(pos_vals/sum_vals) # calculate the rate of positive values for this month


fig, ax = plt.subplots(figsize=(30, 10))
plt.title("Rates of positive values per month")
plt.scatter(months, mean_rates_per_month)
plt.xlabel('Period of Time')
plt.ylabel('Rate of positive values over all features')
plt.savefig(f'C:/Users/Saloberg/Desktop/FU/Master/sem_2/Data-Science/Software-project/light-gbm/images/rates/mean_rates_per_month.png')
plt.show()
