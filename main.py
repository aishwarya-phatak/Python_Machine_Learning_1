import pandas as pd
import sklearn
from pandas.core.interchange import column

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#implementing ML by using house price prediction dataset
# absolute path ----  /Users/vishaljagtap/Desktop/Aishwarya/AI_May_2026/Python_Machine_Learning/dataset/Housing.csv
# path from content root ---- dataset/Housing.csv

# read the csv file
#converting csv to data frame
print("\n ========readcsv========= \n")
data_frame = pd.read_csv("dataset/Housing.csv")
print(data_frame)

#checking dataframe
print("\n ========head========= \n")
head_values = data_frame.head()
print(head_values)

#checking dataframe keys
print("\n ========keys========= \n")
keys_df = data_frame.keys()
print(keys_df)

#checking dataframe info
print("\n ========info========= \n")

info_df = data_frame.info()
print(info_df)

#checking dataframe description
print("\n ========describe========= \n")

describe_df = data_frame.describe()
print(describe_df)

#checking dataframe for null values
print("\n ========isnull========= \n")

null_check_of_df = data_frame.isnull()
print(null_check_of_df)

print("\n======================\n")
print(data_frame.isnull().sum())

print("\n ===========check and drop the rows where values are missing===========\n")
print(data_frame.dropna())

print("\n==========Encoding===========\n")
encoded_df = pd.get_dummies(data_frame)

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

print(encoded_df)

x = encoded_df.drop(columns = ['price'])
y = encoded_df['price']

X_train, X_test, y_train, y_test = train_test_split(x,
                 y,
                 test_size=0.2,
                 train_size=0.8,
                 random_state=42)

print("\n ===========X_train========= \n")
print(X_train)
print("\n ===========y_train========= \n")
print(y_train)
print("\n ===========X_test========= \n")
print(X_test)
print("\n ===========y_test========= \n")
print(y_test)
