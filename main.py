import pandas as pd

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
keys = data_frame.keys()
print(keys)

#checking dataframe info
print("\n ========info========= \n")

info = data_frame.info()
print(info)


