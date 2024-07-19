# Flight Price Prediction (EDA + Feature Engineering)

## Overview

This project involves predicting flight prices using historical flight data. The process includes exploratory data analysis (EDA) and feature engineering to prepare the data for machine learning models.

## Setup

Ensure you have the required libraries installed. You can install them using pip:


`pip install pandas numpy matplotlib seaborn scikit-learn openpyxl `

## Importing Libraries
```python 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```
## LOAD DATA

 <b>Load train and test datasets</b> <br>
 ```python 
train_df = pd.read_excel('Data_Train.xlsx')
test_df = pd.read_excel('Test_Set.xlsx')
```
<b>Display the first few rows of each dataset</b> <br>
```python 
train_df.head()
test_df.head()
```

## COMBINING DATASET

<b>Combine train and test datasets</b><br>

`final_df = pd.concat([train_df, test_df], ignore_index=True)`

# Display basic information and initial rows
```python 
final_df.head()
final_df.tail()
final_df.info()
```

## FEATURE ENGINEERING

# Extracting Data Components:

#Split 'Date_of_Journey' into 'Date', 'month', and 'year' <br>
```python 
final_df['Date'] = final_df['Date_of_Journey'].apply(lambda x: x.split('/')[0])
final_df['month'] = final_df['Date_of_Journey'].apply(lambda x: x.split('/')[1])
final_df['year'] = final_df['Date_of_Journey'].apply(lambda x: x.split('/')[2])
```
#Convert to integer type <br>
```python 
final_df['Date'] = final_df['Date'].astype(int)
final_df['month'] = final_df['month'].astype(int)
final_df['year'] = final_df['year'].astype(int)
```
#Drop the original 'Date_of_Journey' column <br>

`final_df.drop('Date_of_Journey', axis=1, inplace=True)`

## EXTRACTING ARRIVAL TIME

#Extract hour and minute from 'Arrival_Time' <br>
```python 
final_df['Arrival_hour'] = final_df['Arrival_Time'].apply(lambda x: x.split(' ')[0].split(':')[0])
final_df['Arrival_min'] = final_df['Arrival_Time'].apply(lambda x: x.split(' ')[0].split(':')[1])
```
#Convert to integer type <br>
```python 
final_df['Arrival_hour'] = final_df['Arrival_hour'].astype(int)
final_df['Arrival_min'] = final_df['Arrival_min'].astype(int)
```
#Drop the original 'Arrival_Time' column <br>
```python 
final_df.drop('Arrival_Time', axis=1, inplace=True)
```
## EXTRACTING DEPARTURE TIME

#Extract hour and minute from 'Dep_Time' <br>
```python 
final_df['Dep_hour'] = final_df['Dep_Time'].apply(lambda x: x.split(':')[0])
final_df['Dep_min'] = final_df['Dep_Time'].apply(lambda x: x.split(':')[1])
```
#Convert to integer type  <br>
```python 
final_df['Dep_hour'] = final_df['Dep_hour'].astype(int)
final_df['Dep_min'] = final_df['Dep_min'].astype(int)
```
#Drop the original 'Dep_Time' column  <br>

`final_df.drop('Dep_Time', axis=1, inplace=True)`

## ENCODING CATEGORICAL VARIABLES

#Map 'Total_Stops' to numerical values
<br>
`final_df['Total_Stops'] = final_df['Total_Stops'].map({'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4, 'nan': 1})`

#Drop 'Route' column <br>
`final_df.drop('Route', axis=1, inplace=True)`

#Handle 'Duration' column <br>
```python 
final_df['Duration_hour'] = final_df['Duration'].str.split(' ').str[0].str.split('h').str[0]
final_df.drop(6474, axis=0, inplace=True)
final_df.drop(13343, axis=0, inplace=True)
final_df['Duration_hour'] = final_df['Duration_hour'].astype(int)
final_df.drop('Duration', axis=1, inplace=True)
```
## LABEL ENCODING

# from sklearn.preprocessing import LabelEncoder

#Initialize the label encoder <br>
```python 
labelencoder = LabelEncoder()
```
#Encode categorical columns <br>
```python 
final_df['Airline'] = labelencoder.fit_transform(final_df['Airline'])
final_df['Source'] = labelencoder.fit_transform(final_df['Source'])
final_df['Destination'] = labelencoder.fit_transform(final_df['Destination'])
final_df['Additional_Info'] = labelencoder.fit_transform(final_df['Additional_Info'])
```
## ONE HOT ENCODING

from sklearn.preprocessing import OneHotEncoder

#Initialize the one-hot encoder <br>
```python 
ohe = OneHotEncoder(sparse=False)
```
#Apply one-hot encoding to categorical columns <br>
```python 
one_hot_encoded = ohe.fit_transform(final_df[['Airline', 'Source', 'Destination']])
final_df_encoded = pd.DataFrame(one_hot_encoded, columns=ohe.get_feature_names_out(['Airline', 'Source', 'Destination']))
```

#Concatenate with the original DataFrame <br>
```python 
final_df = pd.concat([final_df, final_df_encoded], axis=1)
```
#Drop original categorical columns <br>
```python 
final_df.drop(['Airline', 'Source', 'Destination'], axis=1, inplace=True)
```
## Next Steps

Model Training: Use the processed data to train machine learning models for flight price prediction.<br>
Evaluation: Evaluate model performance and tune hyperparameters as necessary.

## Contributing

Feel free to fork the repository, make improvements, and submit pull requests.




