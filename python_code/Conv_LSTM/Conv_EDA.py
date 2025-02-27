'''
Author: Jacob Sullivan
Date: 2/27/25
Status: Unfinished
Description: The code below is just a simple EDA of the sample csv file we all decided to use. I go through it 
showing the total number of lines, dropping columns that we do not need, and checking values either if their not needed
or the data type to ensure we use the data correctly.
'''
#import libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, TimeDistributed, Dense, Dropout

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split, RandomizedSearchCV, learning_curve, cross_val_score
from sklearn.preprocessing import LabelEncoder

#Visual libraries
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import seaborn as sns

#Load Waterfowl Data of Aug-Sep
csv="ShortTermSetData(Aug-Sept).csv"

#Read data
#dataset=pd.read_csv(csv)
dataset = pd.read_csv(csv, on_bad_lines='skip', engine='python')
dataset.head()

#Identifying data types and names of each column
dataset.dtypes

#Checking for any null values
dataset.isnull().sum()

#Checking out data distribution
for x in dataset.columns:
  print(f"Column: {x}")
  print("-"*20)
  print(dataset[x].value_counts())
  print("")

# Drop irrelevant columns and verification,  if they exist
columns_to_drop = ["battery-charge-percent", "battery-charging-current", "gps-time-to-fix",
                   "orn:transmission-protocol", "tag-voltage", "sensor-type", "acceleration-raw-x",
                   "acceleration-raw-y", "acceleration-raw-z", "ground-speed", "gls:light-level", "study-name"]

for column in columns_to_drop:
    if column in dataset.columns:  # Check if column exists before dropping
        dataset = dataset.drop(columns=[column])
    # else:
    #     print(f"Column '{column}' not found in the DataFrame.") Print a message if the column is not found

dataset.columns #verification

#check data
dataset.head()