# Importing necessary libraries
import socket as soc
import numpy as np
import pandas as pd
import struct
import sys
from sklearn.model_selection import train_test_split

def data_pre_processing(file):
    """
    Removing the null rows from the data in the csv file,
    splitting the data into training and testing datasets,
    and storing the training and testing datasets into csv files.
    """
    # Reading the csv file
    data_read = pd.read_csv(file)

    # Dropping the rows that are null
    modified_data = data_read.dropna()

    # Checking the number of the rows that are null
    # null_count = data_read.isnull().sum()
    # print(null_count)
    
    # Assigning the rows(x-values) and columns(y-values) for the data without missing values
    independent_var = modified_data[['longitude', 'latitude', 'housing_median_age' , 'total_rooms', 'total_bedrooms',
                                     'population', 'households', 'median_income', 'ocean_proximity']]
    dependent_var = modified_data['median_house_value']

    # Splitting the data into training and testing datasets
    x_train, x_test, y_train, y_test = train_test_split(independent_var, dependent_var, test_size=0.25, random_state=42)

    # Storing the training and testing datasets into csv files
    x_train.to_csv('x_train.csv')
    x_test.to_csv('x_test.csv')
    y_train.to_csv('y_train.csv')
    y_test.to_csv('y_test.csv')

    # Returning the training and testing datasets
    return x_train, x_test, y_train, y_test

def csv_concat(train_1, test_1, train_2, test_2):
    """
    Combining the 'train' files(x_train.csv and y_train.csv)
    and the 'test' files(x_test.csv and y_test.csv)
    """
    # Reading the csv files
    x_train = pd.read_csv(train_1)
    y_train = pd.read_csv(train_2)
    x_test = pd.read_csv(test_1)
    y_test = pd.read_csv(test_2)

    # Merging the 'train' files
    train_data = pd.concat([x_train, y_train], axis=1)
    train_data.to_csv('train.csv')

    # Merging the 'test' files
    test_data = pd.concat([x_test, y_test], axis=1)
    test_data.to_csv('test.csv')

# Writing the main function
if __name__ == '__main__':
    # Storing the input file name
    INPUT_FILE = 'housing.csv'

    # Calling the function
    # data_pre_processing(INPUT_FILE)
    csv_concat('x_train.csv', 'x_test.csv', 'y_train.csv', 'y_test.csv')
