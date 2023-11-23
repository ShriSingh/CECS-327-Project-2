# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split

def data_pre_processing(input):
    """
    Doing pre-processing of the data in the csv file
    """
    # Reading the csv file
    data_read = pd.read_csv(input)

    # Dropping the rows that are null
    data_read = data_read.dropna()

    # Checking the number of the rows that are null
    # null_count = data_read.isnull().sum()
    # print(null_count)

    # Writing the pre-processed data to a new csv file
    # data_read.to_csv('housing_pre_processed.csv', index=False)


# Writing the main function
if __name__ == '__main__':
    # Storing the input file name
    INPUT_FILE = 'housing.csv'

    # Calling the function
    data_pre_processing(INPUT_FILE)
