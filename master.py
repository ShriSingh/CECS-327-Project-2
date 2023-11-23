# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split

def data_pre_processing(file):
    """
    Removing the null rows from the data in the csv file
    and split the data into training and testing datasets
    """
    # Reading the csv file
    data_read = pd.read_csv(file)

    # Dropping the rows that are null
    modified_data = data_read.dropna()

    # Checking the number of the rows that are null
    # null_count = data_read.isnull().sum()
    # print(null_count)
    
    # Assigning the rows and columsn for the data with missing values
    dropped_rows = modified_data[['longitude', 'latitude', 'housing_median_age' , 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value', 'ocean_proximity']]
    dropped_columns = modified_data['total_bedrooms'] # Since this is the only column with missing values

    # Splitting the data into training and testing datasets
    x_train, x_test, y_train, y_test = train_test_split(dropped_rows, dropped_columns, test_size=0.1)


    # Writing the pre-processed data to a new csv file
    # data_read.to_csv('housing_pre_processed.csv', index=False)


# Writing the main function
if __name__ == '__main__':
    # Storing the input file name
    INPUT_FILE = 'housing.csv'

    # Calling the function
    data_pre_processing(INPUT_FILE)
