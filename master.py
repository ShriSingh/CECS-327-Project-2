# Libraries for data pre-processing
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# Libraries for doing multicasting
import socket as soc
import struct
import sys
import time

# Storing the input file name
INPUT_FILE = 'housing.csv'
# Defining the multicast group address and port
MULTICAST_GROUP = '224.3.29.71'
SERVER_ADDRESS = ('', 10000)

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
    
    # Assigning the rows(x-values) and columns(y-values) for the data without missing values
    independent_var = modified_data[['longitude', 'latitude', 'housing_median_age' , 'total_rooms', 'total_bedrooms',
                                     'population', 'households', 'median_income', 'ocean_proximity']]
    dependent_var = modified_data['median_house_value']

    # Splitting the data into training and testing datasets
    x_train, x_test, y_train, y_test = train_test_split(independent_var, dependent_var, test_size=0.25, random_state=42)

    # Merging the x_train and y_train datasets into one 'train' file
    train_data = pd.concat([x_train, y_train], axis=1)
    train_data.to_csv('train.csv')

    # Merging the x_test and y_test datasets into one 'test' file
    test_data = pd.concat([x_test, y_test], axis=1)
    test_data.to_csv('test.csv')

def send_file_multicast(option: int):
    """
    Sends a file(either train/test dataset) to the multicast group
    :param: option - An integer to determine what file to send
    """
    # Setting what file to send based on the option
    if option == 1:
        payload_file = open('train.csv', 'rb')
    elif option == 2:
        payload_file = open('test.csv', 'rb')
    else:
        print('Invalid option!')

    # Opening a socket to a UDP socket
    set_socket = soc.socket(soc.AF_INET, soc.SOCK_DGRAM)
    # Setting the time-to-live for file to 1 so they don't go past the local network segment
    file_lifespan = struct.pack('b', 1)
    set_socket.setsockopt(soc.IPPROTO_IP, soc.IP_MULTICAST_TTL, file_lifespan)

    # Reading the file
    payload = payload_file.read(1024)

    # Sending the file to the multicast group
    while payload:
        set_socket.sendto(payload, (MULTICAST_GROUP, 10000))
        payload = payload_file.read(1024)
        time.sleep(0.01)

    # Closing the file
    payload_file.close()
    # Closing the socket
    print('Closing socket...', file=sys.stderr)
    set_socket.close()

def receiver():
    """
    Listens to the multicast group for the prompt.
    Based on the prompt, instructs master to send the 
    training or testing dataset.
    """
    pass


# Writing the main function
if __name__ == '__main__':
    # Doing data pre-processing
    data_pre_processing(INPUT_FILE)
    # Sending the training dataset to the multicast group
    send_file_multicast(1)
    # Activating the receiver to listen to the multicast group
    receiver()

