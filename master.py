# Libraries for doing multicasting
import socket as soc
import struct
import sys
import time

# Libraries for data pre-processing
import pandas as pd
from sklearn.model_selection import train_test_split

# Storing the input file name
INPUT_FILE = 'housing.csv'
# Defining the multicast group address and port
MULTICAST_GROUP = '224.3.29.71'
SERVER_ADDRESS = ('', 10000)
NODES_COUNT = 3  # the total number of work nodes


def data_pre_processing(file):
    """
    Removing the null rows from the data in the csv file, making sure all
    the values are integers, splitting the data into training and testing
    datasets, and storing the training and testing datasets into csv files.
    """
    # Reading the csv file
    data_read = pd.read_csv(file)
    # print(len(data_read))

    # Dropping the rows that are null
    modified_data = data_read.dropna()
    # print(len(modified_data))

    # Changing the values in the 'ocean_proximity' column to integers
    # Uses the conversion: ISLAND = 1, NEAR OCEAN = 2, NEAR BAY = 3, <1H OCEAN
    # = 4, INLAND = 5
    modified_data.loc[modified_data['ocean_proximity']
                      == 'ISLAND', 'ocean_proximity'] = 1
    modified_data.loc[modified_data['ocean_proximity']
                      == 'NEAR OCEAN', 'ocean_proximity'] = 2
    modified_data.loc[modified_data['ocean_proximity']
                      == 'NEAR BAY', 'ocean_proximity'] = 3
    modified_data.loc[modified_data['ocean_proximity']
                      == '<1H OCEAN', 'ocean_proximity'] = 4
    modified_data.loc[modified_data['ocean_proximity']
                      == 'INLAND', 'ocean_proximity'] = 5

    # Assigning the rows(x-values) and columns(y-values) for the data without
    # missing values
    independent_var = modified_data[['longitude',
                                     'latitude',
                                     'housing_median_age',
                                     'total_rooms',
                                     'total_bedrooms',
                                     'population',
                                     'households',
                                     'median_income',
                                     'ocean_proximity']]
    dependent_var = modified_data['median_house_value']

    # Splitting the data into training and testing datasets
    x_train, x_test, y_train, y_test = train_test_split(
        independent_var, dependent_var, test_size=0.25, random_state=42)

    # Merging the x_train and y_train datasets into one 'train' file
    train_data = pd.concat([x_train, y_train], axis=1)
    train_data.to_csv('train.csv', index=False)

    # Merging the x_test and y_test datasets into one 'test' file
    test_data = pd.concat([x_test, y_test], axis=1)
    test_data.to_csv('test.csv',index=False)

    # # Keeping the x_test and y_test datasets separate
    # x_test.to_csv('x_test.csv', index=False)
    # y_test.to_csv('y_test.csv', index=False)


def send_file_multicast(option: int):
    """
    Sends a file(either train/test dataset) to the multicast group
    :param: option - An integer to determine what file to send
    """
    # Initializing the payload file
    # -> This is just a placeholder to not get an error
    payload_file = open('housing.csv', 'rb')

    # Setting what file to send based on the option
    if option == 1:
        payload_file = open('train.csv', 'rb')
    elif option == 2:
        # payload_file = open('x_test.csv', 'rb')
        payload_file = open('test.csv', 'rb')    
    # elif option == 3:
        # payload_file = open('y_test.csv', 'rb')

    # Opening the socket
    print('Opening socket...', file=sys.stderr)
    # Opening a socket to a UDP socket
    set_socket = soc.socket(soc.AF_INET, soc.SOCK_DGRAM)
    # Setting the time-to-live for file to 1 so they don't go past the local
    # network segment
    file_lifespan = struct.pack('b', 1)
    set_socket.setsockopt(soc.IPPROTO_IP, soc.IP_MULTICAST_TTL, file_lifespan)

    # Reading the file
    payload = payload_file.read(10248)
    count = 0  # keeping track of how many times do we need to send all data

    # Sending the file to the multicast group
    while payload:
        set_socket.sendto(payload, (MULTICAST_GROUP, 10000))
        payload = payload_file.read(10248)
        count += 1
        time.sleep(0.05)
    print(count)
    # Indicating that the file has been sent
    print('File sent!', file=sys.stderr)
    set_socket.sendto('File sent!'.encode(), (MULTICAST_GROUP, 10000))

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
    # Creating a socket
    print('Creating socket...', file=sys.stderr)
    # Opening the socket to a UDP socket
    node_to_master_socket = soc.socket(soc.AF_INET, soc.SOCK_DGRAM)
    # Binding the socket to the server address
    node_to_master_socket.bind(SERVER_ADDRESS)

    # Telling the operating system to add the socket to the multicast group
    # on all interfaces.
    print('Adding socket to multicast group...', file=sys.stderr)
    group = soc.inet_aton(MULTICAST_GROUP)
    mreq = struct.pack('4sL', group, soc.INADDR_ANY)
    node_to_master_socket.setsockopt(
        soc.IPPROTO_IP, soc.IP_ADD_MEMBERSHIP, mreq)

    # Initilizing the count of nodes that have completed model training
    ackcount = 0
    # Initilizing the count of nodes that have completed model predicting
    predcount = 0

    # Listening to the multicast group
    while ackcount < NODES_COUNT:
        print('\nWaiting to receive message...', file=sys.stderr)
        # Handle multicast data received on multicast_socket
        data, address = node_to_master_socket.recvfrom(1024)

        # Decoding the bytes to translate it to a string and the send time
        decoded_data = data.decode()

        # # Break the while loop when socket receive data
        # if data:
        #     break

        # Indicating the data has been received
        print('Received successfully from node!', file=sys.stderr)

        # Checking the prompt
        if decoded_data == 'ack':
            ackcount += 1
            print(f'{ackcount} nodes training completed successfully!') 
            if ackcount >= NODES_COUNT:
                time.sleep(0.1)
                send_file_multicast(2)
        # elif decoded_data == 'done':
        #     done_count += 1
        #     print(f'{ackcount} node(s) completed training successfully!')
        #     if done_count >= NODES_COUNT:
        #         time.sleep(0.1)
        #         # Sending the y_test dataset to the multicast group to check for accuracy
        #         send_file_multicast(3)
        elif decoded_data == 'accuracy':
            sys.exit(0)

# Writing the main function
if __name__ == '__main__':
    # Doing data pre-processing
    data_pre_processing(INPUT_FILE)
    # Sending the training dataset to the multicast group
    send_file_multicast(1)
    # Activating the receiver to listen to the multicast group
    receiver()
