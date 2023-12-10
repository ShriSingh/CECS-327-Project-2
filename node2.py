"""Support Vector Regression model"""
import socket
import struct
# import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Defining the multicast group address and port
MULTICAST_GROUP = '224.3.29.71'
SERVER_ADDRESS = ('', 10000)
REGRESSOR = SVR(kernel='rbf')


# Reference: https://pymotw.com/2/socket/multicast.html
def listen(n):  # n == 1: for training 2: for testing
    """
    A function to receive and acknowledge a message from a multicast group
    :param n: An integer to indicate the state for the node to either train or test
    """

    # Create the socket
    multicast_node_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Bind to the server address
    multicast_node_socket.bind(SERVER_ADDRESS)

    # Tell the operating system to add the socket to the multicast group
    # on all interfaces.
    group = socket.inet_aton(MULTICAST_GROUP)
    mreq = struct.pack('4sL', group, socket.INADDR_ANY)
    multicast_node_socket.setsockopt(
        socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    # determine which file to save
    if n == 1:
        filename = 'train_data.csv'
    else:
        filename = 'test_data.csv'
    fo = open(filename, "w")

    count = 0  # counting how many times do we need to send all data
    while True:
        print(f'{count}: Waiting to receive message...')
        # Handle multicast data received on multicast_socket
        data, address = multicast_node_socket.recvfrom(10248)

        # Decoding the bytes to translate it to a string and the send time
        decoded_data = data.decode()

        # Creating a new file at server end and writing the data
        if data:
            if decoded_data == 'File sent!':
                break
            elif decoded_data == 'ack':
                continue
            else:
                fo.write(decoded_data)
        else:
            print("uhhh thats not suppposed to happen")
            break
        count += 1

    print('Received successfully from node 2!')

    if n == 1:
        training()
        # let master know it has finished training
        multicast_node_socket.sendto('ack'.encode(), (MULTICAST_GROUP, 10000))
    elif n == 2:
        testing()
        # send 'done' to master so it can send the y_test(actual values) file
        multicast_node_socket.sendto('done'.encode(), (MULTICAST_GROUP, 10000))
    elif n == 3:
        # let master know it has finished calculating accuracy
        multicast_node_socket.sendto(
            'accuracy'.encode(), (MULTICAST_GROUP, 10000))


def training():
    """
    Training the Support Vector Regression model
    """
    # Reading the training data from the master node
    df = pd.read_csv('train_data.csv')

    # print("spliting data to X and y")
    x_train = df.iloc[:, :-1].astype(float)  # convert string to float
    y_train = df.iloc[:, -1].astype(float)

    # Feature Scaling to remove mean and scale to unit variance
    sc_X = StandardScaler()
    sc_Y = StandardScaler()

    x_train = sc_X.fit_transform(x_train)
    y_train = np.array(y_train).reshape(-1, 1)
    y_train = sc_Y.fit_transform(y_train)

    # Fit the SVM model according to the given training data.
    REGRESSOR.fit(x_train, y_train)
    print('Node2: Training completed!')


def testing():
    """
    Getting the test data from the master node and predict the values
    and then calculate the accuracy of the model by calculating the
    r2 score.
    """
    # Import data from train_data.csv file into a DataFrame
    df = pd.read_csv('test_data.csv')
    # Part of the file to predict the values
    x_test = df.iloc[:-1, :-1].astype(float)
    # Part of the file to test the predicted values
    y_test = df.iloc[:-1, -1].astype(float)
    # Save the actual values to a csv file
    y_test.to_csv('y_test.csv', index=False)

    # Predict the test set results y_pred (y_hat) from X_test
    y_pred = REGRESSOR.predict(x_test)
    print('Node2: Prediction completed!')

    # Calculating the R2 score to measure the accuracy of the model
    accuracy = r2_score(y_test, y_pred)
    print(f"The r2 score for the model is {accuracy * 100}%")

    # Convert the predicted values to a dataframe
    y_pred_file = pd.DataFrame(y_pred)
    # Save the predicted values to a csv file
    y_pred_file.to_csv('y_pred.csv', index=False)


if __name__ == '__main__':
    # Listening to the train the model
    listen(1)
    # Listening to test the model and calculate the accuracy
    listen(2)
