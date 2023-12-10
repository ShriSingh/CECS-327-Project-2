"""Polynomial Regression Model"""
import socket
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import struct
import sys
# Libraries for accuracy measurement
from sklearn import metrics
from sklearn.metrics import accuracy_score

# Defining the multicast group address and port
MULTICAST_GROUP = '224.3.29.71'
SERVER_ADDRESS = ('', 10000)
REGRESSOR = LinearRegression()
# create an instance of PolynomialFeatures with degree=4
POLY = PolynomialFeatures(degree=4)

# Reference: https://pymotw.com/2/socket/multicast.html


def listen(n):  # n == 1: for training
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
        # print(decoded_data)

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

    print('Received successfully from node 3!')
    # fo.close()
    if n == 1:
        training()
        # let master know it has finished training
        multicast_node_socket.sendto('ack'.encode(), (MULTICAST_GROUP, 10000))
    else:
        # y_pred = testing()
        # send 'done' to master so it can send the y_test(actual values) file
        multicast_node_socket.sendto('done'.encode(),(MULTICAST_GROUP, 10000))


def training():
    """
    Training the polynomial regression model
    :param train_data: the training data
    """

    df = pd.read_csv('train_data.csv')
    # Import data from train_data.csv file into a DataFrame
    # df = pd.DataFrame([x.split(',') for x in train_data.split('\n')[1:]],  #spliting columns by ',', rows by '\n'
    # columns=[x for x in train_data.split('\n')[0].split(',')]) #using first
    # row as column name

    # print("spliting data to X and y")
    x_train = df.iloc[:, :-1].astype(float)  # convert string to float
    y_train = df.iloc[:, -1].astype(float)

    # print(X_train)
    # print(y_train)
    # transform the features using fit_transform method of poly
    x_poly = POLY.fit_transform(x_train, y_train)

    # fit
    REGRESSOR.fit(x_poly, y_train)

    print('Node3: Training completed!')


def testing():
    # Import data from train_data.csv file into a DataFrame
    df = pd.read_csv('test_data.csv')
    x_test = df.iloc[:-1, :].astype(float)

    # Predict the test set results y_pred (y_hat) from X_test
    y_pred = REGRESSOR.predict(POLY.transform(x_test))
    print('Node3: Prediction completed!')
    # print(y_pred)

    return y_pred


def accuracy_measurement(prediction):
    """
    Figures out how accurate the model is by calculating
    the percentage of correct predictions from the y-test(actual prices)
    dataset. Converts the string of predicted values into a dataframe and
    calculates the percentage of correct predictions.
    :param result: The predicted values sent from the node
    """
    # Reading the file with actual price values
    actual_prices = pd.read_csv('y_test.csv')

    # Calculating the percentage of correct predictions
    accuracy = accuracy_score(actual_prices, prediction)

    # Printing the accuracy
    print(f"The accuracy of the model is {accuracy * 100}%")


if __name__ == '__main__':
    listen(1)
    listen(2)
    # Storing the predicted values
    y_pred_file = testing()
    # Calculating the accuracy of the model
    accuracy_measurement(y_pred_file)
