import socket
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import struct
import sys
from io import StringIO

# Defining the multicast group address and port
MULTICAST_GROUP = '224.3.29.71'
SERVER_ADDRESS = ('', 10000)
REGRESSOR = LinearRegression()


# Reference: https://pymotw.com/2/socket/multicast.html
def listen(n): #n == 1: for training 2: for testing
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
    multicast_node_socket.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    while True:
        print('\nWaiting to receive message...', file=sys.stderr)
        # Handle multicast data received on multicast_socket
        data, address = multicast_node_socket.recvfrom(1024)

        
        # Decoding the bytes to translate it to a string and the send time
        decoded_data = data.decode()

        # Break the while loop when socket receive data
        if data: 
            break
        
    # # Creating a new file at server end and writing the data 
    # filename = 'train_data'+'.csv'
    # fileno = fileno+1
    # fo = open(filename, "w") 
    # while data: 
    #     if not data:
    #         break
    #     else: 
    #         fo.write(decoded_msg) 

    print('Received successfully from node 1!') 
    # fo.close()
    if n == 1:
        training(decoded_data)
        # let master know it has finished training
        multicast_node_socket.sendto(str("ack").encode(),address)
    else:
        y_pred = testing(decoded_data)
        # send the predicted result to master
        multicast_node_socket.sendto(str(y_pred).encode(),address)

def training(train_data):
    """
    Training the linear regression model
    :param train_data: the training data
    """
    train_data =  train_data[1:]
    print(train_data)
    # Import data from train_data.csv file into a DataFrame
    df = pd.DataFrame([train_data.split(',') for x in train_data.split('\n')[1:]],  #spliting columns by ',', rows by '\n'
                           columns=[x for x in train_data.split('\n')[0].split(';')]) #using first row as column name
    print(df)
    # print("spliting data to X and y")
    X_train = df.iloc[:,:-1].values 
    y_train = df.iloc[:,-1].values

    print(X_train)
    print(y_train)
    # Build a linear regression model with X_train, y_train
    REGRESSOR.fit(X_train ,y_train) 
    print('Received successfully from master!') 
    

def testing(test_data):
    # Import data from train_data.csv file into a DataFrame
    df = pd.read_csv(StringIO(test_data))
    X_test = df.iloc[:,:].values 

    # Predict the test set results y_pred (y_hat) from X_test
    y_pred = REGRESSOR.predict(X_test)
    
    return y_pred

if __name__ == '__main__':
    listen(1)
    # training()

    listen(2)
    # testing()


    
