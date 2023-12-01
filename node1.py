import socket
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import struct
import sys
from io import StringIO

# Defining the multicast group address and port
multicast_group = '224.3.29.71'
server_address = ('', 10000)
regressor = LinearRegression()


# Reference: https://pymotw.com/2/socket/multicast.html
def listen(n): #n == 1: for training 2: for testing
    """
    A function to receive and acknowledge a message from a multicast group
    """

    # Create the socket
    multicast_node_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Bind to the server address
    multicast_node_socket.bind(server_address)

    # Tell the operating system to add the socket to the multicast group
    # on all interfaces.
    group = socket.inet_aton(multicast_group)
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

#training the linear regression model
def training(train_data):
    # Import data from train_data.csv file into a DataFrame
    df = pd.read_csv(StringIO(train_data)) #can change to io.StringIO(train_data), sep="," if this doesnt work
    X_train = df.iloc[:,:-1].values 
    y_train = df.iloc[:,-1].values

    # Build a linear regression model with X_train, y_train
    regressor.fit(X_train ,y_train) 
    

def testing(test_data):
    # Import data from train_data.csv file into a DataFrame
    df = pd.read_csv(StringIO(test_data))
    X_test = df.iloc[:,:].values 

    # Predict the test set results y_pred (y_hat) from X_test
    y_pred = regressor.predict(X_test)
    
    return y_pred

if __name__ == '__main__':
    listen()
    # training()

    listen()
    # testing()


    
