import socket
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

def listen():
    #listen and receive file from master https://www.geeksforgeeks.org/file-transfer-using-tcp-socket-in-python/
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    host = '127.0.0.1'
    port = 8080
    sock.bind((host, port)) #can be changed later
    sock.listen() 
    conn = sock.accept() 
    data = conn.recv(1024).decode() 
  
    if not data: 
        print("Something is wrong =[")
        return -1
    # Creating a new file at server end and writing the data 
    filename = 'train_data'+'.csv'
    fileno = fileno+1
    fo = open(filename, "w") 
    while data: 
        if not data: 
            break
        else: 
            fo.write(data) 
            data = conn[0].recv(1024).decode() 

    print('Received successfully from node 1!') 
    fo.close()

#training the linear regression model
def training():
    # Import data from train_data.csv file into a DataFrame
    df = pd.read_csv("train_data.csv")
    X_train = df.iloc[:,:-1].values 
    y_train = df.iloc[:,-1].values

    # Build a linear regression model with X_train, y_train
    regressor.fit(X_train ,y_train) 

def testing():
    # Import data from train_data.csv file into a DataFrame
    df = pd.read_csv("test_data.csv")
    X_test = df.iloc[:,:].values 

    # Predict the test set results y_pred (y_hat) from X_test
    y_pred = regressor.predict(X_test)
    
    # send y_pred to master

if __name__ == '__main__':
    # listen()
    training()
    # let master know it has finished training
    # listen()
    testing()


    
