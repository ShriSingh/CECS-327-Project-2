### About this Project
This is a repository containing Project 2 files for CECS 327. This is project aims to train, test, and measure the accuracy of the models on a distributed system. The master preprocesses the data and sends it to the nodes as training and testing data. The nodes train their model on the training data. Then the nodes split the testing data into 2 parts: one part is used to test the model and the other part is used to measure the accuracy of the model. Next, the nodes test their model and report their the accuracy of their models through the R2 score they calculated.

To run the program:
- Open docker desktop
- Open command prompt or any terminal and reach or `cd` to the directory where the `compose.yaml` file is located
- Run `docker compose up --build` command
- When the master and all nodes display `exited with code 0`, the program has finished running.
    - If the program does not exit, press `ctrl + c` to exit the program and close the containers.