"""
Client of the federated learning model.


--------------------
NOTE:
    1. Overall, we have 3 message formats to design:
        1. Sending Hand-shake message TO server
        2. Global model message FROM server TO client
        3. Update model message FROM client TO server
        4. (Maybe) Design multiple thread to spilt MODEL into several pieces message
        and send TO server
    2. We need to set up NEW connection to Server:
        1. init TCP socket and bind it with Server socket with PORT number 6000
"""

# import modules
import sys
import os
import json
import torch

# Global variables
ID = sys.argv[1]  # Client ID
PORT = int(sys.argv[2])  # Listen port. From 6001 to 6005
MODEL_OPTION = int(sys.argv[3])  # Model Option. 0 is for GD, 1 is for Mini-Batch GD
T = 100  # Total iterations
E = 2  # Epoch per training iteration


# noinspection PyTypeChecker
def init_client(client_id=ID):
    """
    TODO:
        1. Loading the dataset with given client id from command line argument
        2. Find an efficient way to represent the model in this program
    :return: The model
    """
    train_path = os.path.join("FLdata", "train", "mnist_train_" + str(client_id) + ".json")
    test_path = os.path.join("FLdata", "test", "mnist_test_" + str(client_id) + ".json")
    train_data = {}
    test_data = {}

    with open(os.path.join(train_path), "r") as f_train:
        train = json.load(f_train)
        train_data.update(train['user_data'])
    with open(os.path.join(test_path), "r") as f_test:
        test = json.load(f_test)
        test_data.update(test['user_data'])

    image_train = train_data['0']['x']
    label_train = train_data['0']['y']
    image_test = test_data['0']['x']
    label_test = test_data['0']['y']

    image_train = torch.Tensor(image_train).view(-1, 1, 28, 28).type(torch.float32)
    label_train = torch.Tensor(label_train).type(torch.int64)
    image_test = torch.Tensor(image_test).view(-1, 1, 28, 28).type(torch.float32)
    label_test = torch.Tensor(label_test).type(torch.int64)
    train_samples, test_samples = len(label_train), len(label_test)

    return image_train, label_train, image_test, label_test, train_samples, test_samples


def hand_shaking_to_server():
    """
    TODO:
        1. Initialise socket and set up connection to the Server
            * sending hand-shaking message to Server
    :return: The socket
    """


def generate_log():
    """
    TODO:
        1. Writing the training loss and accuracy of the global model to the FILE
            * Generate it at each communication round
            * File name format ( client(id)_log.txt )
    :return: None
    """


def display_info():
    """
    TODO:
        1. Displaying the training loss and accuracy to terminal
        2. Message format

                * I am client (id)
                * Receiving new global model
                * Training loss: %f
                * Testing accuracy: (%d)%
                * Local training...
                * Sending new local model

    :return: None
    """


def send_local_model():
    """
    TODO:
        1. Sending local model to the Server
    :return: None
    """


def set_parameters():
    """
    TODO:
        1. Receive a global model from the server, replace its local model with the
            global model
    :return: None
    """


def aggregate_models():
    """
    TODO:
        1. Update the new local model
            * Can be finished in E = 2 local iterations
            * Based on Model OPT to generate it (0 GD, 1 Mini-Batch GD)
            * Batch size could be changed
    :return:
    """


def main():
    """
    The body of Client
    TODO:
        1. Init socket connection
        2. Loading dataset
        3. Setting a loop to listen message from Server
    :return :None
    """
    # Init Socket
    hand_shaking_to_server()

    # Loading dataset
    # image_train, label_train, image_test, label_test, train_samples, test_samples = init_client(ID)

    # TODO: Body of the client
    #       1. Outer loop to keep receiving & sending model
    #       2. Train the model with 2 epochs
    for i in range(1, T):
        # TODO: Listen the server

        # TODO: Train the local model with GD or mini-batch SGD
        for e in range(E):
            pass

        # TODO: Send model to the sever


if __name__ == "__main__":
    # main()
    init_client(ID)
