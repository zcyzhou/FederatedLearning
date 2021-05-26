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
import socket
import pickle
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from utils import MLR


# noinspection PyTypeChecker
def init_client(client_id, opt, batch_size=5):
    """
    Loading and splitting the dataset
    :return: image_train, label_train, image_test, label_test, train_samples, test_samples
    """
    train_path = os.path.join("FLdata", "train", "mnist_train_" + str(client_id) + ".json")
    test_path = os.path.join("FLdata", "test", "mnist_test_" + str(client_id) + ".json")
    train_data = {}
    test_data = {}

    with open(os.path.join(train_path), "r") as f_train:
        train_temp = json.load(f_train)
        train_data.update(train_temp['user_data'])
    with open(os.path.join(test_path), "r") as f_test:
        test_temp = json.load(f_test)
        test_data.update(test_temp['user_data'])

    image_train = train_data['0']['x']
    label_train = train_data['0']['y']
    image_test = test_data['0']['x']
    label_test = test_data['0']['y']

    image_train = torch.Tensor(image_train).view(-1, 1, 28, 28).type(torch.float32)
    label_train = torch.Tensor(label_train).type(torch.int64)
    image_test = torch.Tensor(image_test).view(-1, 1, 28, 28).type(torch.float32)
    label_test = torch.Tensor(label_test).type(torch.int64)
    train_samples, test_samples = len(label_train), len(label_test)

    train_data = [(image, label) for image, label in zip(image_train, label_train)]
    test_data = [(image, label) for image, label in zip(image_test, label_test)]

    if opt == 0:    # 0 means GD
        train_loader = DataLoader(train_data, train_samples)
    else:           # 1 means Mini-batch SGD
        train_loader = DataLoader(train_data, batch_size)
    test_loader = DataLoader(test_data, test_samples)

    return train_loader, test_loader


class Client:
    def __init__(self, client_id, port, opt, iterations, epoch, learning_rate=0.01):
        self.client_id = client_id
        self.port = port
        self.opt = opt      # 0: GD, 1: Mini-batch SGD
        self.iterations = iterations
        self.epoch = epoch
        self.train_loader, self.test_loader = init_client(client_id, opt)
        self.model = MLR()
        self.loss = nn.NLLLoss()
        self.optimizer = None
        self.lr = learning_rate

        # Socket for the client (UDP)
        self.listen_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.listen_sock.bind(('localhost', port))
        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def hand_shaking_to_server(self):
        """
        Init the sock for sending message & send hand-shaking message to server
            1. Initialise socket and set up connection to the Server
                * sending hand-shaking message to Server
            2. Message should include: data size & client_id
                Format: "id size" id is just the number, size is the size of training data
        :return: None
        """
        # noinspection PyTypeChecker
        msg = self.client_id[-1] + " " + str(len(self.train_loader.dataset))
        # print(msg)
        self.send_sock.sendto(pickle.dumps(msg), ("localhost", 6000))

    def generate_log(self):
        """
        TODO:
            1. Writing the training loss and accuracy of the global model to the FILE
                * Generate it at each communication round
                * File name format ( client(id)_log.txt )
        :return: None
        """

    def display_info(self):
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

    def set_global_model(self):
        """
        (Used to called set_parameters())
        Receive a global model from the server, replace the local model with the global model
        """
        global_weight = []
        state_dict = self.model.state_dict()
        while 1:
            msg = self.listen_sock.recv(1024)
            msg = pickle.loads(msg)
            msg_header = msg.pop(0)
            if msg_header == "weight":
                global_weight.append(msg)
            elif msg_header == "weight end":
                global_weight = torch.Tensor(global_weight).reshape(10, 784)
                state_dict['fc1.weight'] = global_weight
            elif msg_header == "bias":
                global_bias = torch.Tensor(msg)
                state_dict['fc1.bias'] = global_bias
                self.model.load_state_dict(state_dict)
                break

    def send_local_model(self):
        """
        TODO:
            1. Sending local model to the Server
        :return: None
        """

    def test(self):
        """
        TODO: evaluate the accuracy
        """

    def train(self):
        """
        TODO:
            1. Update the new local model
                * Can be finished in E = 2 local iterations
                * Based on Model OPT to generate it (0 GD, 1 Mini-Batch GD)
                * Batch size could be changed
        :return:
        """

    def run(self):
        """
        The body of Client
        TODO:
            1. Init socket connection
            2. Loading dataset
            3. Setting a loop to listen message from Server
        :return :None
        """
        # Init Socket
        self.hand_shaking_to_server()

        # Init model (Just for test, this should be in the main loop)
        self.set_global_model()

        # TODO: Body of the client
        #       1. Outer loop to keep receiving & sending model
        #       2. Train the model with 2 epochs
        for i in range(1, self.iterations):
            # TODO: Listen the server

            # TODO: Train the local model with GD or mini-batch SGD
            for e in range(self.epoch):
                pass

            # TODO: Send model to the sever


if __name__ == "__main__":
    client = Client(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), 100, 2, 0.01)
    client.run()
