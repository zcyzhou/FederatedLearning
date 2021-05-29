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
    log_path = os.path.join("log", client_id + "_log.txt")
    train_data = {}
    test_data = {}

    f_log = open(log_path, "w+")

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

    return train_loader, test_loader, f_log


class Client:
    def __init__(self, client_id, port, opt, iterations, epoch, learning_rate=0.01):
        self.client_id = client_id
        self.port = port
        self.opt = opt      # 0: GD, 1: Mini-batch SGD
        self.iterations = iterations
        self.epoch = epoch
        self.train_loader, self.test_loader, self.log = init_client(client_id, opt)
        self.model = MLR()
        self.loss = nn.NLLLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

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

    def generate_log(self, training_loss, testing_accuracy, iteration):
        """
        1. Writing the training loss and accuracy of the global model to the FILE
            * Generate it at each communication round
            * File name format (client(id)_log.txt)
        """
        file_content = self.log.readlines()
        file_content.append("Round {}\n".format(iteration + 1))
        file_content.append("Training loss: {}\n".format(training_loss))
        file_content.append("Testing accuracy: {}%\n".format(testing_accuracy))
        file_content.append("\n")
        self.log.write("".join(file_content))

    def set_global_model(self):
        """
        (Used to called set_parameters())
        Receive a global model from the server, replace the local model with the global model
        if receive a "close" header, then shutdown the program
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
            elif msg_header == "close":
                self.log.close()
                sys.exit()

    def send_local_model(self, training_loss, testing_accuracy):
        """
        Sending local model to the Server
        :return: None
        """
        weights = self.model.state_dict()['fc1.weight'].reshape(10, 8, -1)
        bias = self.model.state_dict()['fc1.bias']
        # Send weight
        for weight in weights:
            for w in weight:
                msg = w.tolist()
                msg.insert(0, 'weight ' + self.client_id[-1])
                self.send_sock.sendto(pickle.dumps(msg), ('localhost', 6000))
        self.send_sock.sendto(pickle.dumps(["train {}".format(training_loss)]), ('localhost', 6000))
        self.send_sock.sendto(pickle.dumps(["test {}".format(testing_accuracy)]), ('localhost', 6000))
        self.send_sock.sendto(pickle.dumps(["end " + self.client_id[-1]]), ('localhost', 6000))
        # Send bias
        msg = bias.tolist()
        msg.insert(0, 'bias ' + self.client_id[-1])
        self.send_sock.sendto(pickle.dumps(msg), ('localhost', 6000))

    def test(self):
        """
        TODO: evaluate the accuracy
        """
        test_acc = 0
        self.model.eval()
        for image, label in self.test_loader:
            output = self.model(image)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == label) * 1. / label.shape[0]).item()
        return test_acc

    def train_loss(self):
        """
        Training loss on the training data before training
        """
        training_loss = 0
        self.model.eval()
        for image, label in self.train_loader:
            output = self.model(image)
            training_loss += self.loss(output, label).data
        training_loss = training_loss / len(self.train_loader.dataset)
        return training_loss

    def train(self):
        """
            1. Update the new local model
                * Can be finished in E = 2 local iterations
                * Based on Model OPT to generate it (0 GD, 1 Mini-Batch GD)
                * Batch size could be changed
        :return:
        """
        loss = 0
        self.model.train()
        for epoch in range(1, self.epoch + 1):
            for batch_idx, (image, label) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                output = self.model(image)
                loss = self.loss(output, label)
                loss.backward()
                self.optimizer.step()
        return loss.data

    def run(self):
        """
        The body of Client
            1. Init socket connection
            2. Loading dataset
            3. Setting a loop to listen message from Server
        :return :None
        """
        # Init Socket
        self.hand_shaking_to_server()

        # Main loop of the client
        for i in range(self.iterations):
            # Listen the server
            self.set_global_model()
            print("I am client {}".format(self.client_id[-1]))
            print("Receiving new global model")

            # Evaluate training loss
            training_loss = self.train_loss()
            print("Training loss: {}".format(training_loss))

            # Test accuracy of global model
            testing_accuracy = self.test()
            print("Testing accuracy: {}%".format(testing_accuracy*100))

            # Local training
            self.train()
            print("Local training...")

            # write current status to log
            self.generate_log(training_loss, testing_accuracy, i)

            # Send local model to server
            print("Sending new local model")
            self.send_local_model(training_loss, testing_accuracy)

            # Print an empty line to separate the output
            print("")


if __name__ == "__main__":
    client = Client(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), 100, 2, 0.01)
    client.run()
