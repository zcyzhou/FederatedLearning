"""
Server of the federated learning model.


--------------------
NOTE:
    1. Overall, we have 3 message formats to design:
        1. Hand-shake message FROM client TO server
        2. Global model message FROM server TO client
        3. Update model message FROM client TO server
    2. We need to handle NEW clients:
        1. Everytime we receive a message, we need to classify is it a hand-shake or model-related message
        2. Do not preset a global dictionary which stores 5 clients
"""

# Import modules
import random
import socket
import sys
import pickle
import torch

from matplotlib import pyplot as plt
from utils import MLR


class Server:
    def __init__(self, port, sub_sample):
        self.port = port
        if sub_sample == 0:
            self.sub_sample = 5
        elif sub_sample == 1:
            self.sub_sample = 2
        self.k = 5
        self.iterations = 100
        self.clients = dict()
        self.total_data_size = 0
        # List for storing the average of the training loss and testing accuracy each iteration
        self.training_loss = []
        self.testing_accuracy = []
        # Randomly generate the global multinomial logistic regression model w_0
        self.model = MLR()
        # Socket for listening client message
        self.listen_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.listen_sock.bind(('localhost', port))
        # Socket for sending global model
        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def detect_clients(self):
        """
        Register the clients. Info will be stored in self.clients in format:
        {<client_id>: <data_size>}
        Note that <client_id> only contains the number

        Listen the port to detect the hand-shake message from clients
                * Everytime get a massage from one new client, add it to a list then sleep 30 seconds
                * The message includes: <data-size> <client-id>
        :return: Number of clients registered
        """
        while 1:
            try:
                client_id, client_data_size = pickle.loads(self.listen_sock.recv(1024)).split()
                self.clients[client_id] = int(client_data_size)
                self.listen_sock.settimeout(30)
            except socket.timeout:
                break
        self.listen_sock.settimeout(None)
        # Calculate the total training data size
        for client, data_size in self.clients.items():
            self.total_data_size += int(data_size)

        return len(self.clients)

    def broadcast_to_clients(self):
        """
        Broadcast the new global model to all registered clients
        (no matter whether we use the local model of that client)
            1. Format the message
                [<header>, <weight/bias1>, <weight/bias2>, ...]
                <head>: weight/end/bias
            2. Send message to clients
        :return: None
        """
        weights = self.model.state_dict()['fc1.weight'].reshape(10, 8, -1)
        bias = self.model.state_dict()['fc1.bias']
        # Send to ALL the clients
        for client_id, _ in self.clients.items():
            # Send weight first
            for weight in weights:
                for w in weight:
                    msg = w.tolist()
                    msg.insert(0, 'weight')
                    self.send_sock.sendto(pickle.dumps(msg), ('localhost', 6000 + int(client_id)))
            self.send_sock.sendto(pickle.dumps(["weight end"]), ('localhost', 6000 + int(client_id)))
            # Send bias
            msg = bias.tolist()
            msg.insert(0, 'bias')
            self.send_sock.sendto(pickle.dumps(msg), ('localhost', 6000 + int(client_id)))

    def listen_clients_message(self):
        """
        DO NOT need to call this explicitly, this func would be invoked in aggregate_models() function
        Listen the update message from ALL clients
        NOTE:
            Even if we may only use a few of these clients' message, we still need to get message from all of them
            1. Receive message from ALL clients
            2. Unpack the update information
        :return: Dictionary of all clients' models
                    {
                        client_id: {'weight': <Tensor>, 'bias': <Tensor>},
                        client_id: {...},
                        ...
                    }
        """
        client_models = dict()
        new_connection = dict()
        received_model = 0
        total_training = 0
        total_testing = 0

        for client in self.clients.keys():
            client_models[client] = dict()
            client_models[client]['weight'] = []
        while 1:
            msg = self.listen_sock.recv(1024)
            msg = pickle.loads(msg)

            # determine the type of the msg recv from client
            # if it is model message, then keep receiving
            # if it is hand shaking msg, then store in the temporary connection dict
            if type(msg) == list:
                header = msg.pop(0).split()
                if header[0] == 'weight':
                    client_models[header[1]]['weight'].append(msg)
                elif header[0] == 'end':
                    client_models[header[1]]['weight'] = torch.Tensor(client_models[header[1]]['weight']).reshape(10, 784)
                elif header[0] == 'bias':
                    client_models[header[1]]['bias'] = msg
                    client_models[header[1]]['bias'] = torch.Tensor(client_models[header[1]]['bias'])
                    received_model += 1
                    print("Getting local model from client {}".format(header[1]))
                elif header[0] == 'train':
                    total_training += float(header[1])
                elif header[0] == 'test':
                    total_testing += float(header[1])
                if received_model == len(self.clients):
                    break
            else:
                client_id, client_data_size = msg.split()
                new_connection[client_id] = int(client_data_size)

        # evaluate the global model across all connecting clients
        self.training_loss.append(total_training)
        self.testing_accuracy.append(total_testing / received_model)

        # update new client id and its data size to the self.clients
        if len(new_connection) > 0:
            for client, data_size in new_connection.items():
                self.clients[client] = data_size

        return client_models

    def aggregate_models(self):
        """
        Update the global model managed by server by aggregating updates from all/some of the clients
        This method would invoke listen_clients_message
            1. Pick the subset of the clients
            2. Update the model
        :return: New Model
        """
        new_weight = torch.zeros(10, 784)
        new_bias = torch.zeros(10)
        sub_clients = random.sample(list(self.clients.keys()), self.sub_sample)
        total_samples = sum([self.clients[client] for client in sub_clients])
        client_models = self.listen_clients_message()

        for client in sub_clients:
            new_weight += client_models[client]['weight'] * self.clients[client] / total_samples
            new_bias += client_models[client]['bias'] * self.clients[client] / total_samples

        state_dict = self.model.state_dict()
        state_dict['fc1.weight'] = new_weight
        state_dict['fc1.bias'] = new_bias
        self.model.load_state_dict(state_dict)

    def send_shutdown_msg(self):
        """
        Broadcast all the connecting clients to shut down their program
            1. Format the message
                [<header>, ...]
                <header>: close
        :return: None
        """
        msg = ["close"]
        for client_id, _ in self.clients.items():
            self.send_sock.sendto(pickle.dumps(msg), ('localhost', 6000 + int(client_id)))

    def generate_training_figure(self):
        """
        Generate Training Loss FedAvg figure to the local file.
        :return: None
        """
        plt.figure(1, figsize = (7, 7))
        plt.plot(self.training_loss, label = "FedAvg", linewidth = 1)
        plt.legend(loc = 'upper right', prop = {'size': 12}, ncol = 2)
        plt.ylabel("Training Loss")
        plt.xlabel("Global rounds")
        plt.savefig("Training_Loss.png")

    def generate_testing_figure(self):
        """
        Generate Testing Accuracy FedAvg figure to the local file.
        :return: None
        """
        plt.clf()
        plt.figure(1, figsize = (5, 5))
        plt.plot(self.testing_accuracy, label = "FedAvg", linewidth = 1)
        plt.ylim([0.1, 0.99])
        plt.legend(loc = 'upper right', prop = {'size': 12}, ncol = 2)
        plt.ylabel("Testing Accuracy")
        plt.xlabel("Global rounds")
        plt.savefig("Testing_Accuracy.png")

    def run(self):
        """
        Body of the server
            1. Init the global model
            2. Detect clients
        :return: None
        """
        # Detect clients
        self.detect_clients()

        # Send initial global model
        self.broadcast_to_clients()

        # 1. Listen, Aggregate, Broadcast
        # 2. Handle new clients (Probably handle this by another thread)
        for i in range(0, self.iterations):
            print("Global Iteration {}".format(i + 1))
            print("Total Number of clients: {}".format(len(self.clients)))

            # Listen from clients and Aggregating model
            self.aggregate_models()
            print("Aggregating new global model")

            # Broadcast new model
            print("Boardcasting new global model")
            self.broadcast_to_clients()

            # Print an empty line to separate the output
            print("")

        # After iterations, send the msg to all connecting clients to close their program
        self.send_shutdown_msg()

        # Generate two average figure
        self.generate_training_figure()
        self.generate_testing_figure()
        print("Iterations have finished, program is closing...")
        print("")


if __name__ == "__main__":
    server = Server(int(sys.argv[1]), int(sys.argv[2]))
    server.run()
