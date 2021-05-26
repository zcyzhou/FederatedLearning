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
import socket
import sys
import pickle
import copy

import torch

from utils import MLR


class Server:
    def __init__(self, port, sub_sample):
        self.port = port
        self.sub_sample = sub_sample
        self.k = 5
        self.iterations = 100
        self.clients = dict()
        self.total_data_size = 0
        # Randomly generate the global multinomial logistic regression model w_0
        self.model = MLR()
        # Socket for listening client message
        self.listen_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.listen_sock.bind(('localhost', port))
        # Socket for sending global model
        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def init_model(self):
        """
        NOTE: This has been done in __init__(), this code section is only for test
        TODO:
            1. Randomly generate the global model w_0
                * The model we are recommended to use is <multinomial logistic regression>
            2. Find an efficient way to represent the model in this program
        :return: The model
        """

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
                client_id, client_data_size, client_port = pickle.loads(self.listen_sock.recv(1024)).split()
                self.clients[client_id] = [int(client_data_size), int(client_port)]
                self.listen_sock.settimeout(30)

                # send back the current sever model to the client
                # TODO: Message too long, need to use multiple thread split the model to send
                # self.send_sock.sendto(pickle.dumps(self.model), ("localhost", int(client_port)))
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
        TODO:
            1. Format the message
            2. Send message to clients
        :return: None
        """

        # for client in self.clients:
        #     client_port = self.clients[client][1]
        # TODO: Message too long, need to use multiple thread split the model to send
        # self.send_sock.sendto(pickle.dumps(self.model), ("localhost", int(client_port)))

    def listen_clients_message(self):
        """
        Listen the update message from ALL clients
        NOTE:
            Even if we may only use a few of these clients' message, we still need to get message from all of them
        TODO:
            1. Receive message from ALL clients
            2. Unpack the update information
        :return: Message from clients
        """

    def aggregate_models(self):
        """
        Update the global model managed by server by aggregating updates from all/some of the clients
        TODO:
            1. Pick the subset of the clients
            2. Update the model
        :return: New Model
        """
        server_model = copy.deepcopy(self.model)

        # clear model before aggregation
        # for param in server_model.parameters():
        #     param.data = torch.zeros_like(param.data)

        # TODO: loop through the user dictionary model to generate new server model
        # self.model = copy.deepcopy(server_model)

    def run(self):
        """
        Body of the server
        TODO:
            1. Init the global model
            2. Detect clients
        :return: None
        """
        # Init the model
        self.init_model()

        # Detect clients
        self.detect_clients()

        # Send initial global model
        self.broadcast_to_clients()

        # TODO: The federated learning loop
        #       1. Listen, Aggregate, Broadcast
        #       2. Handle new clients (Probably handle this by another thread)
        for i in range(1, self.iterations):
            # TODO: Listen from clients
            self.listen_clients_message()

            # TODO: Aggregating model
            self.aggregate_models()

            # TODO: Broadcast new model
            self.broadcast_to_clients()


if __name__ == "__main__":
    server = Server(int(sys.argv[1]), int(sys.argv[2]))
    server.run()
