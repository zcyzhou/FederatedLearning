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
import sys

# Global variables
T = 100                                                 # Number of global iterations
PORT = int(sys.argv[1])                                 # Listen port. Fixed to 6000
SUB = 5 if int(sys.argv[2]) == 0 else int(sys.argv[2])  # Sub-sampling number of clients
K = 5                                                   # Number of clients


def init_model():
    """
    TODO:
        1. Randomly generate the global model w_0
            * The model we are recommended to use is <multinomial logistic regression>
        2. Find an efficient way to represent the model in this program
    :return: The model
    """


def detect_clients():
    """
    TODO: Listen the port to detect the hand-shake message from clients
            * Everytime get a massage from one new client, add it to a list then sleep 30 seconds
            * The message includes: <data-size> <client-id>
    :return: The list of clients
    """


def broadcast_to_clients():
    """
    Broadcast the new global model to all registered clients (no matter whether we use the local model of that client)
    TODO:
        1. Format the message
        2. Send message to clients
    :return: None
    """


def listen_clients_message():
    """
    Listen the update message from ALL clients
    NOTE:
        Even if we may only use a few of these clients' message, we still need to get message from all of them
    TODO:
        1. Receive message from ALL clients
        2. Unpack the update information
    :return: Message from clients
    """


def aggregate_models():
    """
    Update the global model managed by server by aggregating updates from all/some of the clients
    TODO:
        1. Pick the subset of the clients
        2. Update the model
    :return: New Model
    """


def main():
    """
    Body of the server
    TODO:
        1. Init the global model
        2. Detect clients
    :return: None
    """
    # Init the model
    init_model()

    # Detect clients
    detect_clients()

    # Send initial global model
    broadcast_to_clients()

    # TODO: The federated learning loop
    #       1. Listen, Aggregate, Broadcast
    #       2. Handle new clients (Probably handle this by another thread)
    for i in range(1, T):
        # TODO: Listen from clients
        listen_clients_message()

        # TODO: Aggregating model
        aggregate_models()

        # TODO: Broadcast new model
        broadcast_to_clients()


if __name__ == "__main__":
    main()
