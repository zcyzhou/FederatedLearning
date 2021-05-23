"""
Client of the federated learning model.


--------------------
NOTE:
    1. Overall, we have 3 message formats to design:
        1. Sending Hand-shake message TO server
        2. Global model message FROM server TO client
        3. Update model message FROM client TO server
        4. Design multiple thread to spilt MODEL into several pieces message
        and send TO server
    2. We need to set up NEW connection to Server:
        1. init TCP socket and bind it with Server socket with PORT number 6000
"""

# import modules
import socket
import sys

# Global variables
ID = sys.argv[1]                    # Client ID
PORT = int(sys.argv[2])             # Listen port. From 6001 to 6005
MODEL_OPTION = int(sys.argv[3])     # Model Option. 0 is for GD, 1 is for Mini-Batch GD

def init_model():
    """
        TODO:
            1. Loading the dataset with given client id from command line argument
            2. Find an efficient way to represent the model in this program
        :return: The model
        """

def init_socket():
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
            1. Uses the global model to create a new local model
                * Can be finished in E = 2 local iterations
                * Based on Model OPT to generate it (0 GD, 1 Mini-Batch GD)
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
    init_socket()

    # Loading dataset
    init_model()


if __name__ == "__main__":
    main()
