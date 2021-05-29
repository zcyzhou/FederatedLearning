# FederatedLearning
Implementing a simple Federated Learning (FL) system including five clients in
total and one server for aggregation in Figs. Each client has its own data used for training
its local model and then contributes its local model to the server through the socket in order
to build the global model.

## Language
- Python3

## Usage
#### Starting the Program
Server:

      - python3 COMP3221_FLServer.py <Port-Server> <Sub-client>

#### For example

      - python3 COMP3221_FLServer.py 6000 0

Client:

      - python3 COMP3221_FLClient.py <Client-id> <Port-Client> <Opt-Method>

#### For example

      - python3 COMP3221_FLClient.py client1 6001 0

## Development Rule

Please develop in Pycharm IDE. Everytime before pushing code please make sure:
1. Pull and solve conflict everytime **before push**
2. Make sure there is no **error** or **warning** prompt in IDE auto check.
   (There should only be a green check mark in the Pycharm)
3. Please comment any modification clearly. Do not use `git add --all`. Add
the files and comment them separately.