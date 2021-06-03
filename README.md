# FederatedLearning
Implementing a simple Federated Learning (FL) system including five clients in
total for model training and one server for model aggregation . Each client has its own data used for training
its local model and then contributes its local model to the server through the socket in order
to build the global model.

## Requirement
* Please put `FLdata` directory in the same level of `COMP3221_FLClient.py` file
* `Python` version >= 3.7
* `Pytorch` version = 1.8.1

## Usage
### Server
To start a **Server**, run the following command in a new terminal
```
python3 COMP3221_FLServer.py <Port-Server> <Sub-client>
```
* `<Port-Server>` should be the port 6000
* `<Sub-client>` is the command to determine how many clients' local model to aggregate to form the global model. There are two possible values:
     * `0`: Use all 5 clients
     * `1`: Use 2 out of 5 clients (Pick randomly)

For example:
```
python3 COMP3221_FLServer.py 6000 0
```

### Client
To start a **Client**, run the following command in a new terminal
```
python3 COMP3221_FLClient.py <Client-id> <Port-Client> <Opt-Method>
```
* <Client-id> should be `client1`, `client2`, ..., `client5` respectively for 5 clients 
* <Port-Server> should be `6001` for `client1`, `6002` for `client2` and so on.
* <Opt-Method> is used to determine which kind of gradient descent method to use:
      * `0` is for Gradient Descent
      * `1` is for Mini-Batch Gradient Descent (default batch size is 5)

For example:
```
python3 COMP3221_FLClient.py client1 6001 1
```
## Notes
1. Always start the **Server** before starting **Client**
2. After start a client, the next client should be started within 30s otherwise the server will not receive handshake message from it.
