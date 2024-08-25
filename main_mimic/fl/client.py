import flwr as fl
from flwr.common import (
    Code, EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Status
)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import utils
from flwr.client.mod.localdp_mod import LocalDpMod
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 
from torch import tensor
import warnings
import pickle
from utils import generate_aes_key, rsa_encrypt, encrypt_data, decrypt_data, load_key

def load_data(file_path, label_column):
    """Load and preprocess the data from a CSV file"""
    data = pd.read_csv(file_path, index_col=0)
    X = data.drop(label_column, axis=1).values
    y = data[label_column].values
    return X, y

# Parameters for Differential Privacy
clipping_norm = 1.0  
sensitivity = 1.0
epsilon = 1.0
delta = 1e-5 

class Client(fl.client.Client):
    """Custom Flower client implementing differential privacy and encryption """

    def __init__(self, net, train_loader, val_loader, local_epochs, optimizer, criterion, num_train, num_val, local_dp_mod, task_no):
        # Assign task number and initialize the neural network, data loaders, and training configurations.
        self.task_no = task_no
        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.local_epochs = local_epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_train = num_train
        self.num_val = num_val
        self.local_dp_mod = local_dp_mod

        # Key management for encryption and decryption
        self.aes_key = generate_aes_key()  # Generate an AES key for symmetric encryption
        server_pk = load_key('server_puk.pem')  # Load the server's public RSA key
        self.encrypted_aes_key = rsa_encrypt(server_pk, self.aes_key)  # Encrypt the AES key with the server's public key

    def fit(self, ins: FitIns) -> FitRes:
        """Train the model locally and return the updated model parameters"""
        global_round = int(ins.config["global_round"])  # Extract the current round number

        # Decrypt and load the global model weights if the round number is greater than 1
        if global_round > 1:
            shared_weight1 = ins.parameters.tensors[0]  # Shared Layer #1
            shared_weight2 = ins.parameters.tensors[1]  # Shared Layer #2

            # Decrypt the weights using the AES key
            shared_weight1 = decrypt_data(self.aes_key, shared_weight1)
            shared_weight2 = decrypt_data(self.aes_key, shared_weight2)

            # Copy the decrypted weights into the local model
            self.net.shared_hidden1.weight.data.copy_(
                tensor(np.frombuffer(shared_weight1, dtype=np.float32)).view_as(self.net.shared_hidden1.weight.data)
            )
            self.net.shared_hidden2.weight.data.copy_(
                tensor(np.frombuffer(shared_weight2, dtype=np.float32)).view_as(self.net.shared_hidden2.weight.data)
            )
        
        # Train the local model using the provided data loader
        self.net.train_model(self.train_loader, self.criterion, self.optimizer, self.local_epochs)
        
        # Prepare the updated model weights
        local_weights = [
            # Convert each layer's weights to a byte array for encryption
            self.net.shared_hidden1.weight.data.numpy().tobytes(),
            self.net.shared_hidden2.weight.data.numpy().tobytes(),
            self.net.personalized_hidden.weight.data.numpy().tobytes(),  #  personalized layer
            self.net.output.weight.data.numpy().tobytes()  #  output layer
        ]

        # Encrypt each set of weights using the AES key
        encrypted_weights = [encrypt_data(self.aes_key, weight) for weight in local_weights]

        # For the first round, also append the encrypted AES key
        if global_round == 1:
            s_key = self.encrypted_aes_key
            encrypted_weights.append(s_key)

        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=Parameters(tensor_type="NUMPY", tensors=encrypted_weights),
            num_examples=self.num_train,
            metrics={'task_no': self.task_no},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Evaluate the model locally and return the loss and accuracy"""
        print("Starting evaluation..>>>>>>.....>>>>>>>.......>>>>")

        # Test the model on the validation data
        avg_loss, accuracy = utils.test_model(self.net, self.val_loader, self.criterion)
        print(f"Evaluation completed - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            loss=float(avg_loss),
            num_examples=self.num_val,
            metrics={"accuracy": float(accuracy)},
        )

def main(client_data_path, label_column, task_no):
    """Main function to set up the client and start the training process"""
    X, y = load_data(client_data_path, label_column)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    num_train = len(X_train)
    num_val = len(X_val)

    # Data preprocessing: scaling the features
    scaler = StandardScaler()
    X_train[:, :-2] = scaler.fit_transform(X_train[:, :-2])
    X_val[:, :-2] = scaler.transform(X_val[:, :-2])

    # Prepare data loaders for training and validation sets
    train_dataset = utils.CustomDataset(X_train, y_train)
    val_dataset = utils.CustomDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    input_dim = X_train.shape[1]
    hidden_dim = input_dim
    output_dim = 1 

    # Initialize the model, optimizer, and loss function
    model = utils.SingleTaskNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    optimizer = optim.Adam([
        # Assign different learning rates and weight decay to different parts of the model
        {'params': model.shared_hidden1.parameters(), 'lr': 0.0001, 'weight_decay': 2e-5},
        {'params': model.shared_hidden2.parameters(), 'lr': 0.0001, 'weight_decay': 2e-5},
        {'params': model.personalized_hidden.parameters(), 'lr': 0.001, 'weight_decay': 2e-3},
        {'params': model.output.parameters(), 'lr': 0.001, 'weight_decay': 2e-3}
    ])    
    criterion = nn.BCEWithLogitsLoss()
    
    # Initialize the differential privacy module with the specified parameters
    local_dp_obj = LocalDpMod(
        clipping_norm=clipping_norm,
        sensitivity=sensitivity,
        epsilon=epsilon,
        delta=delta
    )

    # Create the Flower client with all configurations
    client = Client(
        task_no=task_no,
        train_loader=train_loader,
        val_loader=val_loader,
        net=model,
        local_epochs=15,
        optimizer=optimizer,
        criterion=criterion,
        num_train=num_train,
        num_val=num_val,
        local_dp_mod=local_dp_obj,
    )
    
    try:
        # Start the client and connect to the server
        fl.client.start_client(server_address="localhost:5040", client=client)
    except Exception as e:
        print(f"Unexpected Error: {e}")
    finally:
        sys.exit(0)

if __name__ == "__main__":
    client_data_path = sys.argv[1]  
    label_column = sys.argv[2]
    task_no  = sys.argv[3]
    print(f'Starting Client for Task: {task_no}, Representing Label: {label_column}')
    main(client_data_path, label_column, task_no)
