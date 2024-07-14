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
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 
from torch import tensor
import warnings

# Load dataset with task feature
def load_data(file_path, label_column):
    data = pd.read_csv(file_path,index_col=0)

    X = data.drop(label_column, axis=1).values
    y = data[label_column].values
    return X, y

epochs = 3

class Client(fl.client.Client):
    def __init__(self, net, train_loader, val_loader, local_epochs, optimizer, criterion, num_train, num_val, num_rounds):
        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.local_epochs = local_epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_train = num_train
        self.num_val = num_val
        self.num_rounds = num_rounds  # Total number of rounds

    def fit(self, ins: FitIns) -> FitRes:
        global_round = int(ins.config["global_round"])
        print(f"Global round: {global_round}/{self.num_rounds}")

        # Load shared layer weights from server if not in the first round
        if global_round > 1:
            shared_weight = ins.parameters.tensors[0]
            self.net.shared_hidden.weight.data.copy_(
                tensor(np.frombuffer(shared_weight, dtype=np.float32)).view_as(self.net.shared_hidden.weight.data)
            )
            print('new round out!')
        
        self.net.train_model(self.train_loader, self.criterion, self.optimizer, self.local_epochs)
        local_weights = [self.net.shared_hidden.weight.data.numpy().tobytes(),self.net.output.weight.data.numpy().tobytes()]
        print('# Train the model locally')

        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=Parameters(tensor_type="NUMPY", tensors=local_weights),
            num_examples=self.num_train,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        print("Starting evaluation...")
        avg_loss, accuracy = utils.test_model(self.net, self.val_loader, self.criterion)
        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            loss=float(avg_loss),
            num_examples=self.num_val,
            metrics={"accuracy": float(accuracy)},
        )

def main(client_data_path, label_column, num_rounds):
    X, y = load_data(client_data_path, label_column)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    num_train = len(X_train)
    num_val = len(X_val)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Create datasets and dataloaders
    train_dataset = utils.CustomDataset(X_train, y_train)
    val_dataset = utils.CustomDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    # Initialize model, optimizer, and criterion
    input_dim = X_train.shape[1]

    hidden_dim = 20
    output_dim = 1 
    # Assuming multiple output layers
    output_dims = [1, 1]  

    model = utils.SingleTaskNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    client = Client(
        train_loader=train_loader,
        val_loader=val_loader,
        net=model,
        local_epochs=5,
        optimizer=optimizer,
        criterion=criterion,
        num_train=num_train,
        num_val=num_val,
        num_rounds=num_rounds  # Pass the number of rounds to the client
    )

    fl.client.start_client(server_address="localhost:5040", client=client)

if __name__ == "__main__":
    client_data_path = sys.argv[1]  
    label_column = sys.argv[2] 
    num_rounds = 5 
    main(client_data_path, label_column, num_rounds)
