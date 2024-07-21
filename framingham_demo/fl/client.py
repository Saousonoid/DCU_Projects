import flwr as fl
from flwr.common import (
    Code, EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Status
)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
from  utils import *
from flwr.client.mod.localdp_mod import LocalDpMod
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 
from torch import tensor

def load_data(file_path, label_column):
    data = pd.read_csv(file_path,index_col=0)

    X = data.drop(label_column, axis=1).values
    y = data[label_column].values
    return X, y

epochs = 3
clipping_norm = 1.0  
sensitivity = 1.0
epsilon = 1.0
delta = 1e-5 
class Client(fl.client.Client):
    def __init__(self, net, train_loader, val_loader, local_epochs, optimizer, criterion, num_train, num_val,local_dp_mod):
        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.local_epochs = local_epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_train = num_train
        self.num_val = num_val
        self.local_dp_mod = local_dp_mod

    def fit(self, ins: FitIns) -> FitRes:
        global_round = int(ins.config["global_round"])
        print(f"Current round: {global_round}")

        if global_round > 1:
            shared_weight = ins.parameters.tensors[0]
            self.net.shared_hidden.weight.data.copy_(
                tensor(np.frombuffer(shared_weight, dtype=np.float32)).view_as(self.net.shared_hidden.weight.data)
            )
        
        self.net.train_model(self.train_loader, self.criterion, self.optimizer, self.local_epochs)

        self.net.train_model(self.train_loader, self.criterion, self.optimizer, self.local_epochs)
        local_weights = [self.net.shared_hidden.weight.data.numpy().tobytes(),self.net.output.weight.data.numpy().tobytes()]

    
       
        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=Parameters(tensor_type="NUMPY", tensors=local_weights),
            num_examples=self.num_train,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        print("Starting evaluation...")
        avg_loss, accuracy = test_model(self.net, self.val_loader, self.criterion)
        print("Loss: "+ str(avg_loss))
        print("Accuracy: "+ str(accuracy))

        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            loss=float(avg_loss),
            num_examples=self.num_val,
            metrics={"accuracy": float(accuracy)},
        )

def main(client_data_path, label_column):
    X, y = load_data(client_data_path, label_column)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    num_train = len(X_train)
    num_val = len(X_val)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    input_dim = X_train.shape[1]

    hidden_dim = 3
    output_dim = 1 


    model = SingleTaskNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    
    local_dp_obj = LocalDpMod(
    clipping_norm=clipping_norm,
    sensitivity=sensitivity,
    epsilon=epsilon,
    delta=delta
    )

    client = Client(
        train_loader=train_loader,
        val_loader=val_loader,
        net=model,
        local_epochs=5,
        optimizer=optimizer,
        criterion=criterion,
        num_train=num_train,
        num_val=num_val,
        local_dp_mod=local_dp_obj
    )

    fl.client.start_client(server_address="localhost:5040", client=client)

if __name__ == "__main__":
    client_data_path = sys.argv[1]  
    label_column = sys.argv[2]  
    main(client_data_path, label_column)
