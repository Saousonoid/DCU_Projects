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
    def __init__(self, net, train_loader, val_loader, local_epochs, optimizer, criterion, num_train, num_val,local_dp_mod,task_no):
        self.task_no=task_no
        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.local_epochs = local_epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_train = num_train
        self.num_val = num_val
        self.local_dp_mod = local_dp_mod
        self.aes_key = generate_aes_key()
        server_pk = load_key('server_puk.pem')
        self.encrypted_aes_key = rsa_encrypt(server_pk, self.aes_key)
        


    def fit(self, ins: FitIns) -> FitRes:
        global_round = int(ins.config["global_round"])

        if global_round > 1:
            shared_weight = ins.parameters.tensors[0]
            shared_weight=decrypt_data(self.aes_key,shared_weight)
            self.net.shared_hidden.weight.data.copy_(
                tensor(np.frombuffer(shared_weight,  dtype=np.float32)).view_as(self.net.shared_hidden.weight.data)
            )
        
        self.net.train_model(self.train_loader, self.criterion, self.optimizer, self.local_epochs)
        local_weights = [self.net.shared_hidden.weight.data.numpy().tobytes(),self.net.output.weight.data.numpy().tobytes()]
        encrypted_weights = [encrypt_data(self.aes_key, weight) for weight in local_weights]

        if global_round == 1:
            s_key=self.encrypted_aes_key
            encrypted_weights.append(s_key)
        

        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=Parameters(tensor_type="NUMPY", tensors=encrypted_weights),
            num_examples=self.num_train,
            metrics={'task_no':self.task_no},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        print("Starting evaluation...")
        avg_loss, accuracy = utils.test_model(self.net, self.val_loader, self.criterion)
        print("Loss: "+ str(avg_loss))
        print("Accuracy: "+ str(accuracy))

        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            loss=float(avg_loss),
            num_examples=self.num_val,
            metrics={"accuracy": float(accuracy)},
        )

def main(client_data_path, label_column,task_no):
    X, y = load_data(client_data_path, label_column)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    num_train = len(X_train)
    num_val = len(X_val)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    train_dataset = utils.CustomDataset(X_train, y_train)
    val_dataset = utils.CustomDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    input_dim = X_train.shape[1]

    hidden_dim = 3
    output_dim = 1 

    output_dims = [1, 1]  

    model = utils.SingleTaskNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    
    local_dp_obj = LocalDpMod(
    clipping_norm=clipping_norm,
    sensitivity=sensitivity,
    epsilon=epsilon,
    delta=delta
    )

    client = Client(
        task_no=task_no,
        train_loader=train_loader,
        val_loader=val_loader,
        net=model,
        local_epochs=5,
        optimizer=optimizer,
        criterion=criterion,
        num_train=num_train,
        num_val=num_val,
        local_dp_mod=local_dp_obj,
        
    )
    try:


        fl.client.start_client(server_address="localhost:5040", client=client)
    except  Exception as e:
        print(f"Unexpected Error: {e}")
    finally:
        sys.exit(0)



if __name__ == "__main__":
    client_data_path = sys.argv[1]  
    label_column = sys.argv[2]
    task_no=  sys.argv[3]
    main(client_data_path, label_column,task_no)
