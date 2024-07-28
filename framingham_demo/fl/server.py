import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Parameters, FitIns,FitRes,Scalar,EvaluateIns
from typing import Dict
import pandas as pd
import numpy as np
import torch
from utils import *
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import sys,os
from typing import Callable, Dict, List, Optional, Tuple, Union
import warnings
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

warnings.filterwarnings("ignore", category=DeprecationWarning)

rounds = 1

def load_test_data(file_path):
    data = pd.read_csv(file_path, index_col=0)
    X = data.drop(['prevalentHyp', 'TenYearCHD'], axis=1).values
    y = data.iloc[:, -2:-1].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

def aggregate_weights(weight_list) -> torch.Tensor:
    final_list = []
    for weight_bytes in weight_list:
        weight_array = np.frombuffer(weight_bytes, dtype=np.float32)
        weight_tensor = torch.tensor(weight_array)
        final_list.append(weight_tensor)

    total_weight = torch.zeros_like(final_list[0].clone().detach())
    for weight in final_list:
        total_weight += weight.clone().detach()
    return total_weight / len(final_list)

class CustomFedAvg(FedAvg):
    def __init__(self, num_rounds: int,rsa_prk, **kwargs):
        super().__init__(**kwargs)
        self.num_rounds = num_rounds
        self.current_round = 0
        self.accuracy_list = []
        self.loss_list = []
        self.avg_accuracy = 0
        self.optim_model= None
        self.optim_acc=0
        self.optim_loss=0
        self.avg_loss = float(1.00000)
        self.model = None
        self.client_keys={}
        self.rsa_prk=rsa_prk
        
        

    def initialize_parameters(self, client_manager):
        return None

    def aggregate_fit(self, rnd: int, results, failures) -> Tuple[ Parameters, Dict[str, Scalar]]:
        results.sort(key=lambda result: result[1].metrics["task_no"])
        if rnd==1:
            for result in results:
                client_id = result[0].cid  # Get client ID
                s_key = result[1].parameters.tensors[2]
                self.client_keys[client_id]=rsa_decrypt(self.rsa_prk,s_key)


        weight_list, output_list = zip(*[(
            decrypt_data(self.client_keys[result[0].cid],result[1].parameters.tensors[0]), 
            decrypt_data(self.client_keys[result[0].cid],result[1].parameters.tensors[1])) for result in results])
        aggregated_weights = aggregate_weights(weight_list)

        aggregated_weights=aggregated_weights.numpy().tobytes()
        multitask_layers=Parameters(
            tensors=[aggregated_weights]+list(output_list),
            tensor_type="NUMPY"
        )

        print(self.multitask_evaluate(rnd, multitask_layers ))
        global rounds
        rounds += 1
        metrics: Dict[str, Scalar] = {
            "round": rnd,
            "client_count": len(results)
        }

        
        return multitask_layers, metrics

    def aggregate_evaluate(self, rnd: int, results, failures) -> Dict[str, float]:
        accuracies, losses = zip(*[(client[1].metrics['accuracy'], client[1].loss) for client in results])

        avg_accuracy = sum(accuracies) / len(accuracies)
        avg_loss = float(sum(losses) / len(losses))

        self.avg_accuracy = max(self.avg_accuracy, avg_accuracy)
        self.avg_loss = min(self.avg_loss, avg_loss)

        mtl_acc_avg=sum(self.accuracy_list)/len(self.accuracy_list)
        mtl_loss_avg=sum(self.loss_list)/len(self.loss_list)
        max_accuracy, min_accuracy  = max(self.accuracy_list),min(self.accuracy_list)

        if mtl_acc_avg> self.optim_acc and mtl_loss_avg<= self.optim_loss and abs(max_accuracy-min_accuracy)<=0.08:
            self.optim_model=self.model
            self.optim_acc=mtl_acc_avg
            self.optim_loss =mtl_loss_avg
        for i in range(len(self.accuracy_list)):
            chk = check_thres(self.accuracy_list[i], self.loss_list[i], self.avg_accuracy, self.avg_loss, 0.85, 1.5)
            if not chk:
                if rnd == self.num_rounds:
                    self.model=self.optim_model
                    print('Training Completed, Optimal MTL Model Metrics:')
                    print(f'Optimal Average Accuracy: {self.optim_acc:.4f}')
                    print(f'Optimal Average Loss: {self.optim_loss:.4f}')
                return None, {"avg_accuracy: ": avg_accuracy, "avg_loss: ": avg_loss}

        print('Performance Conditions Met, Ending all Round Training')
        sys.exit(0)
        return None, {"avg_accuracy": avg_accuracy, "avg_loss": avg_loss}

    


    def configure_fit(
        self, server_round: int, parameters:Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        

        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        if server_round == 1:
            if self.on_fit_config_fn is not None:
                config = self.on_fit_config_fn(server_round)
            fit_ins = FitIns(parameters, config)
            return [(client, fit_ins) for client in clients]

        fit_instructions = []

        for client in clients:
            client_id = client.cid
            client_key = self.client_keys[client_id]
            enc_weights = encrypt_data(client_key, parameters.tensors[0])
            enc_outputs = [encrypt_data(client_key,output ) for output in  parameters.tensors[1:]]
            enc_parameters = Parameters(
            tensors=[enc_weights]+ enc_outputs,
            tensor_type="NUMPY"
        )

            config = {}
            if self.on_fit_config_fn is not None:
                config = self.on_fit_config_fn(server_round)
            fit_ins = FitIns(enc_parameters, config)
            fit_instructions.append((client, fit_ins))
        return fit_instructions



    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, EvaluateIns]]:
        if self.fraction_evaluate == 0.0:
            return []
        config = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)
        parameters = Parameters(
            tensors=[parameters.tensors[0]],
            tensor_type="NUMPY"
        )
        evaluate_ins = EvaluateIns(parameters, config)
        sample_size, min_num_clients = self.num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        return [(client, evaluate_ins) for client in clients]
    
    def evaluate(self, server_round: int, parameters: Parameters):
        return None

    def multitask_evaluate(self, server_round: int, parameters: Parameters):
        self.accuracy_list = []
        self.loss_list = []

        X_test, y_test = load_test_data(self.test_data_path)
        test_dataset = CustomDataset(X_test, y_test)

        input_dim = X_test.shape[1]
        hidden_dim = 3
        no_outputs=len(parameters.tensors[1:])
        output_dims = [1]*no_outputs
        shared_hidden_layer = torch.nn.Linear(input_dim, hidden_dim)
        output_layers = [torch.nn.Linear(hidden_dim, output_dim) for output_dim in output_dims]
        model = MultiTaskNN(shared_hidden=shared_hidden_layer, output_heads=output_layers)
        self.model = model
        shared_weights = parameters.tensors[0]

        model.shared_hidden.weight.data.copy_(torch.tensor(np.frombuffer(shared_weights, dtype=np.float32)).view_as(model.shared_hidden.weight.data))
        for i, output_layer in enumerate(model.output_heads):
            output_layer.weight.data.copy_(torch.tensor(
                np.frombuffer(parameters.tensors[1+i], dtype=np.float32)).view_as(output_layer.weight.data))

        test_df = pd.read_csv('test.csv')
        X_test = test_df.iloc[:, 1:-2].values
        y_test_task1 = test_df.iloc[:, -2].values
        y_test_task2 = test_df.iloc[:, -1].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_test)

        test_dataset_task1 = CustomDataset(X_scaled, y_test_task1)
        test_loader_task1 = DataLoader(test_dataset_task1, batch_size=32, shuffle=False)

        test_dataset_task2 = CustomDataset(X_scaled, y_test_task2)
        test_loader_task2 = DataLoader(test_dataset_task2, batch_size=32, shuffle=False)
        criterion = torch.nn.BCEWithLogitsLoss()

        loss1, accuracy1 = test_model(model, test_loader_task1, criterion=criterion, task_id=1)
        self.accuracy_list.append(accuracy1)
        self.loss_list.append(loss1)

        loss2, accuracy2 = test_model(model, test_loader_task2, criterion=criterion, task_id=2)
        self.accuracy_list.append(accuracy2)
        self.loss_list.append(loss2)


        return {"Loss Task 1: ": float(loss1), "Loss Task 2: ": float(loss2)}, {"accuracy Task 1:": float(accuracy1), "accuracy Task 2:": float(accuracy2)}

def config_func(rnd: int) -> Dict[str, str]:
    config = {
        "global_round": str(rnd),
    }
    return config

def main(test_data_path, num_rounds):
    if os.path.exists('server_puk.pem'):
        print('removing existing puK')
        os.remove('server_puk.pem')
    prk,puk=  generate_rsa_key_pair()
    with open("server_puk.pem", "wb") as file:
        file.write(puk)
    print('RSA key pair Generated Successfully')

    X, y = load_test_data(test_data_path)
    strategy = CustomFedAvg(
        fraction_fit=1,
        min_available_clients=2,
        min_evaluate_clients=2,
        min_fit_clients=2,
        on_fit_config_fn=config_func,
        on_evaluate_config_fn=config_func,
        num_rounds=num_rounds,
        rsa_prk=prk
    )
    strategy.test_data_path = test_data_path
    try:
        rounds = num_rounds
        fl.server.start_server(
            server_address="localhost:5040",
            config=fl.server.ServerConfig(num_rounds=rounds),
            strategy=strategy
        )
    except Exception as e:
        print('Unexpected Error Occurred')
        print(f"Unexpected Error: {e}")
    finally:
        if(strategy.model is not None):
            save_model(strategy.model, 'multitask.pth')
        else:
            sys.exit(0)

if __name__ == "__main__":
    test_data_path = sys.argv[1]
    num_rounds = int(sys.argv[2])
    main(test_data_path, num_rounds)
