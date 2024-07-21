import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Parameters, FitIns,FitRes,Scalar,EvaluateIns
from typing import Dict
import pandas as pd
import numpy as np
import torch
from utils import MultiTaskNN, test_model,SingleTaskNN, check_thres, save_model
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from utils import CustomDataset
import sys
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
    def __init__(self, num_rounds: int, **kwargs):
        super().__init__(**kwargs)
        self.num_rounds = num_rounds
        self.current_round = 0
        self.accuracy_list = []
        self.loss_list = []
        self.avg_accuracy = 0
        self.avg_loss = float(1.00000)
        self.model = None  

    def initialize_parameters(self, client_manager):
        return None

    def aggregate_fit(self, rnd: int, results, failures) -> Tuple[Parameters, Dict[str, Scalar]]:
        weight_list, output_list = zip(*[(result[1].parameters.tensors[0], result[1].parameters.tensors[1]) for result in results])
        aggregated_weights = aggregate_weights(weight_list)
        global rounds
        rounds += 1
        aggregated_parameters = Parameters(
            tensors=[aggregated_weights.numpy().tobytes()] + list(output_list),
            tensor_type="NUMPY"
        )
        metrics: Dict[str, Scalar] = {
            "round": rounds,
            "client_count": len(results)
        }
        return aggregated_parameters, metrics

    def aggregate_evaluate(self, rnd: int, results, failures) -> Dict[str, float]:
        accuracies, losses = zip(*[(client[1].metrics['accuracy'], client[1].loss) for client in results])
        avg_accuracy = sum(accuracies) / len(accuracies)
        avg_loss = float(sum(losses) / len(losses))

        self.avg_accuracy = max(self.avg_accuracy, avg_accuracy)
        self.avg_loss = min(self.avg_loss, avg_loss)

        for i in range(len(self.accuracy_list)):
            chk = check_thres(self.accuracy_list[i], self.loss_list[i], self.avg_accuracy, self.avg_loss, 0.85, 1.5)
            if not chk:
                return None, {"avg_accuracy: ": avg_accuracy, "avg_loss: ": avg_loss}
        print('Performance Conditions Met, Ending all Round Training')
        sys.exit(0)
        return None, {"avg_accuracy": avg_accuracy, "avg_loss": avg_loss}

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
        self.accuracy_list = []
        self.loss_list = []
        if server_round == 0:
            return None

        X_test, y_test = load_test_data(self.test_data_path)
        test_dataset = CustomDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        input_dim = X_test.shape[1]
        hidden_dim = 3
        output_dims = [1, 1]
        shared_hidden_layer = torch.nn.Linear(input_dim, hidden_dim)
        output_layers = [torch.nn.Linear(hidden_dim, output_dim) for output_dim in output_dims]

        model = MultiTaskNN(shared_hidden=shared_hidden_layer, output_heads=output_layers)
        self.model = model
        shared_weights = parameters.tensors[0]

        model.shared_hidden.weight.data.copy_(torch.tensor(np.frombuffer(shared_weights, dtype=np.float32)).view_as(model.shared_hidden.weight.data))

        for i, output_layer in enumerate(model.output_heads):
            output_layer.weight.data.copy_(torch.tensor(
                np.frombuffer(parameters.tensors[i + 1], dtype=np.float32)).view_as(output_layer.weight.data))

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
    X, y = load_test_data(test_data_path)
    strategy = CustomFedAvg(
        fraction_fit=1,
        min_available_clients=2,
        min_evaluate_clients=2,
        min_fit_clients=2,
        on_fit_config_fn=config_func,
        on_evaluate_config_fn=config_func,
        num_rounds=num_rounds
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
    finally:
        save_model(strategy.model, 'multitask.pth')

if __name__ == "__main__":
    test_data_path = sys.argv[1]
    num_rounds = int(sys.argv[2])
    main(test_data_path, num_rounds)
