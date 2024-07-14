import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Parameters, FitIns,FitRes,Scalar,EvaluateIns
from typing import Dict
import pandas as pd
import numpy as np
import torch
from utils import MultiTaskNN, test_model,SingleTaskNN
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from utils import CustomDataset
import sys
from typing import Callable, Dict, List, Optional, Tuple, Union
import warnings
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

# warnings.filterwarnings("ignore", category=DeprecationWarning)
rounds=1

# Load and merge test dataset
def load_test_data(file_path):
    data = pd.read_csv(file_path, index_col=0)  # Ignoring the index column
    X = data.drop(['prevalentHyp', 'TenYearCHD'], axis=1).values
    y = data.iloc[:,-2:-1].values


 

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

def aggregate_weights(weight_list) -> torch.Tensor:
    """Aggregate the shared hidden layer weights from clients."""
    final_list = []
    for weight_bytes in weight_list:
    # Deserialize the byte string to a NumPy array
        weight_array = np.frombuffer(weight_bytes, dtype=np.float32)
        
        # Convert the NumPy array to a PyTorch tensor
        weight_tensor = torch.tensor(weight_array)
        
        # Append to the list
        final_list.append(weight_tensor)

    total_weight = torch.zeros_like(final_list[0].clone().detach())
    for weight in final_list:
        total_weight += weight.clone().detach()

    return total_weight / len(final_list)
    
    

class CustomFedAvg(FedAvg):


    def __init__(self, num_rounds: int,avg_accuracy,avg_loss, **kwargs):
        super().__init__(**kwargs)
        self.num_rounds = num_rounds
        self.current_round = 0
        self.avg_accuracy=avg_accuracy
        self.avg_loss=avg_loss


    def initialize_parameters(self, client_manager):
        """Override this method to prevent sending initial parameters in round 0."""
        # Return None to indicate no initial parameters are sent
        return None



        
    def aggregate_fit(self,rnd: int,results,failures,)-> Tuple[Parameters, Dict[str, Scalar]]:
        print("I am here")
        print(results)
        weight_list, output_list = zip(*[(result[1].parameters.tensors[0], result[1].parameters.tensors[1]) for result in results])
        aggregated_weights = aggregate_weights(weight_list)
        global rounds
        rounds+=1
        # Return aggregated weights
        aggregated_parameters = Parameters(
            tensors=[aggregated_weights.numpy().tobytes()]+list(output_list),
            tensor_type="NUMPY"
        )

        metrics: Dict[str, Scalar] = {
            "round": rounds,
            "client_count": len(results)
        }
        print('# Return FitRes instead of FitIns')
        return aggregated_parameters, metrics

    


    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        parameters=Parameters(
            tensors=[parameters.tensors[0]],
            tensor_type="NUMPY"
        )
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients] 


    def evaluate(self, server_round: int, parameters: Parameters):
        print(server_round)
        if server_round ==0:
            # Skip evaluation for intermediate rounds
            print('skip initial evaluation')
            return None
        

        X_test, y_test = load_test_data(self.test_data_path)
        test_dataset = CustomDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        input_dim = X_test.shape[1]
        # print(input_dim)
        hidden_dim = 20
        output_dims = [1, 1]  # Adjust according to the number of tasks
        shared_hidden_layer = torch.nn.Linear(input_dim, hidden_dim)
        output_layers=[
                torch.nn.Linear(hidden_dim, output_dim) for output_dim in output_dims
            ]
        print('# Initialize the model')
        model =       MultiTaskNN(shared_hidden=shared_hidden_layer, output_heads=output_layers)
        print('Model Setup Successful')
        shared_weights=parameters.tensors[0]

        model.shared_hidden.weight.data.copy_(
                torch.tensor(np.frombuffer(shared_weights, dtype=np.float32)).view_as(model.shared_hidden.weight.data))


        
        print('# Load output layer weights (if any)')
        for i, output_layer in enumerate(model.output_heads):
            output_layer.weight.data.copy_(torch.tensor(
            np.frombuffer(parameters.tensors[i+1], dtype=np.float32)).view_as(output_layer.weight.data))
                
        print("\nFinal Evaluation on Test Data using Multi-Task Model:")

        test_df = pd.read_csv('test.csv')
        X_test = test_df.iloc[:, 1:-2].values
        y_test_task1 = test_df.iloc[:, -2].values
        y_test_task2 = test_df.iloc[:, -1].values
        print('do data now')

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_test)
        print('# Create datasets and dataloaders for testing')
        test_dataset_task1 = CustomDataset(X_scaled, y_test_task1)
        test_loader_task1 = DataLoader(test_dataset_task1, batch_size=32, shuffle=False)

        test_dataset_task2 = CustomDataset(X_scaled, y_test_task2)
        test_loader_task2 = DataLoader(test_dataset_task2, batch_size=32, shuffle=False)
        criterion = torch.nn.BCEWithLogitsLoss()
        print('Extract Model Metrics')
        loss1, accuracy1=test_model(model, test_loader_task1, criterion=criterion, task_id=1)
        print("Loss Task 1: "+str(loss1))
        print("Accuracy Task 1: "+str(accuracy1))
        loss2, accuracy2=test_model(model, test_loader_task2, criterion=criterion, task_id=2)
        print('Loss Task 2: '+str(loss2))
        print('Accuracy Task 2: '+str(accuracy2))

        return {"Loss Task 1: ":float(loss1),"Loss Task 2: ":float(loss2)}, {"accuracy Task 1:": float(accuracy1),"accuracy Task 2:": float(accuracy2)}



def config_func(rnd: int) -> Dict[str, str]:
    """Return a configuration with global epochs."""
    config = {
        "global_round": str(rnd),
    }
    return config
def aggregate_metrics(metrics: List[Dict[str, float]])-> Dict[str, float]:
    num_ex=0
    accuracy=0
    loss=0
    print('passing by')
    print(metrics)
    for values in metrics:

        num_ex+=values[0]
        accuracy+=values[1]["accuracy"]

    accuracy=accuracy/len(metrics)
    return {"Number of Examples": num_ex, "Accuracy": accuracy}

def main(test_data_path):
    X, y = load_test_data(test_data_path)
    strategy = CustomFedAvg(
        fraction_fit=1,
        min_available_clients=2,
        min_evaluate_clients=2,
        min_fit_clients=2,
        on_fit_config_fn=config_func,
        on_evaluate_config_fn=config_func,
        # evaluate_metrics_aggregation_fn=aggregate_metrics,
        num_rounds=10,
        avg_accuracy=0,
        avg_loss=0
    )
    strategy.test_data_path = test_data_path
    try:
        fl.server.start_server(
            server_address="localhost:5040",
            config=fl.server.ServerConfig(num_rounds=10),
            strategy=strategy
        )
    except Exception as e:
        print(e)
        sys.exit(0)

if __name__ == "__main__":
    test_data_path = sys.argv[1]  # Path to test dataset
    main(test_data_path)
