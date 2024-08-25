import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Parameters, FitIns, FitRes, Scalar, EvaluateIns
from typing import Dict
import pandas as pd
import numpy as np
import torch
from utils import *
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import sys, os
from typing import Callable, Dict, List, Optional, Tuple, Union
import warnings
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

warnings.filterwarnings("ignore", category=DeprecationWarning)

rounds = 1

def load_test_data(file_path):
    '''Load and preprocess the test data from a CSV file. This function scales the feature values 
       (excluding the target labels) and returns the features and target labels separately.'''
    data = pd.read_csv(file_path, index_col=0)
    X = data.drop(['diabetes','ckd'], axis=1).values
    y1 = data['ckd'].values
    y2 = data['diabetes'].values
    scaler = StandardScaler()
    X[:, :-2] = scaler.fit_transform(X[:, :-2])
    return X, [y1, y2]

class CustomFedAvg(FedAvg):
    '''Custom implementation of the FedAvg strategy that includes additional logic 
       for encryption, weight aggregation based on task accuracy, and model evaluation.'''
    def __init__(self, num_rounds: int, rsa_prk, x, y, **kwargs):
        '''Initialize the CustomFedAvg strategy.
        
        Parameters:
        - num_rounds: Total number of federated learning rounds to be conducted.
        - rsa_prk: RSA private key for decrypting client data.
        - x, y: The test data (features and labels) for evaluating the global model.
        '''
        super().__init__(**kwargs)
        self.num_rounds = num_rounds
        self.current_round = 0
        self.accuracy_list = []
        self.loss_list = []
        self.avg_accuracy = 0
        self.avg_loss = float(1.00000)
        self.model = None
        self.optim_model = None
        self.optim_acc = 0
        self.optim_loss = 10
        self.client_keys = {}  # Dictionary to store each client's RSA decryption key
        self.rsa_prk = rsa_prk  # RSA private key
        self.x = x  # Test features
        self.y = y  # Test labels

    def initialize_parameters(self, client_manager):
        '''This method is overridden to avoid any parameter initialization for clients 
           at the start of the federated learning process.'''
        return None

    def aggregate_weights(self, rnd, weight_list) -> torch.Tensor:
        '''Aggregate model weights across clients, with optional accuracy-based weighting.
        
        - rnd: The current round number.
        - weight_list: List of encrypted model weights received from clients.
        
        The function decrypts the weights, calculates accuracy-based weights for each client,
        and aggregates the weights accordingly.
        '''
        final_list = []
        for weight_bytes in weight_list:
            weight_array = np.frombuffer(weight_bytes, dtype=np.float32)
            weight_tensor = torch.tensor(weight_array)
            final_list.append(weight_tensor)
        
        if rnd > 1:    
            # Calculate accuracy-based weights for aggregation
            max_accuracy = max(self.accuracy_list)
            raw_weights = [max_accuracy / acc for acc in self.accuracy_list]
            sum_raw_weights = sum(raw_weights)
            normalized_weights = [w / sum_raw_weights for w in raw_weights]
            total_weight = torch.zeros_like(normalized_weights[0] * final_list[0].clone().detach())
            for i, weight in enumerate(final_list):
                total_weight += normalized_weights[i] * weight.clone().detach()
        else:
            # Standard averaging for the first round
            total_weight = torch.zeros_like(final_list[0].clone().detach())
            for weight in final_list:
                total_weight += weight.clone().detach()
        
        return total_weight / len(final_list)  # Return the aggregated weight

    def aggregate_fit(self, rnd: int, results, failures) -> Tuple[Parameters, Dict[str, Scalar]]:
        '''Aggregate client model updates after each training round.
        
        - rnd: Current round number.
        - results: List of client results after local training.
        - failures: List of clients that failed during the round.
        
        This function sorts the results, decrypts client weights, aggregates them,
        and evaluates the aggregated model on the test set.
        '''
        results.sort(key=lambda result: result[1].metrics["task_no"])  # Sort results by task number
        if rnd == 1:
            for result in results:
                client_id = result[0].cid  # Get client ID
                s_key = result[1].parameters.tensors[-1]  # Adjusted index for the key
                self.client_keys[client_id] = rsa_decrypt(self.rsa_prk, s_key)  # Decrypt and store client's RSA key

        # Decrypt and extract weights for aggregation
        weight_list1, weight_list2, personalized_hidden_list, output_list = zip(*[
            (decrypt_data(self.client_keys[result[0].cid], result[1].parameters.tensors[0]),
             decrypt_data(self.client_keys[result[0].cid], result[1].parameters.tensors[1]),
             decrypt_data(self.client_keys[result[0].cid], result[1].parameters.tensors[2]),
             decrypt_data(self.client_keys[result[0].cid], result[1].parameters.tensors[3])) for result in results
        ])
        
        aggregated_weights1 = self.aggregate_weights(rnd, weight_list1)
        aggregated_weights2 = self.aggregate_weights(rnd, weight_list2)

        # Combine aggregated shared layers and personalized layers into a multitask model
        multitask_layers = Parameters(
            tensors=[aggregated_weights1.numpy().tobytes(), aggregated_weights2.numpy().tobytes()] +
                    list(personalized_hidden_list) + list(output_list),
            tensor_type="NUMPY"
        )

        # Evaluate the aggregated model on the test data
        print(self.multitask_evaluate(rnd, multitask_layers))
        global rounds
        rounds += 1
        metrics: Dict[str, Scalar] = {
            "round": rnd,
            "client_count": len(results)
        }

        return multitask_layers, metrics

    def aggregate_evaluate(self, rnd: int, results, failures) -> Dict[str, float]:
        '''Aggregate evaluation results from clients after each round.
        
        - rnd: Current round number.
        - results: List of client evaluation results.
        - failures: List of clients that failed during evaluation.
        
        This function calculates the average accuracy and loss across clients and determines
        if the aggregated model should be saved based on performance thresholds.
        '''
        accuracies, losses = zip(*[(client[1].metrics['accuracy'], client[1].loss) for client in results])

        avg_accuracy = sum(accuracies) / len(accuracies)
        avg_loss = float(sum(losses) / len(losses))

        self.avg_accuracy = max(self.avg_accuracy, avg_accuracy)
        self.avg_loss = min(self.avg_loss, avg_loss)

        mtl_acc_avg = sum(self.accuracy_list) / len(self.accuracy_list)
        mtl_loss_avg = sum(self.loss_list) / len(self.loss_list)
        max_accuracy, min_accuracy = max(self.accuracy_list), min(self.accuracy_list)
        if mtl_acc_avg > self.optim_acc and mtl_loss_avg <= self.optim_loss and abs(max_accuracy - min_accuracy) <= 0.10:
            self.optim_model = self.model  # Save the current model as the optimal model
            self.optim_acc = mtl_acc_avg
            self.optim_loss = mtl_loss_avg
        
        for i in range(len(self.accuracy_list)):
            chk = check_thres(self.accuracy_list[i], self.loss_list[i], self.avg_accuracy, self.avg_loss, 0.85, 1.5, 0.6)
            if not chk:
                if rnd == self.num_rounds:
                    self.model = self.optim_model
                    print('Training Completed, Optimal MTL Model Metrics:')
                    print(f'Optimal Average Accuracy: {self.optim_acc:.4f}')
                    print(f'Optimal Average Loss: {self.optim_loss:.4f}')
                return None, {"avg_accuracy: ": avg_accuracy, "avg_loss: ": avg_loss}
        
        print('Performance Conditions Met, Ending all Round Training')
        sys.exit(0)
        return None, {"avg_accuracy": avg_accuracy, "avg_loss": avg_loss}

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        '''Configure the training instructions sent to each client.
        
        - server_round: The current round number.
        - parameters: The global model parameters to be sent to the clients.
        - client_manager: Manages the connected clients.
        
        This function encrypts the global model parameters before sending them to the clients.
        '''
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
            client_key = self.client_keys[client_id]  # Retrieve the client-specific RSA key
            enc_weights = encrypt_data(client_key, parameters.tensors[0])  # Encrypt the global model weights
            enc_outputs = [encrypt_data(client_key, output) for output in parameters.tensors[1:]]
            enc_parameters = Parameters(
                tensors=[enc_weights] + enc_outputs,
                tensor_type="NUMPY"
            )

            config = {}
            if self.on_fit_config_fn is not None:
                config = self.on_fit_config_fn(server_round)
            fit_ins = FitIns(enc_parameters, config)
            fit_instructions.append((client, fit_ins))
        
        return fit_instructions

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, EvaluateIns]]:
        '''Configure the evaluation instructions sent to each client.
        
        - server_round: The current round number.
        - parameters: The global model parameters to be sent to the clients for evaluation.
        - client_manager: Manages the connected clients.
        
        This function handles how the global model is evaluated by the clients.
        '''
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
        '''Override the evaluate function if any custom evaluation logic is required.
        Currently, it does nothing and returns None.
        '''
        return None

    def multitask_evaluate(self, server_round: int, parameters: Parameters):
        '''Evaluate the aggregated model on multiple tasks using the test dataset.
        
        - server_round: The current round number.
        - parameters: The aggregated global model parameters.
        
        This function builds the global model from the aggregated parameters, evaluates it
        on the test dataset for each task, and updates the accuracy and loss lists.
        '''
        self.accuracy_list = []
        self.loss_list = []
        X_test = self.x
        y_test_task1, y_test_task2 = self.y[0], self.y[1]

        input_dim = self.x.shape[1]
        hidden_dim = input_dim
        no_outputs = len(parameters.tensors[2:]) // 2  # Adjust for the personalized layers and output layers
        output_dims = [1] * no_outputs

        # Define the shared layers and task-specific layers
        shared_hidden_layer1 = torch.nn.Linear(input_dim, hidden_dim)
        shared_hidden_layer2 = torch.nn.Linear(hidden_dim, hidden_dim)
        
        personalized_hidden_layers = [torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(no_outputs)]
        output_layers = [torch.nn.Linear(hidden_dim, output_dim) for output_dim in output_dims]

        # Build the multitask model with the shared layers
        model = MultiTaskNN(
            shared_hidden=nn.Sequential(shared_hidden_layer1, shared_hidden_layer2), 
            personalized_hidden=nn.ModuleList(personalized_hidden_layers),
            output_heads=nn.ModuleList(output_layers)
        )
        self.model = model

        # Load the shared weights from the aggregated model
        shared_weights1 = parameters.tensors[0]
        shared_weights2 = parameters.tensors[1]

        model.shared_hidden[0].weight.data.copy_(torch.tensor(np.frombuffer(shared_weights1, dtype=np.float32)).view_as(model.shared_hidden[0].weight.data))
        model.shared_hidden[1].weight.data.copy_(torch.tensor(np.frombuffer(shared_weights2, dtype=np.float32)).view_as(model.shared_hidden[1].weight.data))

        # Load the personalized hidden layers and output layers
        for i in range(no_outputs):
            personalized_weights = parameters.tensors[2 + i]
            model.personalized_hidden[i].weight.data.copy_(torch.tensor(np.frombuffer(personalized_weights, dtype=np.float32)).view_as(model.personalized_hidden[i].weight.data))
            
            output_weights = parameters.tensors[2 + no_outputs + i]
            model.output_heads[i].weight.data.copy_(torch.tensor(np.frombuffer(output_weights, dtype=np.float32)).view_as(model.output_heads[i].weight.data))
        
        # Evaluate on task 1 and task 2 using the complete model
        test_dataset_task1 = CustomDataset(X_test, y_test_task1)
        test_loader_task1 = DataLoader(test_dataset_task1, batch_size=32, shuffle=False)

        test_dataset_task2 = CustomDataset(X_test, y_test_task2)
        test_loader_task2 = DataLoader(test_dataset_task2, batch_size=32, shuffle=False)
        criterion = torch.nn.BCEWithLogitsLoss()

        # Evaluate Task 1
        loss1, accuracy1 = test_model(model, test_loader_task1, criterion=criterion, task_id=1)
        self.accuracy_list.append(accuracy1)
        self.loss_list.append(loss1)

        # Evaluate Task 2
        loss2, accuracy2 = test_model(model, test_loader_task2, criterion=criterion, task_id=2)
        self.accuracy_list.append(accuracy2)
        self.loss_list.append(loss2)

        return {"Loss Task 1: ": float(loss1), "Loss Task 2: ": float(loss2)}, {"accuracy Task 1:": float(accuracy1), "accuracy Task 2:": float(accuracy2)}

def config_func(rnd: int) -> Dict[str, str]:
    '''Generate a configuration dictionary for each round.
    
    - rnd: The current round number.
    
    This function can be used to pass specific settings to clients in each round.
    '''
    config = {
        "global_round": str(rnd),
    }
    return config

def main(test_data_path, num_rounds):
    '''Main function to start the federated learning server and manage the training rounds'''
    
    if os.path.exists('server_puk.pem'):
        print('removing existing puK')
        os.remove('server_puk.pem')
    prk, puk = generate_rsa_key_pair()
    with open("server_puk.pem", "wb") as file:
        file.write(puk)
    print('RSA key pair Generated Successfully')
    X_test, Y_test = load_test_data(test_data_path)

    strategy = CustomFedAvg(
        fraction_fit=1,
        min_available_clients=2,
        min_evaluate_clients=2,
        min_fit_clients=2,
        on_fit_config_fn=config_func,
        on_evaluate_config_fn=config_func,
        num_rounds=num_rounds,
        rsa_prk=prk,
        x=X_test,
        y=Y_test
    )
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
