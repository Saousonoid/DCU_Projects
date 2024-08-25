import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Dataset class for handling data
class TaskDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Neural network model for individual tasks
class TaskModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(TaskModel, self).__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.hidden_layer(x))
        return self.output_layer(x)

# Multi-task model for handling multiple related tasks
class MultiTaskModel(nn.Module):
    def __init__(self, shared_hidden, task1_output, task2_output):
        super(MultiTaskModel, self).__init__()
        self.shared_hidden = shared_hidden
        self.task1_output = task1_output
        self.task2_output = task2_output
    
    def forward(self, x, task_id):
        x = F.relu(self.shared_hidden(x))
        if task_id == 1:
            return self.task1_output(x)
        elif task_id == 2:
            return self.task2_output(x)
        else:
            raise ValueError("Invalid task_id: should be 1 or 2")

# Function to evaluate the model's performance
def evaluate_model(model, dataloader, criterion, task_id=None):
    model.eval()
    total_loss = 0
    predictions, true_values = [], []

    with torch.no_grad():
        for data, target in dataloader:
            output = model(data, task_id).squeeze() if task_id else model(data).squeeze()
            loss = criterion(output, target.squeeze())
            total_loss += loss.item()
            predictions.extend(torch.sigmoid(output).numpy())
            true_values.extend(target.numpy())

    avg_loss = total_loss / len(dataloader)
    predictions = (np.array(predictions) > 0.5).astype(int)
    accuracy = accuracy_score(true_values, predictions)

    return avg_loss, accuracy

# Function to load and prepare the data
def load_data(filename):
    df = pd.read_csv(filename)
    X = df.iloc[:, 1:-1].values
    y = df.iloc[:, -1].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val, scaler

# Main function to run the training and evaluation process
def main():
    # Load data for both tasks
    X_train_task1, X_val_task1, y_train_task1, y_val_task1, scaler_task1 = load_data('client1.csv')
    X_train_task2, X_val_task2, y_train_task2, y_val_task2, scaler_task2 = load_data('client2.csv')

    input_dim = X_train_task1.shape[1]
    hidden_dim = 20
    max_rounds = 100
    accuracy_threshold = 0.84

    weight_task1, weight_task2 = 0.5, 0.5

    for round_num in range(max_rounds):
        print(f"\n=== Round {round_num + 1} ===")

        # Initialize models for both tasks
        model_task1 = TaskModel(input_dim=input_dim, hidden_dim=hidden_dim)
        model_task2 = TaskModel(input_dim=input_dim, hidden_dim=hidden_dim)

        # Create DataLoaders for training and validation
        train_loader_task1 = DataLoader(TaskDataset(X_train_task1, y_train_task1), batch_size=32, shuffle=True)
        val_loader_task1 = DataLoader(TaskDataset(X_val_task1, y_val_task1), batch_size=32, shuffle=False)

        train_loader_task2 = DataLoader(TaskDataset(X_train_task2, y_train_task2), batch_size=32, shuffle=True)
        val_loader_task2 = DataLoader(TaskDataset(X_val_task2, y_val_task2), batch_size=32, shuffle=False)

        # Loss function and optimizers
        criterion = nn.BCEWithLogitsLoss()
        optimizer_task1 = optim.Adam(model_task1.parameters(), lr=0.001)
        optimizer_task2 = optim.Adam(model_task2.parameters(), lr=0.001)

        # Training loop for both tasks
        for model, optimizer, loader, criterion in [
            (model_task1, optimizer_task1, train_loader_task1, criterion),
            (model_task2, optimizer_task2, train_loader_task2, criterion)
        ]:
            model.train()
            for epoch in range(5):
                for data, target in loader:
                    optimizer.zero_grad()
                    output = model(data).squeeze()
                    loss = criterion(output, target.squeeze())
                    loss.backward()
                    optimizer.step()

        # Aggregate the shared hidden layer's weights
        shared_hidden_weight = (weight_task1 * model_task1.hidden_layer.weight.data +
                                weight_task2 * model_task2.hidden_layer.weight.data)

        # Update the shared hidden layers in both models
        model_task1.hidden_layer.weight.data = shared_hidden_weight.clone()
        model_task2.hidden_layer.weight.data = shared_hidden_weight.clone()

        # Validation to check performance after aggregation
        val_loss_task1, val_acc_task1 = evaluate_model(model_task1, val_loader_task1, criterion)
        val_loss_task2, val_acc_task2 = evaluate_model(model_task2, val_loader_task2, criterion)

        print(f"Task 1 - Loss: {val_loss_task1:.4f}, Accuracy: {val_acc_task1:.4f}")
        print(f"Task 2 - Loss: {val_loss_task2:.4f}, Accuracy: {val_acc_task2:.4f}")

        # Stop training if accuracy threshold is met
        if (val_acc_task1 + val_acc_task2)/2 >= accuracy_threshold and min(val_acc_task1,val_acc_task2) >= (0.95 *accuracy_threshold):
            print(f"Accuracy threshold met in round {round_num + 1}.")
            break

    # Final model aggregation after training is complete
    shared_hidden_layer = nn.Linear(input_dim, hidden_dim)
    shared_hidden_layer.weight.data = shared_hidden_weight.clone()

    # Construct the multi-task model using the shared hidden layer
    multi_task_model = MultiTaskModel(
        shared_hidden=shared_hidden_layer,
        task1_output=model_task1.output_layer,
        task2_output=model_task2.output_layer
    )

    # Final Evaluation on Test Data
    test_df = pd.read_csv('test.csv')
    X_test = test_df.iloc[:, 1:-2].values
    y_test_task1 = test_df.iloc[:, -2].values
    y_test_task2 = test_df.iloc[:, -1].values

    # Standardize test data
    X_test_scaled_task1 = scaler_task1.transform(X_test)
    X_test_scaled_task2 = scaler_task2.transform(X_test)

    test_loader_task1 = DataLoader(TaskDataset(X_test_scaled_task1, y_test_task1), batch_size=32, shuffle=False)
    test_loader_task2 = DataLoader(TaskDataset(X_test_scaled_task2, y_test_task2), batch_size=32, shuffle=False)

    # Final results on the test set for each task
    print("\nFinal Evaluation:")
    print("Task 1:", evaluate_model(multi_task_model, test_loader_task1, criterion, task_id=1))
    print("Task 2:", evaluate_model(multi_task_model, test_loader_task2, criterion, task_id=2))

if __name__ == '__main__':
    main()
