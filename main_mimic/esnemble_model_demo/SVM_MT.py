import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import argparse


class MultiTaskNN(nn.Module):
    def __init__(self, shared_hidden, personalized_hidden, output_heads):
        super(MultiTaskNN, self).__init__()
        self.shared_hidden = shared_hidden
        self.personalized_hidden = nn.ModuleList(personalized_hidden)
        self.output_heads = nn.ModuleList(output_heads)
    
    def forward(self, x, task_id):
        x = F.relu(self.shared_hidden(x))
        if 0 < task_id <= len(self.output_heads):
            x = F.relu(self.personalized_hidden[task_id-1](x))  # Apply task-specific personalized layer
            return self.output_heads[task_id-1](x)
        else:
            raise ValueError(f"Invalid task_id: should be between 1 and {len(self.output_heads)}")


def load_data(file_path):
    data = pd.read_csv(file_path, index_col=0)
    features = data.drop(['diabetes', 'ckd'], axis=1)
    scale_exclude = ['heart_rate', 'dbp', 'mbp', 'bmi', 'serum_creatinine', 'age', 'glucose', 'creatinine', 
                     'urea_nitrogen', 'albumin', 'hba1c_dia', 'hematocrit_ckd', 'eGFR']
    targets = data[['diabetes', 'ckd']]

    scaler = StandardScaler()
    features[scale_exclude] = scaler.fit_transform(features[scale_exclude])
    return features, targets, scaler

# Prepare Level 0 Multi-Task Model
def train_mtl_model(model, X_train, task_id):
    with torch.no_grad():
        model_output_train = model(torch.tensor(X_train.values, dtype=torch.float32), task_id=task_id).numpy()
    return model_output_train

# Fit Data into Level 1 SVM model
def train_svm_model(stacked_features_train, y_train):
    svm = SVC(kernel='linear')
    svm.fit(stacked_features_train, y_train)
    return svm

# Evaluate MTL model
def evaluate_model(model, X_test, y_test, task_id):
    with torch.no_grad():
        model_output_test = model(torch.tensor(X_test.values, dtype=torch.float32), task_id=task_id).numpy()
    predictions_test = (model_output_test > 0.5).astype(int).flatten()
    accuracy = accuracy_score(y_test, predictions_test)
    return accuracy, model_output_test

# Prepare Level 1 SVM Model
def evaluate_svm_model(svm, stacked_features_test, y_test):
    y_pred = svm.predict(stacked_features_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def main(options):
    # Load and split the data
    X, y, scaler = load_data(options.dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
    # Define input features for each task
    input1 = ['heart_rate', 'dbp', 'mbp', 'bmi', 'serum_creatinine', 'age', 'glucose', 'creatinine', 
              'urea_nitrogen', 'albumin', 'ethnicity', 'gender']
    input2_diabetes = ['hba1c_dia']
    input2_ckd = ['hematocrit_ckd', 'eGFR']
    
    # Prepare tensors for the model
    X1_train_tensor = X_train[input1]
    X_test_tensor_diabetes = X_test[input1 + input2_diabetes]
    X_test_tensor_ckd = X_test[input1 + input2_ckd]

    # Load the pre-trained model with personalized layers
    shared_hidden_lay1 = nn.Linear(len(input1), 12)
    shared_hidden_lay2 = nn.Linear(12, 12)
    
    personalized_hidden_layers = [nn.Linear(12, 12) for _ in range(2)]  # Assuming two tasks
    output_heads = [nn.Linear(12, 1), nn.Linear(12, 1)]
    
    shared_hidden = nn.Sequential(shared_hidden_lay1, shared_hidden_lay2)
    model = MultiTaskNN(shared_hidden, personalized_hidden_layers, output_heads)
    
    # Load the model's state dict (including the personalized layers)
    model.load_state_dict(torch.load(options.multitask_path))
    model.eval()
    
    # Training and evaluating for Diabetes (Task 1)
    model_output_train_diabetes = train_mtl_model(model, X1_train_tensor, task_id=1)
    accuracy_diabetes_mtl, model_output_test_diabetes = evaluate_model(model, X_test_tensor_diabetes[input1], y_test['diabetes'].values, task_id=1)
    print(f"Accuracy of the first level MTL model for Diabetes: {accuracy_diabetes_mtl * 100:.2f}%")
    
    # SVM Training and Evaluation for Diabetes
    stacked_features_train_diabetes = np.hstack((model_output_train_diabetes, X_train[input2_diabetes].values))
    stacked_features_test_diabetes = np.hstack((model_output_test_diabetes, X_test_tensor_diabetes[input2_diabetes].values))
    
    svm_diabetes = train_svm_model(stacked_features_train_diabetes, y_train['diabetes'].values)
    accuracy_diabetes_svm = evaluate_svm_model(svm_diabetes, stacked_features_test_diabetes, y_test['diabetes'].values)
    print(f"Accuracy of the stacked SVM model for Diabetes: {accuracy_diabetes_svm * 100:.2f}%")
    
    # Training and evaluating for CKD (Task 2)
    model_output_train_ckd = train_mtl_model(model, X1_train_tensor, task_id=2)
    accuracy_ckd_mtl, model_output_test_ckd = evaluate_model(model, X_test_tensor_ckd[input1], y_test['ckd'].values, task_id=2)
    print(f"Accuracy of the first level MTL model for CKD: {accuracy_ckd_mtl * 100:.2f}%")
    
    # SVM Training and Evaluation for CKD
    stacked_features_train_ckd = np.hstack((model_output_train_ckd, X_train[input2_ckd].values))
    stacked_features_test_ckd = np.hstack((model_output_test_ckd, X_test_tensor_ckd[input2_ckd].values))
    
    svm_ckd = train_svm_model(stacked_features_train_ckd, y_train['ckd'].values)
    accuracy_ckd_svm = evaluate_svm_model(svm_ckd, stacked_features_test_ckd, y_test['ckd'].values)
    print(f"Accuracy of the stacked SVM model for CKD: {accuracy_ckd_svm * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-task learning with a shared hidden layer and SVM stacking.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the training dataset CSV file.')
    parser.add_argument('--multitask_path', type=str, required=True, help='Path to the MTL Model States.')

    options = parser.parse_args()
    main(options)
