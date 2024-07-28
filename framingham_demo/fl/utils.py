import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP, AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad

# Custom dataset class for PyTorch
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Return data and label as tensors
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

# Single-task neural network model
class SingleTaskNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SingleTaskNN, self).__init__()
        self.shared_hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.shared_hidden(x))
        return self.output(x)
    
    def train_model(self, train_loader, criterion, optimizer, epochs=5):
        self.train()  # Set the model to training mode
        for epoch in range(epochs):
            for data, target in train_loader:
                optimizer.zero_grad()  # Clear gradients
                output = self(data).squeeze()  # Forward pass
                loss = criterion(output, target)  # Compute loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Update weights
                
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Multi-task neural network model
class MultiTaskNN(nn.Module):
    def __init__(self, shared_hidden, output_heads):
        super(MultiTaskNN, self).__init__()
        self.shared_hidden = shared_hidden
        self.output_heads = nn.ModuleList(output_heads)
    
    def forward(self, x, task_id):
        x = F.relu(self.shared_hidden(x))
        if 0 < task_id <= len(self.output_heads):
            return self.output_heads[task_id-1](x)
        else:
            raise ValueError(f"Invalid task_id: should be between 0 and {len(self.output_heads)-1}")

# Test the model on a dataset
def test_model(model, dataloader, criterion, task_id=None):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    predictions, true_values = [], []
    print('Initiating Stats')
    with torch.no_grad():
        for data, target in dataloader:
            if task_id is not None:
                output = model(data, task_id).squeeze()
            else:
                output = model(data).squeeze()
            loss = criterion(output, target)
            total_loss += loss.item()
            predictions.extend(torch.sigmoid(output).numpy())
            true_values.extend(target.numpy())

    avg_loss = total_loss / len(dataloader)
    predictions = (np.array(predictions) > 0.5).astype(int)  # Binarize predictions
    metric = accuracy_score(true_values, predictions)
    
    return avg_loss, metric

# Check if the multi-task model meets specified thresholds
def check_thres(mt_accuracy: float, mt_loss: float, st_accuracy: float, st_loss: float, accuracy_threshold: float, loss_threshold: float) -> bool:
    print('Checking Conditions')

    accuracy_pass = mt_accuracy >= (st_accuracy * accuracy_threshold) and mt_accuracy >= 0.8
    loss_pass = mt_loss <= (st_loss * loss_threshold) and mt_loss <= 0.7
    return accuracy_pass and loss_pass

# Save the model to a file
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved at: {path}")

# RSA Functions
def generate_rsa_key_pair():
    key = RSA.generate(2048)
    private_key = key.export_key()
    public_key = key.publickey().export_key()
    return private_key, public_key

def rsa_encrypt(public_key: bytes, data: bytes) -> bytes:
    rsa_key = RSA.import_key(public_key)
    cipher = PKCS1_OAEP.new(rsa_key)
    encrypted_data = cipher.encrypt(data)
    return encrypted_data

def rsa_decrypt(private_key: bytes, encrypted_data: bytes) -> bytes:
    rsa_key = RSA.import_key(private_key)
    cipher = PKCS1_OAEP.new(rsa_key)
    decrypted_data = cipher.decrypt(encrypted_data)
    return decrypted_data

#Load RSA Public Key
def load_key(filename: str) -> bytes:
    with open(filename, 'rb') as f:
        key = f.read()
    return key


# AES Functions
def generate_aes_key() -> bytes:
    return get_random_bytes(16)  # 16 bytes for AES-128; use 32 bytes for AES-256

def encrypt_data(aes_key: bytes, data: bytes) -> bytes:
    iv = get_random_bytes(16)  # 16 bytes for AES block size
    cipher = AES.new(aes_key, AES.MODE_CBC, iv)
    encrypted_data = cipher.encrypt(pad(data, AES.block_size))
    return iv + encrypted_data  # Prepend the IV for use in decryption

def decrypt_data(aes_key: bytes, encrypted_data: bytes) -> bytes:
    iv = encrypted_data[:16]  # Extract the first 16 bytes as the IV
    encrypted_data = encrypted_data[16:]
    cipher = AES.new(aes_key, AES.MODE_CBC, iv)
    decrypted_data = unpad(cipher.decrypt(encrypted_data), AES.block_size)
    return decrypted_data

