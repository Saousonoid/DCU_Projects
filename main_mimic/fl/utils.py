import torch
import torch.nn as nn
import torch.nn.functional as F
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP, AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
import numpy as np
from sklearn.metrics import accuracy_score

# Custom Dataset class for PyTorch DataLoader
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

# SingleTask Neural Network Model
class SingleTaskNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SingleTaskNN, self).__init__()
        # Shared hidden layers
        self.shared_hidden1 = nn.Linear(input_dim, hidden_dim)
        self.shared_hidden2 = nn.Linear(hidden_dim, hidden_dim)  # Second shared layer
        self.personalized_hidden = nn.Linear(hidden_dim, hidden_dim)  # Client-specific personalized layer
        self.output = nn.Linear(hidden_dim, output_dim)  # Final output layer
        self.dropout = nn.Dropout(p=0.4)  # Dropout for regularization
    
    def forward(self, x):
        x = F.relu(self.shared_hidden1(x))
        x = F.relu(self.shared_hidden2(x))
        x = F.relu(self.personalized_hidden(x))  
        x = self.dropout(x) 
        return self.output(x) #BCEwithLoss output

    # Training function for the model
    def train_model(self, train_loader, criterion, optimizer, epochs=10, max_norm=3):
        self.train()
        for epoch in range(epochs):
            for data, target in train_loader:
                optimizer.zero_grad()
                output = self(data).squeeze()
                loss = criterion(output, target)
                loss.backward()
                
                # Apply gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)
                
                optimizer.step()
            
            # Print loss at the end of each epoch for monitoring
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# MultiTask Neural Network Model
class MultiTaskNN(nn.Module):
    def __init__(self, shared_hidden, personalized_hidden, output_heads):
        super(MultiTaskNN, self).__init__()
        self.shared_hidden = shared_hidden  # Shared hidden layers
        self.personalized_hidden = nn.ModuleList(personalized_hidden)  # List of personalized layers
        self.output_heads = nn.ModuleList(output_heads)  # List of output heads (one per task)
    
    def forward(self, x, task_id):
        x = self.shared_hidden(x)
        x = self.personalized_hidden[task_id-1](x)
        if 0 < task_id <= len(self.output_heads):
            return self.output_heads[task_id-1](x)
        else:
            raise ValueError(f"Invalid task_id: should be between 0 and {len(self.output_heads)}")

# Model testing function for both SingleTask and MultiTask models

def test_model(model, dataloader, criterion, task_id=None):
    model.eval()
    total_loss = 0
    predictions, true_values = [], []
    print('Initiating Stats')
    with torch.no_grad():
        for data, target in dataloader:
            if task_id is not None:
                # Forward pass for multitask model
                output = model(data, task_id).squeeze()
            else:
                # Forward pass for single task model
                output = model(data).squeeze()
            loss = criterion(output, target)
            total_loss += loss.item()
            predictions.extend(torch.sigmoid(output).numpy())  # Apply sigmoid for binary classification
            true_values.extend(target.numpy())
    # Convert probabilities to binary predictions
    avg_loss = total_loss / len(dataloader)
    predictions = (np.array(predictions) > 0.5).astype(int)  
    metric = accuracy_score(true_values, predictions)  # Calculate accuracy
    
    return avg_loss, metric

# Threshold function for checking the performance of MultiTask models during training
def check_thres(mt_accuracy: float, mt_loss: float, st_accuracy: float, st_loss: float, accuracy_threshold: float, loss_threshold: float, min_acc) -> bool:
    '''
    Compares the performance of a MultiTask model against a SingleTask model.

    mt_accuracy: MultiTask model's accuracy for a specific task
    mt_loss: MultiTask model's loss for a specific task
    st_accuracy: SingleTask model's accuracy for the same task
    st_loss: SingleTask model's loss for the same task
    accuracy_threshold: Minimum ratio of mt_accuracy to st_accuracy
    loss_threshold: Maximum ratio of mt_loss to st_loss
    min_acc: Minimum acceptable accuracy for the MultiTask model
    '''


    print('Checking Conditions')

    accuracy_pass = mt_accuracy >= (st_accuracy * accuracy_threshold) and mt_accuracy >= min_acc
    loss_pass = mt_loss <= (st_loss * loss_threshold) and mt_loss < 1
    return accuracy_pass and loss_pass

# Save the model's state dictionary to a file
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved at: {path}")

# RSA Key Pair Generation
def generate_rsa_key_pair():
    '''
    Generates a 2048-bit RSA key pair (public and private keys).
    Returns:
    private_key: RSA private key
    public_key: RSA public key
    '''
    key = RSA.generate(2048)
    private_key = key.export_key()
    public_key = key.publickey().export_key()
    return private_key, public_key

# RSA Encryption
def rsa_encrypt(public_key: bytes, data: bytes) -> bytes:
    '''
    Encrypts data using RSA encryption with the provided public key.

    public_key: RSA public key for encryption
    data: Data to encrypt
    '''
    rsa_key = RSA.import_key(public_key)
    cipher = PKCS1_OAEP.new(rsa_key)
    encrypted_data = cipher.encrypt(data)
    return encrypted_data

# RSA Decryption
def rsa_decrypt(private_key: bytes, encrypted_data: bytes) -> bytes:
    '''
    Decrypts RSA-encrypted data using the provided private key.

    private_key: RSA private key for decryption
    encrypted_data: Encrypted data to decrypt
    '''
    rsa_key = RSA.import_key(private_key)
    cipher = PKCS1_OAEP.new(rsa_key)
    decrypted_data = cipher.decrypt(encrypted_data)
    return decrypted_data

# Load RSA Key from File
def load_key(filename: str) -> bytes:
    '''
    Loads an RSA key (either public or private) from a file.

    filename: Path to the key file
    '''
    with open(filename, 'rb') as f:
        key = f.read()
    return key

# AES Key Generation
def generate_aes_key() -> bytes:
    '''
    Generates a 16-byte AES key for symmetric encryption.

    '''
    return get_random_bytes(16)

# AES Data Encryption
def encrypt_data(aes_key: bytes, data: bytes) -> bytes:
    '''
    Encrypts data using AES encryption with the provided key.

    aes_key: AES key for encryption
    data: Data to encrypt
    '''
    iv = get_random_bytes(16)  # Generate a random initialization vector (IV)
    cipher = AES.new(aes_key, AES.MODE_CBC, iv)
    encrypted_data = cipher.encrypt(pad(data, AES.block_size))  # Pad data to block size and encrypt
    return iv + encrypted_data  # Return IV + encrypted data

# AES Data Decryption
def decrypt_data(aes_key: bytes, encrypted_data: bytes) -> bytes:
    '''
    Decrypts AES-encrypted data using the provided key.

    aes_key: AES key for decryption
    encrypted_data: Data to decrypt (including the IV)
    '''
    iv = encrypted_data[:16] 
    encrypted_data = encrypted_data[16:]  
    cipher = AES.new(aes_key, AES.MODE_CBC, iv)
    decrypted_data = unpad(cipher.decrypt(encrypted_data), AES.block_size)  
    return decrypted_data
