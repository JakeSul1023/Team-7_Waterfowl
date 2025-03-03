import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


def load_data(file_path, num_ducks=5):
    df = pd.read_csv(file_path)
    
    duck_ids = df['tag-local-identifier'].unique()[:num_ducks]
    #filter
    df_duck = df[df['tag-local-identifier'].isin(duck_ids)].copy()
    
    # Convert 
    df_duck['timestamp'] = pd.to_datetime(df_duck['timestamp'])
    
    # Sort data by timestamp
    df_duck = df_duck.sort_values(by='timestamp')
    
    # Reset index
    df_duck.reset_index(drop=True, inplace=True)
    
    # Normalize data using StandardScaler 
    scaler = StandardScaler()
    df_duck[['location-lat', 'location-long']] = scaler.fit_transform(df_duck[['location-lat', 'location-long']])
    
   
    features = df_duck[['location-lat', 'location-long']].values
    coords = df_duck[['timestamp', 'location-lat', 'location-long']].values
    return features, coords, scaler, duck_ids

# RNN Model Definition
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN layer
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        # Fully connected layer for output (latitude, longitude)
        self.fc = nn.Linear(hidden_size, 2)  # Output 2 values (latitude, longitude)

    def forward(self, x):
        # Initial hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # RNN output
        out, _ = self.rnn(x, h0)
        
        # We only need the last output of the sequence for prediction
        out = out[:, -1, :]
        
        # Pass the output through the fully connected layer
        out = self.fc(out)
        return out

# Prepare data (grab the first 5 ducks dynamically)
file_path = 'GNN_inital_test.csv'

features, coords, scaler, duck_ids = load_data(file_path)

# Convert to torch tensors
seq_length = features.shape[0] 
input_size = features.shape[1]  

# Check the number of elements in features
total_elements = seq_length * input_size

batch_size = len(duck_ids)
if total_elements % batch_size != 0:
    print(f"Warning: Total elements {total_elements} is not divisible by batch size {batch_size}.")
    trunc_length = seq_length - (seq_length % batch_size)  
    features = features[:trunc_length]

# Reshape data to (batch_size, seq_length, input_size)
inputs = torch.tensor(features).float().reshape(-1, batch_size, input_size)  # Shape: (num_batches, batch_size, input_size)

# Create a DataLoader for batching
dataset = TensorDataset(inputs)
dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

# Initialize model
hidden_size = 64  
num_layers = 2  
model = RNNModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop with epochs
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    
    for batch in dataloader:
        inputs = batch[0]
        
        # Forward pass
        outputs = model(inputs)
        
        # Dummy target data (Replace with actual coordinates of the next day)
        targets = torch.zeros_like(outputs)  # This should be the actual target latitude/longitude values
        
        # Compute loss
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# Predict the next 5 days for all ducks together
model.eval()
with torch.no_grad():
    # Aggregate predictions for all ducks
    all_predictions = []
    for batch in dataloader:
        inputs = batch[0]
        
        # Predict the next day's coordinates for all ducks in the batch
        outputs = model(inputs)
        
        # Inverse scale the predicted coordinates (latitude, longitude)
        predicted_coords = scaler.inverse_transform(outputs.numpy())
        
        all_predictions.append(predicted_coords)

print("Predictions for the next 5 days for all ducks:")
for day in range(15):  # set number of days
    print(f"Day {day+1}:")
    for i, duck_id in enumerate(duck_ids):
        predicted_lat, predicted_lon = all_predictions[day][i]
        print(f"  Duck {duck_id}: Latitude = {predicted_lat:.4f}, Longitude = {predicted_lon:.4f}")
    print("=" * 40)  # seperates
