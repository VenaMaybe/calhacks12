import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import os


CLASS_LABELS = {
    0:  "Rock",
    1:  "Paper",
    2:  "Scissors"
}

NUM_OUTPUTS = 3 
INPUT_DIM = 0 # CHANGE DEPENDING ON OUR DATA READINGS
MODEL_PATH = "mlp_classifier_model.pth" # CHANGE 
SCALER_PATH = "scaler.pkl" # CHANGE
HIDDEN_SIZE = 64
NUM_LAYERS = 2

class myCNN(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, activation, output_dim=3):
        super(myCNN, self).__init__()
        layers = [nn.Linear(input_dim, hidden_size), activation]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), activation]
        layers.append(nn.Linear(hidden_size, output_dim))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)



def feed_through_nn(model, scaler, raw_input_data: pd.Series):

    input_array = raw_input_data.values.astype(np.float32)
    scaled_array = scaler.transform(input_array)
    input_tensor = torch.tensor(scaled_array, dtype=torch.float32)

    model.eval()

    # 5. Make prediction (disable gradient calculation)
    with torch.no_grad():
        # Output is logits (raw scores)
        logits = model(input_tensor)
        
        # Apply Softmax to convert logits to probabilities
        probabilities = torch.softmax(logits, dim=1)
        
        # Get the predicted class index (index with the highest probability)
        predicted_index = torch.argmax(logits, dim=1).item()
        
    # 6. Interpret result
    readable_label = CLASS_LABELS.get(predicted_index, f"Unknown Index: {predicted_index}")
    
    # Extract probability tensor for display
    probs_list = probabilities.squeeze().tolist()
    
    return predicted_index, readable_label, probs_list




def main():
    # TAKE IN DATA INPUT HERE (as a pd dataframe) AND SAVE IT TO LIVE_DATA VAR
    live_data = "placeholder"

    live_output = feed_through_nn(live_data[1])
    print(live_output)
    # also not sure how to return the value in a manner that is readable to the site :( vena help 
