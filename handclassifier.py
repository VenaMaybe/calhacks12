
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib


# this script creates a torch nn object and saves the weights and biases into some directory 

# data shape: vector column, class column 
num_outputs = 3
# change depending on how many possible outputs we have -- for rock, paper, scissors we have 3 

# define nn class 
class myCNN(nn.Module):
    # constructor 
    def __init__(self, input_dim, hidden_size, num_layers, activation, output_dim=num_outputs):
        super(myCNN, self).__init__()
        layers = [nn.Linear(input_dim, hidden_size), activation] # creating input later
        for i in range(num_layers - 1): 
            layers += [nn.Linear(hidden_size, hidden_size), activation] # creating hidden layers for 
        layers.append(nn.Linear(hidden_size, output_dim)) # creating output layer 
        self.model = nn.Sequential(*layers) # chaining layers together 
    def forward(self, x):
        return self.model(x)


def load_data(full_data, batch_size):
    print("fix pls")
    training_data, testing_data = train_test_split(full_data, random_state=42, test_size=0.2)
    # change the scaler depending on data lowk
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    input_col = ["input_col"]
    output_col = ["output_col"]

    training_input = training_data[input_col]
    testing_input = testing_data[input_col]

    training_target = training_data[output_col]
    testing_target = testing_data[output_col]

      # convert to pytorch tensors

    train_target_output =  training_target.values
    train_input_features = scaler_X.fit_transform(training_input)

    test_input_features = scaler_X.transform(testing_input)
    test_target_output = testing_target.values


    X_train = torch.tensor(train_input_features, dtype=torch.float32)
    y_train = torch.tensor(train_target_output, dtype=torch.long)

    X_test = torch.tensor(test_input_features, dtype=torch.float32)
    y_test = torch.tensor(test_target_output, dtype=torch.long)

    # final datasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return (train_loader, test_loader, scaler_X)
    # load the data and prepare tensors
    # normalize data, make tensors, prepare dataset, split into testing and training

def training(dataloader, loss_fn, optimizer, model):
    total_loss = 0
    model.train()
    for batch, (X, y) in enumerate(dataloader): # this will be set to training data
        outputs = model(X) # change to accomodate data
        loss = loss_fn(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Average loss: {avg_loss:.4f}")
    return avg_loss

def test(dataloader,loss_fn, model):
    model.eval()
    num_samples = len(dataloader.dataset)
    test_loss = 0.0
    correct = 0 
    total = 0
    with torch.no_grad():
        for X, y in dataloader:
            outputs = model(X)
            loss = loss_fn(outputs, y)
            test_loss += loss.item() * X.size(0)

            predictions = outputs.argmax(1)
            correct += (predictions==y).sum().item()
            total += y.size(0)
    avg_loss = test_loss / total
    accuracy = correct / total

    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy

def main(save_path="model.pth"):

    # IMPORT THE PD DATAFRAME INTO THE TEST AND TRAIN CODE! READ FROM PATH AND TRAIN


    input_dim = 10
    hidden_size = 64
    num_layers = 2
    num_classes = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = myCNN(input_dim, hidden_size, num_layers, nn.ReLU(), num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loader, test_loader, scaler_X = load_data(full_data, batch_size)
    # --- training loop ---
    for epoch in range(20):
        print(f"Epoch {epoch+1}")
        training(train_loader, loss_fn, optimizer, model)
        test(test_loader, loss_fn, model)

    # --- save the trained model ---
    torch.save(model.state_dict(), save_path)
    joblib.dump(scaler_X, "scaler.pkl")

