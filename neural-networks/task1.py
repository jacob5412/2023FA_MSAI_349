import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
print('Data Loading and Preprocesssing...')
# Load and preprocess the data
def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].map({'Good': 0, 'Neutral': 1, 'Bad': 2}).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
# Load the dataset
train_X, train_y = load_data('/content/three_train.csv')
valid_X, valid_y = load_data('/content/three_valid.csv')
test_X, test_y = load_data('/content/three_test.csv')
# Create data loaders
batch_size = 2
train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=batch_size)
valid_loader = DataLoader(TensorDataset(valid_X, valid_y), batch_size=batch_size)
test_loader = DataLoader(TensorDataset(test_X, test_y), batch_size=batch_size)
# Define the neural network class
class SimpleFFNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleFFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x
# Softmax function implementation
def softmax(x):
    return torch.exp(x) / torch.sum(torch.exp(x), axis=1).view(-1, 1)

# Function to evaluate the network
def evaluate_network(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())
    return all_labels, all_preds
# Define hyperparameters
epochs = 30
input_size = 3
hidden_size = 16
output_size = 3
learning_rate = 0.01
epochs = 30
print('initialize the model...')
# Initialize the model, criterion, and optimizer
model = SimpleFFNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
# Function to calculate accuracy
def calculate_accuracy(model, data_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total
# train the network and record losses and accuracies
def train_network(model, criterion, optimizer, train_loader, valid_loader, epochs=10):
    train_losses, valid_losses = [], []
    train_accuracies, valid_accuracies = [], []

    # Initialize tqdm progress bar
    pbar = tqdm(total=epochs, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}", colour='blue')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accuracy = calculate_accuracy(model, train_loader)
        train_accuracies.append(train_accuracy)

        model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
        avg_valid_loss = total_loss / len(valid_loader)
        valid_losses.append(avg_valid_loss)
        valid_accuracy = calculate_accuracy(model, valid_loader)
        valid_accuracies.append(valid_accuracy)

        # Update tqdm bar
        pbar.update(1)
        pbar.set_description(f"Epoch {epoch + 1}/{epochs}")

        print(f"Epoch {epoch + 1}/{epochs} complete: Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_valid_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {valid_accuracy:.2f}%")

    pbar.close()
    return train_losses, valid_losses, train_accuracies, valid_accuracies
print('Training the Model...')
# Train the model and get losses and accuracies
train_losses, valid_losses, train_accuracies, valid_accuracies = train_network(model, criterion, optimizer, train_loader, valid_loader, epochs)
# plot learning curve
def plot_loss(train_losses, valid_losses):
    # Plotting the learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.title('Learning Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
#plot accuracy curve
def plot_accuracy(train_accuracies, valid_accuracies):
    # Plotting the accuracy curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(valid_accuracies, label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
# plot metrics
def metrics(model, test_loader):
    # Evaluate the model
    labels, preds = evaluate_network(model, test_loader)
    conf_matrix = confusion_matrix(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    # Print results
    print("Confusion Matrix:\n", conf_matrix)
    print("F1 Score:", f1)
    return conf_matrix
#plot confusion matrix
def plot_matrix(conf_matrix):
    # Plotting the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Reds')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()
print('Plot Learning Curve...')
plot_loss(train_losses, valid_losses)
print('Plot Accuracy Curve...')
plot_accuracy(train_accuracies, valid_accuracies)
print('Finding: F1 Score and confusion Matrix')
conf_matrix = metrics(model, test_loader)
plot_matrix(conf_matrix)

print('Experiment with different hyperparameters...')
# Hyperparameters for the experiments
settings = [
    {"lr": 0.001, "hidden_units": 8, "epochs": 20},
    {"lr": 0.0001, "hidden_units": 32, "epochs": 25},
    {"lr": 0.05, "hidden_units": 64, "epochs": 30}
]

# Loop through each set of hyperparameters
for setting in settings:
    print(f"Starting training with learning rate: {setting['lr']}, hidden units: {setting['hidden_units']}, epochs: {setting['epochs']}")

    # Model initialization
    model = SimpleFFNN(input_size, setting['hidden_units'], output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=setting['lr'])

    # Train the model
    train_losses, valid_losses, train_accuracies, valid_accuracies = train_network(
        model, criterion, optimizer, train_loader, valid_loader, setting['epochs']
    )
    plot_loss(train_losses, valid_losses)
    conf_matrix = metrics(model, test_loader)
    plot_matrix(conf_matrix)

