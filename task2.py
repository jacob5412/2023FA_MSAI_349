from torchvision import transforms
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
# Load and preprocess the MNIST data
def load_mnist_data(file_path, transform=None):
    data = pd.read_csv(file_path, header=None)
    X = data.iloc[:, 1:785].values / 255.0  # Normalizing the pixel values
    y = data.iloc[:, 0].values
    if transform:
        X = transform(X)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
# Load the MNIST dataset
mnist_train_X, mnist_train_y = load_mnist_data('/content/mnist_train.csv')
mnist_valid_X, mnist_valid_y = load_mnist_data('/content/mnist_valid.csv')
mnist_test_X, mnist_test_y = load_mnist_data('/content/mnist_test.csv')
batch_size = 2
# Create data loaders for MNIST
mnist_train_loader = DataLoader(TensorDataset(mnist_train_X, mnist_train_y), batch_size=batch_size)
mnist_valid_loader = DataLoader(TensorDataset(mnist_valid_X, mnist_valid_y), batch_size=batch_size)
mnist_test_loader = DataLoader(TensorDataset(mnist_test_X, mnist_test_y), batch_size=batch_size)


class MNISTFFNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MNISTFFNN, self).__init__()

        # Check if the number of hidden layers matches the length of the hidden_sizes list
        assert len(hidden_sizes) == 4, "hidden_sizes list should have 4 elements for 4 hidden layers."

        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
# Define hyperparameters for MNIST
hidden_sizes = [256, 64, 64, 32]
mnist_input_size = 784
mnist_output_size = 10
mnist_learning_rate = 0.0001
epochs=30
# Initialize the MNIST model, criterion, and optimizer
mnist_model = MNISTFFNN(mnist_input_size, hidden_sizes, mnist_output_size)
mnist_criterion = nn.CrossEntropyLoss()
mnist_optimizer = optim.Adam(mnist_model.parameters(), lr=mnist_learning_rate)
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
train_losses, valid_losses, train_accuracies, valid_accuracies = train_network(mnist_model, mnist_criterion, mnist_optimizer, mnist_train_loader, mnist_valid_loader, epochs)
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
conf_matrix = metrics(mnist_model, mnist_test_loader)
plot_matrix(conf_matrix)
# Apply Different Parameter and Loss functions
optimizers = {
    'SGD': optim.SGD(mnist_model.parameters(), lr=mnist_learning_rate),
    'Adam': optim.Adam(mnist_model.parameters(), lr=mnist_learning_rate),
    'RMSprop': optim.RMSprop(mnist_model.parameters(), lr=mnist_learning_rate)
}

loss_functions = {
    'CrossEntropy': nn.CrossEntropyLoss(),
    'NLLLoss': nn.NLLLoss()
}

# Function to train the network with different optimizer and loss function
def train_with_different_optimizers_and_losses(model, train_loader, valid_loader, optimizers, loss_functions, epochs=10):
    for opt_name, optimizer in optimizers.items():
        for loss_name, criterion in loss_functions.items():
            print(f"\nTraining with {opt_name} optimizer and {loss_name} loss function.")
            trained_model = MNISTFFNN(mnist_input_size, hidden_sizes, mnist_output_size)
            train_losses, valid_losses, train_accuracies, valid_accuracies = train_network(trained_model, criterion, optimizer, train_loader, valid_loader, epochs)
            print('Plot Learning Curve...')
            plot_loss(train_losses, valid_losses)
            print('Plot Accuracy Curve...')
            plot_accuracy(train_accuracies, valid_accuracies)
            print('Finding: F1 Score and confusion Matrix')
            conf_matrix = metrics(mnist_model, mnist_test_loader)
            plot_matrix(conf_matrix)


print('Training the Model with different optimizers and loss functions...')
train_with_different_optimizers_and_losses(mnist_model, mnist_train_loader, mnist_valid_loader, optimizers, loss_functions, epochs=5)

