import torch


def train_network(dataloader, model, loss_func, device="cpu", learning_rate=0.001):
    model.train()

    num_batches = 0
    train_loss = 0
    correct_predictions = 0
    total_samples = 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        # Forward pass
        pred = model(X)
        loss = loss_func(pred, y)

        # Calculate gradients manually
        model.zero_grad()
        loss.backward()

        # Manual update of weights
        with torch.no_grad():
            for param in model.parameters():
                param.data -= learning_rate * param.grad

        # Calculate accuracy
        _, predicted_labels = torch.max(pred, 1)
        correct_predictions += (predicted_labels == y).sum().item()
        total_samples += y.size(0)

        train_loss += loss.item()
        num_batches += 1

    train_loss /= num_batches
    train_accuracy = correct_predictions / total_samples
    return train_loss, train_accuracy
