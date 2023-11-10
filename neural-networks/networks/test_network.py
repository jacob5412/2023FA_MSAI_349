import torch


def test_network(dataloader, model, loss_func, device="cpu"):
    num_batches = 0
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_func(pred, y).item()
            num_batches = num_batches + 1

    test_loss /= num_batches
    print(f"Test Loss: {test_loss:>8f}\n")
    return test_loss
