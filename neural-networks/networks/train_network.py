def train_network(dataloader, model, loss_func, optimizer, device="cpu"):
    model.train()
    num_batches = 0
    train_loss = 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        # make some predictions and get the error
        pred = model(X)
        loss = loss_func(pred, y)

        # backpropogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        num_batches = num_batches + 1
    train_loss /= num_batches
    print(f"Avg Train Loss: {train_loss:.6f}")
    return train_loss
