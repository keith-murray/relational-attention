from tqdm import tqdm
import torch

def train_model(
    model,
    data_loader,
    optimizer=None,
    loss_fn=None,
    epochs=10,
    device=None,
    return_history=True,
    verbose=True,
):
    """
    Generic training loop with tqdm progress bar.

    Args:
        model (nn.Module): PyTorch model to train.
        data_loader (DataLoader): Iterable providing (X, y) batches.
        optimizer (torch.optim.Optimizer, optional): Optimizer instance.
            If None, defaults to Adam(model.parameters()).
        loss_fn (callable, optional): Loss function.
            If None, defaults to BCEWithLogitsLoss.
        epochs (int): Number of training epochs.
        device (torch.device, optional): Device for computation.
            If None, inferred from model or defaults to CPU.
        return_history (bool): Whether to return the list of losses per epoch.
        verbose (bool): Whether to show tqdm progress bar.

    Returns:
        list[float]: Average loss per epoch (if return_history=True).
    """
    # Defaults
    device = device or next(model.parameters()).device
    optimizer = optimizer or torch.optim.Adam(model.parameters())
    loss_fn = loss_fn or torch.nn.BCEWithLogitsLoss()

    # Put model in training mode
    model.train()

    loss_history = []

    # Optionally wrap tqdm
    progress = range(epochs)
    if verbose:
        progress = tqdm(progress, desc="Training", total=epochs, leave=True)

    for epoch in progress:
        total_loss = 0.0

        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            preds = model(X)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        loss_history.append(avg_loss)

        # Update progress bar if active
        if verbose:
            progress.set_postfix({"loss": f"{avg_loss:.4f}"})

    if verbose:
        progress.close()

    return loss_history if return_history else None


def evaluate_model(model, loader):
    """
    Evaluate a binary classification model on a held-out test DataLoader.

    Args:
        model (nn.Module):
            A trained PyTorch model that outputs a single logit per example.
        loader (DataLoader):
            A DataLoader that yields (X, y) pairs where:
                X : tensor of input examples, shape (batch_size, ..., ...)
                y : tensor of binary labels, shape (batch_size,)

    Returns:
        float:
            Accuracy in the range [0, 1], computed as
            (# correct predictions) / (# total predictions).
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in loader:
            logits = model(X)                 # (batch,)
            preds = (torch.sigmoid(logits) > 0.5)

            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total
