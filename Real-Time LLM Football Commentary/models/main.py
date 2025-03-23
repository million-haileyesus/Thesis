from .lstm import LSTM
from .neural_network import NeuralNetwork
from .optimizer_utils import get_optimizer, get_scheduler
from .train_utils import train_one_epoch, validate_one_epoch


def get_model(model, params=None):
    if model:
        if params is None:
            raise ValueError("Parameters for 'NeuralNetwork' must be provided in params.")
        return model(**params)
    else:
        raise ValueError("Unsupported model type. Use the correct model.")


def train_model(model, train_loader, validation_loader, epochs, optimizer_name, criterion, learning_rate, device, verbose=True, is_rnn=False, is_gnn=False, is_transformer=False):
    optimizer = get_optimizer(optimizer_name, model.parameters(), learning_rate)
    scheduler = None
    if optimizer_name == "sgd":
        scheduler = get_scheduler(optimizer)

    history = {"training_accuracy": [], "validation_accuracy": []}
    width = len(str(epochs - 1))
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, is_rnn=is_rnn, is_gnn=is_gnn, is_transformer=is_transformer)
        val_loss, val_acc, val_metrics = validate_one_epoch(model, validation_loader, criterion, device, is_rnn=is_rnn, is_gnn=is_gnn, is_transformer=is_transformer)

        history["training_accuracy"].append(train_acc)
        history["validation_accuracy"].append(val_acc)

        l_rate = optimizer.param_groups[0]["lr"]
        if scheduler:
            scheduler.step(val_loss)

        if verbose:
            print(f"Epoch {epoch:{width}d}/{epochs}: "
                  f"Train accuracy: {train_acc * 100:.2f}% | "
                  f"Val accuracy: {val_acc * 100:.2f}% | "
                  f"Train loss: {train_loss:.4f} | "
                  f"Val loss: {val_loss:.4f} | "
                  f"learning rate: {l_rate:.6f} | "
                  f"Precision: {val_metrics['precision'] * 100:.2f}% | Recall: {val_metrics['recall'] * 100:.2f}% | F1: {val_metrics['f1'] * 100:.2f}%")

    return history, train_metrics["train_labels"], train_metrics["train_preds"], val_metrics["val_labels"], val_metrics["val_preds"]
