import inspect

from .lstm import LSTM
from .neural_network import NeuralNetwork
from .optimizer_utils import get_optimizer, get_scheduler
from .train_utils import train_one_epoch, validate_one_epoch


def get_model(model, params=None):
    if model:
        if params is None:
            raise ValueError("Parameters for 'NeuralNetwork' must be provided in params.")

        expected_params = inspect.signature(model.__init__).parameters.keys()
        # Remove 'self' which is always the first parameter
        expected_params = [p for p in expected_params if p != 'self']
        
        # Filter the params dictionary to only include expected parameters
        filtered_params = {k: params[k] for k in expected_params if k in params}
        
        return model(**filtered_params)
    else:
        raise ValueError("Unsupported model type. Use the correct model.")


def train_model(model, train_loader, validation_loader, epochs, optimizer_name, criterion, learning_rate, device, verbose=True, is_rnn=False, is_gnn=False, is_transformer=False, accumulation_steps=8):
    optimizer = get_optimizer(optimizer_name, model.parameters(), learning_rate)
    scheduler = None
    if optimizer_name == "sgd":
        scheduler = get_scheduler(optimizer)

    history = {"training_accuracy": [], "validation_accuracy": []}
    width = len(str(epochs))
    
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


def test_model(model, validation_loader, criterion, device, verbose=True, is_rnn=False, is_gnn=False, is_transformer=False):
    history = {"validation_accuracy": []}
    val_loss, val_acc, val_metrics = validate_one_epoch(model, validation_loader, criterion, device, is_rnn=is_rnn, is_gnn=is_gnn, is_transformer=is_transformer)
    history["validation_accuracy"].append(val_acc)

    if verbose:
        print(f"Val accuracy: {val_acc * 100:.2f}% | "
              f"Val loss: {val_loss:.4f} | "
              f"Precision: {val_metrics['precision'] * 100:.2f}% | Recall: {val_metrics['recall'] * 100:.2f}% | F1: {val_metrics['f1'] * 100:.2f}%")

    return history, val_metrics["val_labels"], val_metrics["val_preds"]
