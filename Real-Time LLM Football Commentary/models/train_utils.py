import torch
from sklearn.metrics import precision_score, recall_score, f1_score


def train_one_epoch(model, train_loader, optimizer, criterion, device, is_seq_model):
    model.train()
    train_loss = 0.0
    train_acc = 0

    train_dataset_len = len(train_loader.dataset)

    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(data)
        if is_seq_model:
            outputs = outputs.permute(0, 2, 1)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, pred = torch.max(outputs, 1)
        train_acc += (pred == labels).float().sum().item()

    epoch_loss = train_loss / len(train_loader)
    if is_seq_model:
        sequence_length = data.size(1)
        epoch_accuracy = train_acc / (train_dataset_len * sequence_length)
    else:
        epoch_accuracy = train_acc / len(train_loader.dataset)

    return epoch_loss, epoch_accuracy


def validate_one_epoch(model, validation_loader, criterion, device, is_seq_model):
    model.eval()
    val_loss = 0.0
    val_acc = 0

    val_preds = []
    val_labels = []

    val_dataset_len = len(validation_loader.dataset)

    with torch.no_grad():
        for data, label in validation_loader:
            data, label = data.to(device), label.to(device)
            outputs = model(data)
            if is_seq_model:
                outputs = outputs.permute(0, 2, 1)

            loss = criterion(outputs, label)
            val_loss += loss.item()

            _, pred = torch.max(outputs, 1)
            val_acc += (pred == label).float().sum().item()

            val_preds.extend(pred.cpu().numpy().flatten())
            val_labels.extend(label.cpu().numpy().flatten())

    epoch_loss = val_loss / len(validation_loader)
    if is_seq_model:
        sequence_length = data.size(1)
        epoch_accuracy = val_acc / (val_dataset_len * sequence_length)
    else:
        epoch_accuracy = val_acc / len(validation_loader.dataset)

    metrics = {
        "precision": precision_score(val_labels, val_preds, average="weighted", zero_division=0),
        "recall": recall_score(val_labels, val_preds, average="weighted", zero_division=0),
        "f1": f1_score(val_labels, val_preds, average="weighted", zero_division=0)
    }

    return epoch_loss, epoch_accuracy, metrics
