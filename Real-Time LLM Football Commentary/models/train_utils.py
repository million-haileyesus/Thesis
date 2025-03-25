import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score


def train_one_epoch(model, train_loader, optimizer, criterion, device, is_rnn, is_gnn, is_transformer):
    model.train()
    train_loss = 0.0
    train_acc = 0
    
    train_preds = []
    train_labels = []

    train_dataset_len = len(train_loader.dataset)

    if is_gnn:
        # For GNN models, the DataLoader yields a PyG Data object.
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)  # forward pass
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            # Compute accuracy: assume data.y is of shape [num_graphs] or [num_graphs, 1]
            _, pred = torch.max(output, 1)
            train_acc += (pred == data.y.squeeze()).float().sum().item()

            train_preds.extend(pred.cpu().numpy().flatten())
            train_labels.extend(data.y.cpu().numpy().flatten())
        
        epoch_loss = train_loss / len(train_loader)
        epoch_accuracy = train_acc / train_dataset_len

    else:
        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
    
            # Special handling for transformer models
            if is_transformer:
                # Check if it's an encoder-only or encoder-decoder transformer
                is_encoder_only = hasattr(model, 'encoder_only') and model.encoder_only
                
                if is_encoder_only:
                    # Encoder-only case - simpler forward pass
                    output = model(data)
                else:
                    # Encoder-decoder transformer with teacher forcing
                    num_classes = model.decoder.num_classes
                    label_onehot = F.one_hot(label, num_classes=num_classes).float()
                    output = model(data, label_onehot, teacher_forcing_ratio=0.5)
                
                # Reshape for loss calculation
                output_flat = output.reshape(-1, output.size(2))
                label_flat = label.reshape(-1)
                loss = criterion(output_flat, label_flat)
                
                # Get predictions from the output
                _, pred = torch.max(output, 2)
            else:
                output = model(data)
                if is_rnn:
                    output = output.permute(0, 2, 1)
                
                loss = criterion(output, label)
                _, pred = torch.max(output, 1)
    
            loss.backward()
            optimizer.step()
    
            train_loss += loss.item()
            train_acc += (pred == label).float().sum().item()

            train_preds.extend(pred.cpu().numpy().flatten())
            train_labels.extend(label.cpu().numpy().flatten())
            
        metrics = {
            "train_labels": np.array(train_preds),
            "train_preds": np.array(train_labels)
        }
    
        epoch_loss = train_loss / len(train_loader)
        if is_rnn or is_transformer:
            sequence_length = data.size(1)
            epoch_accuracy = train_acc / (train_dataset_len * sequence_length)
        else:
            epoch_accuracy = train_acc / train_dataset_len
    
    return epoch_loss, epoch_accuracy, metrics


def validate_one_epoch(model, validation_loader, criterion, device, is_rnn, is_gnn, is_transformer):
    model.eval()
    val_loss = 0.0
    val_acc = 0

    val_preds = []
    val_labels = []

    val_dataset_len = len(validation_loader.dataset)

    with torch.no_grad():
        if is_gnn:
            for data in validation_loader:
                data = data.to(device)
                output = model(data)
                loss = criterion(output, data.y)
                val_loss += loss.item()
                
                _, pred = torch.max(output, 1)
                val_acc += (pred == data.y.squeeze()).float().sum().item()
                
                val_preds.extend(pred.cpu().numpy().flatten())
                val_labels.extend(data.y.cpu().numpy().flatten())
        else:
            for data, label in validation_loader:
                data, label = data.to(device), label.to(device)
                
                # Special handling for transformer models
                if is_transformer:
                    # During validation, don't use teacher forcing
                    output = model(data, teacher_forcing_ratio=0.0)
                    
                    # Reshape for loss calculation
                    output_flat = output.reshape(-1, output.size(2))
                    label_flat = label.reshape(-1)
                    loss = criterion(output_flat, label_flat)
                    
                    # Get predictions
                    _, pred = torch.max(output, 2)
                else:
                    output = model(data)
                    if is_rnn:
                        output = output.permute(0, 2, 1)
                
                    loss = criterion(output, label)
                    _, pred = torch.max(output, 1)
                
                val_loss += loss.item()
                val_acc += (pred == label).float().sum().item()
    
                val_preds.extend(pred.cpu().numpy().flatten())
                val_labels.extend(label.cpu().numpy().flatten())
    
    epoch_loss = val_loss / len(validation_loader) 
    if is_rnn or is_transformer:
        sequence_length = data.size(1)
        epoch_accuracy = val_acc / (val_dataset_len * sequence_length)
    else:
        epoch_accuracy = val_acc / val_dataset_len

    metrics = {
        "precision": precision_score(val_labels, val_preds, average="weighted", zero_division=0),
        "recall": recall_score(val_labels, val_preds, average="weighted", zero_division=0),
        "f1": f1_score(val_labels, val_preds, average="weighted", zero_division=0),
        "val_labels": np.array(val_labels),
        "val_preds": np.array(val_preds)
    }

    return epoch_loss, epoch_accuracy, metrics
