import torch.optim as optim


def get_optimizer(optimizer_name, model_parameters, learning_rate):
    if optimizer_name.lower() == "adam":
        return optim.Adam(model_parameters, lr=learning_rate)
    elif optimizer_name.lower() == "sgd":
        return optim.SGD(model_parameters, lr=learning_rate, momentum=0.9)
    else:
        raise ValueError("Unsupported optimizer type. Use 'adam' or 'sgd'.")


def get_scheduler(optimizer, scheduler_name="ReduceLROnPlateau"):
    if scheduler_name == "ReduceLROnPlateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    else:
        return None
