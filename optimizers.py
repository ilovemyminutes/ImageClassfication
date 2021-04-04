from torch import nn, optim

def get_optim(model: nn.Module, optim_type_: str, lr: float):
    if optim_type_ == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optim_type_ == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    return optimizer