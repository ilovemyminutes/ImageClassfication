from torch import nn, optim
from adamp import AdamP
from config import Config


def get_optim(model: nn.Module, optim_type: str, lr: float):
    if optim_type == Config.Adam:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optim_type == Config.SGD:
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optim_type == Config.Momentum:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optim_type == Config.AdamP:
        optimizer = AdamP(
            model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-2
        )
    return optimizer


def get_scheduler(scheduler_type: str, optimizer):
    if scheduler_type == Config.CosineScheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
    else:
        raise NotImplementedError()

    return scheduler
