from dataclasses import dataclass
from torch import optim, nn

N_CLASS = {'mask': 3, 'gender': 2, 'age':2, 'ageg':3, 'class': 18}

@dataclass
class Config:
    Train: str = "./preprocessed/train"
    Valid: str = "./preprocessed/valid"
    Test: str = "./preprocessed/test"
    Eval: str = "./input/data/eval/images"

    BatchSize: int = 64
    LR: float = 5e-3
    Adam: str = 'adam'
    SGD: str = 'sgd'
    Epochs: int = 3
    Seed: int = 42

    BaseTransform: str = "base"
    FixTransform: str = "fix"

    VanillaResNet: str = "VanillaResNet"
    VanillaEfficientNet: str = "VanillaEfficientNet"
    ModelPath: str = "./saved_models"

    Inference: str = "./prediction"
    Metadata: str = './preprocessed/metadata.json'


@dataclass
class Task:
    Mask: str='mask'
    Gender: str='gender'
    Age: str='age'
    Ageg: str='ageg'
    Main: str='class'


def Optimizer(model: nn.Module, optim_type_: str, lr: float):
    if optim_type_ == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optim_type_ == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    return optimizer
    

