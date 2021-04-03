from dataclasses import dataclass
from torch import optim, nn



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
    Epochs: int = 5
    Seed: int = 42

    BaseTransform: str = "base"
    FixTransform: str = "fix"

    VanillaResNet: str = "VanillaResNet"
    VanillaEfficientNet: str = "VanillaEfficientNet"
    ModelPath: str = "./saved_models"

    Inference: str = "./prediction"
    Metadata: str = './preprocessed/metadata.json'


class Task:
    Mask: str='mask'
    Gender: str='gender'
    Age: str='age'
    Ageg: str='ageg'
    Main: str='main'


def get_class_num(task):
    num_class_meta = {'mask': 3, 'gender': 2, 'age':1, 'ageg':3, 'main': 18}
    return num_class_meta[task]


def get_optim(model: nn.Module, optim_type_: str, lr: float):
    if optim_type_ == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optim_type_ == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    return optimizer
    

