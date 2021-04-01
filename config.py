from dataclasses import dataclass
from torch import optim, nn


@dataclass
class Config:
    Train: str = "./preprocessed/train"
    Valid: str = "./preprocessed/valid"
    Test: str = "./preprocessed/test"
    Eval: str = "./input/data/eval/images"

    BatchSize: int = 64
    LR: float = 5e-4
    Adam: str = 'adam'
    SGD: str = 'sgd'
    Epochs: int = 10
    Seed: int = 42

    BaseTransform: str = "base"
    FixTransform: str = "fix"

    VanillaResNet: str = "VanillaResNet"
    VanillaEfficientNet: str = "VanillaEfficientNet"
    ModelPath: str = "./saved_models"

    Inference: str = "./prediction"
    Info: str = "./preprocessed/info.pkl"


def Optimizer(model: nn.Module, optim_type_: str, lr: float):
    if optim_type_ == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optim_type_ == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    return optimizer


class LabelEncoder:
    Encoder= {
        'mask': {'incorrect': 0, 'wear': 1, 'not_wear': 2},
        'gender': {'male': 0, 'female': 1}
        }
    Decoder= {
        'mask': {0: 'incorrect', 1:'wear', 2:'not_wear'},
        'gender': {0:'male', 1:'female'}
        }
        
    def transform(self, label, task: str='mask'):
        output = self.Encoder[task][label]
        return output
    
    def inverse_transform(self, label, task: str='mask'):
        output = self.Decoder[task][label]
        return output
    

