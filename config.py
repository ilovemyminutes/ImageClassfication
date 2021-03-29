from dataclasses import dataclass


@dataclass
class Config:
    Train: str='./preprocessed/train'
    Eval: str='./input/eval'
    BatchSize: int=128
    LR: float=5e-4
    BaseTransform: str='base'

