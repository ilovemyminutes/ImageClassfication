from dataclasses import dataclass


@dataclass
class Config:
    Train: str='./input/data'
    Eval: str='./input/eval'
