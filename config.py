from dataclasses import dataclass


@dataclass
class Config:
    Train: str = "./preprocessed/train"
    Eval: str = "./input/data/eval/images"
    Inference: str='./prediction'
    Info: str='./input/data/eval/info.csv'
    BatchSize: int = 128
    LR: float = 25e-5
    Epochs: int = 30
    BaseTransform: str = "base"
    VanillaResNet: str = "vanillaresnet"
    ModelPath: str = "./saved_models"
    Seed: int = 42
