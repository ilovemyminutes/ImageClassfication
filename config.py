from dataclasses import dataclass


@dataclass
class Config:
    Train: str = "./preprocessed/train"
    Valid: str = "./preprocessed/valid"
    Test: str = "./preprocessed/test"
    Eval: str = "./input/data/eval/images"

    Inference: str = "./prediction"
    Info: str = "./input/data/eval/info.csv"

    BatchSize: int = 128
    LR: float = 5e-4
    Epochs: int = 5
    BaseTransform: str = "base"
    VanillaResNet: str = "VanillaResNet"
    ModelPath: str = "./saved_models"
    Seed: int = 42
