from dataclasses import dataclass


@dataclass
class Config:
    Train: str = "./preprocessed/train"
    Valid: str = "./preprocessed/valid"
    Test: str = "./preprocessed/test"
    Eval: str = "./input/data/eval/images"

    BatchSize: int = 128
    LR: float = 25e-5
    Epochs: int = 30
    Seed: int = 42

    BaseTransform: str = "base"
    FixTransform: str = "fix"

    VanillaResNet: str = "VanillaResNet"
    VanillaEfficientNet: str = "VanillaEfficientNet"
    ModelPath: str = "./saved_models"

    Inference: str = "./prediction"
    Info: str = "./preprocessed/info.pkl"
    
