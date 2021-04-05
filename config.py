from dataclasses import dataclass


@dataclass
class Config:
    Train: str = "./preprocessed/train"
    Valid: str = "./preprocessed/valid"
    Test: str = "./preprocessed/test"
    TrainS: str = "./preprocessed_stratified/train"
    ValidS: str = "./preprocessed_stratified/valid"
    TestS: str = "./preprocessed_stratified/test"

    Eval: str = "./input/data/eval/images"

    BatchSize: int = 64
    LR: float = 5e-3
    Adam: str = 'adam'
    SGD: str = 'sgd'
    Momentum: str = 'momentum'
    Epochs: int = 15
    Seed: int = 42

    VanillaResNet: str = "VanillaResNet"
    VanillaEfficientNet: str = "VanillaEfficientNet"
    THANet_MK1: str = 'THANet_MK1' # wrong structure yet
    THANet_MK2: str = 'THANet_MK2' # wrong structure yet
    ModelPath: str = "./saved_models"

    Inference: str = "./prediction"
    Metadata: str = './preprocessed/metadata.json'
    Info: str = './preprocessed/info.pkl'

@dataclass
class Aug:
    BaseTransform: str = "base"
    FaceCrop: str = 'facecrop'
    FixTransform: str = "fix"

class Task:
    Mask: str='mask'
    Gender: str='gender'
    Age: str='age'
    AgeC: str='age_clf'
    Ageg: str='ageg'
    Main: str='main'
    All: str='all'

class Loss:
    CE: str='crossentropyloss'
    FL: str='focalloss'
    MSE: str='mseloss'
    SML1: str='smoothl1loss'


def get_class_num(task):
    num_class_meta = {'mask': 3, 'gender': 2, 'age':1, 'age_clf': 61, 'ageg':3, 'main': 18}
    return num_class_meta[task]