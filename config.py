from dataclasses import dataclass


@dataclass
class Config:
    Train: str = "./preprocessed/train"
    Valid: str = "./preprocessed/test"
    TrainS: str = "./preprocessed_stratified/train"
    ValidS: str = "./preprocessed_stratified/test"

    Eval: str = "./input/data/eval/images"

    Batch64: int = 64
    Batch32: int = 32
    LRBase: float = 5e-3
    LRFaster: float = 1e-2
    LRSlower: float = 1e-3
    CosineScheduler: str = 'cosine'
    Adam: str = 'adam'
    AdamP: str = 'adamp'
    SGD: str = 'sgd'
    Momentum: str = 'momentum'
    Epochs: int = 50
    Seed: int = 42

    VanillaResNet: str = "VanillaResNet"
    VanillaEfficientNet: str = "VanillaEfficientNet"
    MultiClassTHANet: str = 'MultiClassTHANet'
    MultiClassTHANet_MK1: str = 'MultiClassTHANet_MK1'
    MultiLabelTHANet: str = 'MultiLabelTHANet'
    THANet_MK1: str = 'THANet_MK1' # wrong structure yet
    THANet_MK2: str = 'THANet_MK2' # wrong structure yet
    ModelPath: str = "./saved_models"

    Inference: str = "./prediction"
    Metadata: str = './preprocessed_stratified/metadata.json'
    Info: str = './preprocessed/info.pkl'

@dataclass
class Aug:
    BaseTransform: str = "base"
    FaceCrop: str = 'facecrop'
    FixTransform: str = "fix"
    Random: str = 'random'

@dataclass
class Task:
    Mask: str='mask'
    Gender: str='gender'
    Age: str='age'
    AgeC: str='age_clf'
    Ageg: str='ageg'
    Main: str='main'
    All: str='all'

@dataclass
class Loss:
    CE: str='crossentropyloss'
    FL: str='focalloss'
    MSE: str='mseloss'
    SML1: str='smoothl1loss'
    LS: str='labelsmoothingLoss'
    

def get_class_num(task):
    num_class_meta = {'mask': 3, 'gender': 2, 'age':1, 'age_clf': 61, 'ageg':3, 'main': 18}
    return num_class_meta[task]