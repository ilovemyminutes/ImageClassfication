import os
from glob import glob
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
from transform_settings import configure_transform
from utils import load_pickle, load_json
from config import Config, Task


def get_dataloader(
    task: str='class', # class, gender, ageg, age
    phase: str='train',
    data_root: str=Config.Train,
    transform_type: str='base',
    batch_size: int = 64,
    shuffle: bool = True,
    drop_last: bool = True,
):
    transform = configure_transform(phase, transform_type)

    if phase in ["train", "valid", "test"]:
        meta_path = Config.Metadata if os.path.isfile(Config.Metadata) else '../preprocessed/metadata.json'
        dataset = TrainDataset(root=data_root, transform=transform, task=task, meta_path=meta_path)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
        )
    else:
        dataset = EvalDataset(data_root, transform)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
        )

    return dataloader


class TrainDataset(Dataset):
    def __init__(self, root: str, transform=None, task: str=Task.Main, meta_path:str=None):
        """마스크 상태, 나이, 나이대, 클래스(0~17)의 4가지 태스크에 따른 레이블링을 지원하는 데이터셋
        """
        self.img_paths = glob(os.path.join(root, '*'))
        self.metadata = load_json(meta_path)
        self.task = task
        self.transform = transform
    
    def __getitem__(self, index):
        name = os.path.basename(self.img_paths[index])
        img = Image.open(self.img_paths[index])
        if self.task == 'all':
            label = self.metadata[name]
        else:
            label = self.metadata[name][self.task]

        if self.transform is not None:
            img = self.transform(img)

        return img, label
        
    def __len__(self):
        return len(self.img_paths)


class CustomImageFolder(DatasetFolder):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader, is_valid_file=None):
        args = (loader, IMG_EXTENSIONS, transform, target_transform, is_valid_file)
        super(CustomImageFolder, self).__init__(root, *args)
    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort(key=lambda x: int(x))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx


class EvalDataset(Dataset):
    def __init__(self, data_root, transform=None):
        info = load_pickle(Config.Info) # 추론 순서를 맞추기 위해
        self.img_paths = list(map(lambda x: os.path.join(data_root, x), info))
        self.transform = transform

    def __getitem__(self, index):
        name = os.path.basename(self.img_paths[index])
        image = Image.open(self.img_paths[index])
        if self.transform:
            image = self.transform(image)
        return name, image

    def __len__(self):
        return len(self.img_paths)


class LabelEncoder:
    Encoder= {
        'mask': {'incorrect': 0, 'wear': 1, 'not_wear': 2},
        'gender': {'male': 0, 'female': 1},
        'ageg': {'young': 0, 'middle': 1, 'old': 2}
        }
    Decoder= {
        'mask': {0: 'incorrect', 1:'wear', 2:'not_wear'},
        'gender': {0:'male', 1:'female'},
        'ageg': {0: 'young', 1: 'middle', 2: 'old'}
        }

    def transform(self, label, task: str='mask'):
        output = self.Encoder[task][label]
        return output
    
    def inverse_transform(self, label, task: str='mask'):
        output = self.Decoder[task][label]
        return output
