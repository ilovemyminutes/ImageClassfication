import os
from glob import glob
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
from augmentation import configure_transform
from utils import load_pickle, load_json, age2ageg
from config import Config, Task, Aug


MULTI2MAIN = {
    (1, 0, 0): 0,  # (mask, gender, ageg)
    (1, 0, 1): 1,
    (1, 0, 2): 2,
    (1, 1, 0): 3,
    (1, 1, 1): 4,
    (1, 1, 2): 5,
    (0, 0, 0): 6,
    (0, 0, 1): 7,
    (0, 0, 2): 8,
    (0, 1, 0): 9,
    (0, 1, 1): 10,
    (0, 1, 2): 11,
    (2, 0, 0): 12,
    (2, 0, 1): 13,
    (2, 0, 2): 14,
    (2, 1, 0): 15,
    (2, 1, 1): 16,
    (2, 1, 2): 17,
}


def get_dataloader(
    task: str = Task.Main,  # class, gender, ageg, age
    phase: str = "train",
    data_root: str = Config.Train,
    transform_type: str = Aug.BaseTransform,
    batch_size: int = Config.Batch32,
    shuffle: bool = True,
    drop_last: bool = True,
):
    transform = configure_transform(phase, transform_type)

    if phase in ["train", "valid"]:
        meta_path = (
            Config.Metadata
        )  # if os.path.isfile(Config.Metadata) else '../preprocessed/metadata.json'
        dataset = TrainDataset(
            root=data_root, transform=transform, task=task, meta_path=meta_path
        )
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
        )
    else:
        dataset = EvalDataset(root=data_root, transform=transform)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
        )

    return dataloader


class TrainDataset(Dataset):
    def __init__(
        self,
        root: str,
        transform=None,
        task: str = Task.Main,
        age_filter: int = 60,
        meta_path: str = None,
    ):
        """마스크 상태, 나이, 나이대, 클래스(0~17)의 4가지 태스크에 따른 레이블링을 지원하는 데이터셋"""
        self.img_paths = glob(os.path.join(root, "*"))
        self.metadata = load_json(meta_path)
        self.task = task
        self.transform = transform
        self.age_filter = age_filter
        self.label_encoder = LabelEncoder()

    def __getitem__(self, index):
        name = os.path.basename(self.img_paths[index])
        img = Image.open(self.img_paths[index])
        label = self.metadata[name]

        if self.task != Task.MultiLabel:
            if self.task == Task.Main:  # Main Task: 0~17 클래스에 대한 예측
                if label[Task.Age] >= self.age_filter:  # 예) 58세 이상을 60세 이상 클래스로 간주할 경우
                    mask_state = label[Task.Mask]
                    gender = label[Task.Gender]
                    ageg = 2  # old class
                    label = self.label_encoder.transform(
                        (mask_state, gender, ageg), task=Task.Main
                    )
                else:
                    label = label[self.task]

            elif self.task == Task.Ageg:
                label = (
                    2 if label[Task.Age] >= self.age_filter else label[self.task]
                )  # 예) 58세 이상을 60세 이상 클래스로 간주할 경우

            else:
                label = label[self.task]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.img_paths)


class EvalDataset(Dataset):
    def __init__(self, root, transform=None, info_path: str = Config.Info):
        if not os.path.isfile(info_path):
            info_path = "../preprocessed/info.pkl"  # for notebook env
        info = load_pickle(info_path)  # 추론 순서를 맞추기 위해
        self.img_paths = list(map(lambda x: os.path.join(root, x), info))
        self.transform = transform

    def __getitem__(self, index):
        name = os.path.basename(self.img_paths[index])
        image = Image.open(self.img_paths[index])
        if self.transform:
            image = self.transform(image)
        return name, image

    def __len__(self):
        return len(self.img_paths)


class CustomImageFolder(DatasetFolder):
    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        loader=default_loader,
        is_valid_file=None,
    ):
        args = (loader, IMG_EXTENSIONS, transform, target_transform, is_valid_file)
        super(CustomImageFolder, self).__init__(root, *args)

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort(key=lambda x: int(x))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx


class LabelEncoder:
    Encoder = {
        "mask": {"incorrect": 0, "wear": 1, "not_wear": 2},
        "gender": {"male": 0, "female": 1},
        "ageg": {"young": 0, "middle": 1, "old": 2},
        "main": MULTI2MAIN,
    }
    Decoder = {
        "mask": {0: "incorrect", 1: "wear", 2: "not_wear"},
        "gender": {0: "male", 1: "female"},
        "ageg": {0: "young", 1: "middle", 2: "old"},
    }

    def transform(self, label, task: str = "mask"):
        output = self.Encoder[task][label]
        return output

    def inverse_transform(self, label, task: str = "mask"):
        output = self.Decoder[task][label]
        return output
