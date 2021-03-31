import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
from transform_settings import configure_transform
from utils import load_pickle
from config import Config


def get_dataloader(
    phase: str,
    data_root: str,
    transform_type: str,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = True,
):
    transform = configure_transform(phase, transform_type)

    if phase in ["train", "valid", "test"]:
        dataset = CustomImageFolder(root=data_root, transform=transform)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
        )
    else:
        dataset = EvalDataset(data_root, transform)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
        )

    return dataloader


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
