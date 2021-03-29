from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from transform_settings import configure_transform


def get_dataloader(
    phase: str,
    data_root: str,
    transform_type: str,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = True,
):
    transform = configure_transform(phase, transform_type)
    dataset = ImageFolder(data_root, transform)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )
    return dataloader
