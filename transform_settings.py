from torchvision import transforms


def configure_transform(phase: str, transform_type: int):
    if phase == "train":
        if transform_type == "base":
            transform = transforms.Compose(
                [
                    transforms.CenterCrop((384, 384)),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            raise NotImplementedError()

    else:
        if transform_type == "base":
            transform = transforms.Compose(
                [
                    transforms.CenterCrop((384, 384)),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            raise NotImplementedError()

    return transform
