from torchvision import transforms
from PIL import Image
from torchvision.transforms.transforms import ToTensor


def configure_transform(phase: str, transform_type: str):
    if phase == "train":
        if transform_type == "base":
            transform = transforms.Compose(
                [
                    transforms.Resize((512, 384), Image.BILINEAR),
                    transforms.CenterCrop((384, 384)),
                    transforms.RandomResizedCrop((224, 224)),
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
                    transforms.Resize((512, 384), Image.BILINEAR),
                    transforms.CenterCrop((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            raise NotImplementedError()
            
    return transform
