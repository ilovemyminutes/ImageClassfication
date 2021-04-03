import torch
from torch import nn
from torchvision import models
from efficientnet_pytorch import EfficientNet


def load_model(model_type: str, n_class: int, load_state_dict: str, freeze: bool=False):
    if model_type == "VanillaResNet":
        model = VanillaResNet(n_class, freeze)
    elif model_type == 'VanillaEfficientNet':
        model = VanillaEfficientNet(n_class, freeze)
    else:
        raise NotImplementedError()
    if load_state_dict:
        model.load_state_dict(torch.load(load_state_dict))
        print(f'Loaded pretrained weights from {load_state_dict}')
    return model


class VanillaEfficientNet(nn.Module):
    def __init__(self, n_class: int, freeze: bool = False):
        super(VanillaEfficientNet, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b3')
        if freeze:
            self._freeze()
        self.linear = nn.Linear(in_features=1000, out_features=n_class)

    def forward(self, x):
        output = self.efficientnet(x)
        output = self.linear(output)
        return output

    def _freeze(self):
        for param in self.efficientnet.parameters():
            param.requires_grad = False


class VanillaResNet(nn.Module):
    def __init__(self, n_class: int, freeze: bool = False):
        super(VanillaResNet, self).__init__()
        self.resnet = models.resnet101(pretrained=True)
        if freeze:
            self._freeze()
        self.resnet.fc = nn.Linear(in_features=2048, out_features=n_class)

    def forward(self, x):
        output = self.resnet(x)
        return output

    def _freeze(self):
        for name, param in self.resnet.named_parameters():
            if name not in ["fc.weight", "fc.bias"]:
                param.requires_grad = False