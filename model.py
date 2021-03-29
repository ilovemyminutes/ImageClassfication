import torch
from torch import nn
from torchvision import models


class VanillaResNet(nn.Module):
    def __init__(self, freeze: bool=True):
        super(VanillaResNet, self).__init__()
        self.resnet = models.resnet101(pretrained=True)
        if freeze:
            self._freeze()
        self.resnet.fc = nn.Linear(in_features=2048, out_features=18, bias=True)

    def forward(self, x):
        output = self.resnet(x)
        return output

    def _freeze(self):
        for name, param in self.resnet.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False

    def _init(self):
        torch.nn.init.xavier_uniform(self.resnet.fc.weight)


class ThreeHeadsNet(nn.Module):
    def __init__(self):
        super(ThreeHeadsNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self._freeze()

    def forward(self, x):
        output = self.resnet(x)

    def _freeze(self):
        for name, param in self.resnet.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False
                