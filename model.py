from torch import nn
from torchvision import models


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self._freeze()

    def forward(self, x):
        output = self.resnet(x)

    def _freeze(self):
        for name, param in self.resnet.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False
                