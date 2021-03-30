import torch
from torch import nn
from torchvision import models
from efficientnet_pytorch import EfficientNet

def load_model(model_type: str, load_state_dict: str):
    if model_type == "VanillaResNet":
        model = VanillaResNet()
    elif model_type == 'VanillaEfficientNet':
        model = VanillaEfficientNet()
    else:
        raise NotImplementedError()
    if load_state_dict:
        model.load_state_dict(torch.load(load_state_dict))
    return model


class VanillaEfficientNet(nn.Module):
    def __init__(self, freeze: bool = True):
        super(VanillaEfficientNet, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b3')
        if freeze:
            self._freeze()
        self.linear = nn.Linear(in_features=1000, out_features=18)

    def forward(self, x):
        output = self.efficientnet(x)
        output = self.linear(output)
        return output

    def _freeze(self):
        for param in self.efficientnet.parameters():
            param.require_grad = False


class VanillaResNet(nn.Module):
    def __init__(self, freeze: bool = True):
        super(VanillaResNet, self).__init__()
        self.resnet = models.resnet101(pretrained=True)
        if freeze:
            self._freeze()
        self.resnet.fc = nn.Linear(in_features=2048, out_features=18, bias=True)
        self._initialize()

    def forward(self, x):
        output = self.resnet(x)
        return output

    def _freeze(self):
        for name, param in self.resnet.named_parameters():
            if name not in ["fc.weight", "fc.bias"]:
                param.requires_grad = False

    def _initialize(self):
        torch.nn.init.xavier_uniform_(self.resnet.fc.weight)
        self.resnet.fc.bias.data.fill_(0.01)


# class ThreeHeadsNet(nn.Module):
#     def __init__(self):
#         super(ThreeHeadsNet, self).__init__()
#         self.resnet = models.resnet50(pretrained=True)
#         self._freeze()
#         raise NotImplementedError()

#     def forward(self, x):
#         output = self.resnet(x)

#     def _freeze(self):
#         for name, param in self.resnet.named_parameters():
#             if name not in ["fc.weight", "fc.bias"]:
#                 param.requires_grad = False
