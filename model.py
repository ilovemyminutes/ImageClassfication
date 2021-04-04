import os
import torch
from torch import nn
from torchvision import models
from efficientnet_pytorch import EfficientNet
from config import Config, get_class_num


def load_model(model_type: str, task: str, load_state_dict: str, freeze: bool=False):
    n_class = get_class_num(task)
    if model_type == "VanillaResNet":
        model = VanillaResNet(n_class, freeze)
    elif model_type == 'VanillaEfficientNet':
        model = VanillaEfficientNet(n_class, freeze)
    elif model_type == 'THANet_MK1':
        model = THANet_MK1(freeze)
    elif model_type == 'THANet_MK2':
        model = THANet_MK2(freeze)
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


class THANet_MK1(nn.Module): # Three-headed attention EfficientNEt
    NUM_CLASS = {'mask': 3, 'ageg': 3, 'gender': 2, 'main': 18}
    def __init__(self, freeze: bool=False):
        super(THANet_MK1, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b3')
        if freeze:
            self._freeze()

        self.linear_mask = nn.Linear(1000, self.NUM_CLASS['mask'])
        self.linear_ageg = nn.Linear(1000, self.NUM_CLASS['ageg'])
        self.linear_gender = nn.Linear(1000, self.NUM_CLASS['gender'])
        self.linear_main = nn.Linear(1000, self.NUM_CLASS['main'])

    def forward(self, x):
        x = self.backbone(x)
        output_mask = self.linear_mask(x)
        output_ageg = self.linear_ageg(x)
        output_gender = self.linear_gender(x)
        output_main = self.linear_main(x)
        return output_mask, output_ageg, output_gender, output_main

    def _freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False


class THANet_MK2(nn.Module): # Three-headed attention EfficientNEt
    NUM_CLASS = {'mask': 3, 'ageg': 3, 'gender': 2, 'main': 18}
    PRETRAINED = {
        'gender': os.path.join(Config.ModelPath, 'VanillaEfficientNet_taskgender_epoch02_lr0.005_transformbase_optimadam_loss0.0001_eval0.9658_seed42.pth'),
        'mask': os.path.join(Config.ModelPath, 'VanillaEfficientNet_taskmask_epoch04_lr0.005_transformbase_optimadam_loss0.0000_eval0.9909_seed42.pth'),
        'ageg': os.path.join(Config.ModelPath, 'VanillaEfficientNet_taskageg_epoch04_lr0.005_transformbase_optimadam_loss0.0002_eval0.9118_seed42.pth')
    }

    def __init__(self, freeze: bool=False):
        super(THANet_MK2, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b3')
        if freeze:
            self._freeze()

        linear_gender = VanillaEfficientNet(n_class=self.NUM_CLASS['gender'])
        linear_gender.load_state_dict(torch.load(self.PRETRAINED['gender']))
        self.linear_gender = linear_gender.linear

        linear_mask = VanillaEfficientNet(n_class=self.NUM_CLASS['mask'])
        linear_mask.load_state_dict(torch.load(self.PRETRAINED['mask']))
        self.linear_mask = linear_mask.linear

        linear_ageg = VanillaEfficientNet(n_class=self.NUM_CLASS['ageg'])
        linear_ageg.load_state_dict(torch.load(self.PRETRAINED['ageg']))
        self.linear_ageg = linear_ageg.linear
        self.linear_main = nn.Linear(1000, self.NUM_CLASS['main'])

    def forward(self, x):
        x = self.backbone(x)
        output_mask = self.linear_mask(x)
        output_ageg = self.linear_ageg(x)
        output_gender = self.linear_gender(x)
        output_main = self.linear_main(x)
        return output_mask, output_ageg, output_gender, output_main

    def _freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False


