import torch
from torch import nn
from torch.nn import functional as F
from config import Loss


def get_criterion(loss_type: str=Loss.CE):
    if loss_type == Loss.CE:
        criterion = nn.CrossEntropyLoss()
    elif loss_type == Loss.FL:
        criterion = FocalLoss(gamma=5)
    elif loss_type == Loss.MSE:
        criterion = nn.MSELoss()
    elif loss_type == Loss.SML1:
        criterion = nn.SmoothL1Loss
    return criterion



class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss