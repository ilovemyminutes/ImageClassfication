import torch
from torch import nn
from torch.nn import functional as F
from config import Loss, get_class_num


def get_criterion(loss_type: str = Loss.CE, task: str = None):
    if loss_type == Loss.CE:
        criterion = nn.CrossEntropyLoss()
    elif loss_type == Loss.FL:
        criterion = FocalLoss(gamma=5)
    elif loss_type == Loss.MSE:
        criterion = nn.MSELoss()
    elif loss_type == Loss.SML1:
        criterion = nn.SmoothL1Loss
    elif loss_type == Loss.LS:
        n_class = get_class_num(task)
        criterion = LabelSmoothingLoss(classes=n_class)
    return criterion


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight  # weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.cross_entropy(
            input, target, reduction=self.reduction, weight=self.weight
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.2, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
