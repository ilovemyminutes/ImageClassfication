import os
import math
import torch
from torch import nn
from torchvision import models
from efficientnet_pytorch import EfficientNet
from torch.nn import functional as F
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
    elif model_type == 'MultiClassTHANet_MK1':
        model = MultiClassTHANet_MK1()
    else:
        raise NotImplementedError()
    if load_state_dict:
        model.load_state_dict(torch.load(load_state_dict))
        print(f'Loaded pretrained weights from {load_state_dict}')
    return model


class MultiClassTHANet_MK1(nn.Module):
    TASK = {'mask': 3, 'ageg': 3, 'gender': 2, 'all': 8, 'main': 18} # all: 8 = 3 + 3 + 2
    def __init__(self, d_model: int=64, num_heads: int=8):
        super(MultiClassTHANet_MK1, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b3') # 마지막 1000
        self.bn = nn.BatchNorm1d(1000)
        self.dropout = nn.Dropout()
        self.linear_mask = nn.Linear(1000, d_model) # mask
        self.linear_ageg = nn.Linear(1000, d_model) # ageg
        self.linear_gender = nn.Linear(1000, d_model) # gender
        
        self.w_query = nn.Linear(in_features=d_model, out_features=d_model)
        self.w_key = nn.Linear(in_features=d_model, out_features=d_model)
        self.w_value = nn.Linear(in_features=d_model, out_features=d_model)

        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.attn_layer_norm = nn.LayerNorm(normalized_shape=d_model)

        self.output_mask = nn.Linear(d_model, self.TASK['mask'])
        self.output_ageg = nn.Linear(d_model, self.TASK['ageg'])
        self.output_gender = nn.Linear(d_model, self.TASK['gender'])
        self.output_relu = nn.ReLU()

        self.output_main = nn.Linear(self.TASK['all'], self.TASK['main'])

    def forward(self, x):
        x = self.backbone(x)
        x = self.bn(x)
        x = self.dropout(x)

        x_mask = self.linear_mask(x)
        x_ageg = self.linear_ageg(x)
        x_gender = self.linear_gender(x)
        x_all = torch.cat([x_mask.unsqueeze(1), x_ageg.unsqueeze(1), x_gender.unsqueeze(1)], dim=1) # concat hidden states of mask, ageg, gender

        qkv = self._get_qkv(x_all)
        attn_scores = self.attention(*qkv)
        attn_scores += x_all

        x_all_splited = self._split_task(attn_scores)
        task_outputs = self._get_task_output(*x_all_splited)
        x_main = self.output_main(task_outputs)

        return x_main
        
    def _get_qkv(self, x):
        q = self.w_query(x)
        k = self.w_key(x)
        v = self.w_value(x)
        return (q, k, v)

    def _split_task(self, x):
        x_mask = x.transpose(0, 1)[0]
        x_ageg = x.transpose(0, 1)[1]
        x_gender = x.transpose(0, 1)[2]
        return x_mask, x_ageg, x_gender

    def _get_task_output(self, x_mask, x_ageg, x_gender):
        output_mask = self.output_mask(x_mask)
        output_ageg = self.output_ageg(x_ageg)
        output_gender = self.output_gender(x_gender)
        output = torch.cat([output_mask, output_ageg, output_gender], dim=1)
        output = self.output_relu(output)
        return output


class MultiLabelTHANet(nn.Module):
    TASK = {'mask': 3, 'ageg': 3, 'gender': 2, 'all': 8, 'main': 18} # all: 8 = 3 + 3 + 2
    def __init__(self, d_model: int=64, num_heads: int=8):
        super(MultiLabelTHANet, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0') # 마지막 1000
        self.bn = nn.BatchNorm1d(1000)
        self.dropout = nn.Dropout()
        self.linear_mask = nn.Linear(1000, d_model) # mask
        self.linear_ageg = nn.Linear(1000, d_model) # ageg
        self.linear_gender = nn.Linear(1000, d_model) # gender
        
        self.w_query = nn.Linear(in_features=d_model, out_features=d_model)
        self.w_key = nn.Linear(in_features=d_model, out_features=d_model)
        self.w_value = nn.Linear(in_features=d_model, out_features=d_model)

        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.attn_layer_norm = nn.LayerNorm(normalized_shape=d_model)

        self.output_mask = nn.Linear(d_model, self.TASK['mask'])
        self.output_ageg = nn.Linear(d_model, self.TASK['ageg'])
        self.output_gender = nn.Linear(d_model, self.TASK['gender'])
        self.output_main = nn.Linear(self.TASK['main'], self.TASK['main'])

    def forward(self, x):
        x = self.backbone(x)
        x = self.bn(x)
        x = self.dropout(x)

        x_mask = self.linear_mask(x)
        x_ageg = self.linear_ageg(x)
        x_gender = self.linear_gender(x)
        x_all = torch.cat([x_mask.unsqueeze(1), x_ageg.unsqueeze(1), x_gender.unsqueeze(1)], dim=1) # concat hidden states of mask, ageg, gender

        qkv = self._get_qkv(x_all)
        attn_scores = self.attention(*qkv)
        attn_scores += x_all

        x_all_splited = self._split_task(attn_scores)
        output_mask, output_ageg, output_gender = self._get_task_output(*x_all_splited)
        return output_mask, output_ageg, output_gender
        
    def _get_qkv(self, x):
        q = self.w_query(x)
        k = self.w_key(x)
        v = self.w_value(x)
        return (q, k, v)

    def _split_task(self, x):
        x_mask = x.transpose(0, 1)[0]
        x_ageg = x.transpose(0, 1)[1]
        x_gender = x.transpose(0, 1)[2]
        return x_mask, x_ageg, x_gender

    def _get_task_output(self, x_mask, x_ageg, x_gender):
        x_mask = self.output_mask(x_mask)
        x_ageg = self.output_ageg(x_ageg)
        x_gender = self.output_gender(x_gender)
        return x_mask, x_ageg, x_gender


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


class VanillaEfficientNet(nn.Module):
    def __init__(self, n_class: int, freeze: bool = False):
        super(VanillaEfficientNet, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b3')
        if freeze:
            self._freeze()
        self.batchnorm = nn.BatchNorm1d(num_features=1000)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        self.linear = nn.Linear(in_features=1000, out_features=n_class)

    def forward(self, x):
        x = self.efficientnet(x)
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = self.relu(x)
        output = self.linear(x)
        return output

    def _freeze(self):
        for param in self.efficientnet.parameters():
            param.requires_grad = False


class VanillaResNet(nn.Module):
    def __init__(self, n_class: int, freeze: bool = False):
        super(VanillaResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        if freeze:
            self._freeze()
        self.batchnorm = nn.BatchNorm1d(num_features=1000)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        self.linear = nn.Linear(in_features=1000, out_features=n_class)

    def forward(self, x):
        x = self.resnet(x)
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = self.relu(x)
        output = self.linear(x)
        return output

    def _freeze(self):
        for name, param in self.resnet.named_parameters():
            if name not in ["fc.weight", "fc.bias"]:
                param.requires_grad = False


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        return

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """Query, Key, Value 텐서를 입력 받아 Attention 텐서를 리턴. mask 인스턴스가 True일 경우 Masked Multi-headed Attention을 수행
        Args:
            query (torch.Tensor): Query 텐서. (batch_size, max_len, hidden_dim)
            key (torch.Tensor): Key 텐서. (batch_size, max_len, hidden_dim)
            value (torch.Tensor): Value 텐서. (batch_size, max_len, hidden_dim)
        Returns:
            torch.Tensor: Attention 텐서. (batch_size, max_len, hidden_dim)
        """
        self.batch_size = query.size(0)
        query = query.view(self.batch_size, -1, self.num_heads, self.d_k)
        key = key.view(self.batch_size, -1, self.num_heads, self.d_k)
        value = value.view(self.batch_size, -1, self.num_heads, self.d_k)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attention = self.get_self_attention(query, key, value)

        return attention

    def get_self_attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """attention 텐서를 리턴. multi-head attention 수행 후, aggregate된 최종 attention 텐서를 리턴
        Args:
            query (torch.Tensor): Query 텐서. (batch_size, num_heads, max_len, d_k)
            key (torch.Tensor): Key 텐서. (batch_size, num_heads, max_len, d_k)
            value (torch.Tensor): Value 텐서. (batch_size, num_heads, max_len, d_k)
        Returns:
            torch.Tensor: Attention 텐서. (batch_size, max_len, d_model)
        """
        attention_raw = F.softmax(
            torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.d_model), dim=-1
        )
        attention = torch.matmul(attention_raw, value).view(
            self.batch_size, -1, self.d_model
        )
        return attention