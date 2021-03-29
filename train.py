import os
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.nn import functional as F
from model import VanillaResNet
from config import Config
from dataset import TrainLoader


def train(model: str=Config.VanillaResNet, data_root: str=Config.Train, transform_type: str=Config.BaseTransform, epochs: int=Config.Epochs, batch_size: int=Config.BatchSize, lr: float=Config.LR, load_state_dict: str=None, save_path: str=Config.ModelPath):
    dataloader = TrainLoader(data_root, transform_type, batch_size)
    
    optimizer = optim.Adam(lr=lr)
    criterion = nn.CrossEntropyLoss()

    model = VanillaResNet()
    if load_state_dict:
        model.load_state_dict(torch.load(load_state_dict))
    model.cuda()
    model.train()

    for epoch in range(epochs):
        for idx, (imgs, labels) in tqdm(enumerate(dataloader), desc='Train with Mini Batch'):
            optimizer.zero_grad()

            imgs, labels = imgs.cuda(), labels.cuda()

            pred = model(imgs)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

            if idx != 0 and idx % 50 == 0:
                

                


