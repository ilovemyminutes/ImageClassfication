import torch
from torch import nn, optim
from model import VanillaResNet
from config import Config
from dataset import TrainLoader



def train(model: str=Config.VanillaResNet, epochs: int=Config.Epochs, batch_size: int=Config.BatchSize, lr: float=Config.LR, load_state_dict: str=None):