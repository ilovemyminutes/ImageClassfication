import argparse
import os
import math
from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
# import fire

from model import load_model
from config import Config, get_optim, Task, get_class_num
from dataset import get_dataloader
from utils import set_seed, verbose


def train(
    task: str=Task.Age,
    model_type: str = Config.VanillaEfficientNet,
    load_state_dict: str = None,
    data_root: str = Config.Train,
    transform_type: str = Config.BaseTransform,
    epochs: int = Config.Epochs,
    batch_size: int = Config.BatchSize,
    optim_type: str = Config.Adam,
    lr: float = Config.LR,
    save_path: str = Config.ModelPath,
    seed: int = Config.Seed,
):
    print("="*100)
    verbose(task, model_type, load_state_dict, transform_type, epochs, batch_size, lr, optim_type, seed)
    print("="*100)

    set_seed(seed)
    trainloader = get_dataloader(task, "train", data_root, transform_type, batch_size)
    validloader = get_dataloader(task, "valid", data_root, transform_type, batch_size)

    n_classes = get_class_num(task)
    model = load_model(model_type, n_classes, load_state_dict)
    model.cuda()
    model.train()

    optimizer = get_optim(model, optim_type_=optim_type, lr=lr)

    if task != Task.Age: # main, ageg, mask, gender
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            print(f"Epoch: {epoch}")

            total_loss = 0
            total_corrects = 0
            num_samples = 0


            for idx, (imgs, labels) in tqdm(enumerate(trainloader), desc="Train"):
                imgs = imgs.cuda()
                labels = labels.cuda()

                output = model(imgs)
                loss = criterion(output, labels)
                _, pred_labels = torch.max(output.data, dim=1)

                total_corrects += (labels == pred_labels).sum().item()
                total_loss += loss
                num_samples += imgs.size(0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if idx != 0 and idx % 100 == 0:
                    val_loss, val_eval = validate(task, model, validloader, criterion)
                    temp_avg_loss = total_loss / num_samples
                    temp_avg_acc = total_corrects / num_samples
                    print(f"[Train] Avg Loss: {temp_avg_loss:.4f} Avg Acc: {temp_avg_acc:.4f}")

            avg_loss = total_loss / num_samples
            avg_acc = total_corrects / num_samples

            if save_path:
                name = f"{model_type}_task{task}_epoch{epoch:0>2d}_lr{lr}_transform{transform_type}_optim{optim_type}_loss{val_loss:.4f}_eval{val_eval:.4f}_seed{seed}.pth"
                torch.save(model.state_dict(), os.path.join(save_path, name))
        
    else: # age
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            print(f"Epoch: {epoch}")
            for idx, (imgs, labels) in tqdm(enumerate(trainloader), desc="Train"):
                imgs = imgs.cuda()
                labels = labels.float().cuda()

                output = model(imgs)
                loss = criterion(output, labels.unsqueeze(1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if idx != 0 and idx % 100 == 0:
                    val_loss, val_eval = validate(task, model, validloader, criterion)

            if save_path:
                name = f"{model_type}_task{task}_epoch{epoch:0>2d}_lr{lr}_transform{transform_type}_optim{optim_type}_loss{val_loss:.4f}_eval{val_eval:.4f}_seed{seed}.pth"
                torch.save(model.state_dict(), os.path.join(save_path, name))


def validate(task, model, validloader, criterion):
    
    if task != Task.Age:
        total_loss = 0
        total_corrects = 0
        num_samples = 0

        with torch.no_grad():
            model.eval()
            for imgs, labels in tqdm(validloader, desc="Valid"):
                imgs = imgs.cuda()
                labels = labels.cuda()

                output = model(imgs)
                loss = criterion(output, labels).item()
                _, pred_labels = torch.max(output.data, dim=1)

                total_corrects += (labels == pred_labels).sum().item()
                total_loss += loss
                num_samples += imgs.size(0)

            avg_loss = total_loss / num_samples
            avg_acc = total_corrects / num_samples
            print(f"[Valid] Avg Loss: {avg_loss:.4f} Avg Acc: {avg_acc:.4f}")
            model.train()

        return avg_loss, avg_acc

    else: # main, mask, ageg, gender
        mse = 0
        rmse = 0
        num_samples = 0

        with torch.no_grad():
            model.eval()
            for imgs, labels in tqdm(validloader, desc="Valid"):
                imgs = imgs.cuda()
                labels = labels.float().cuda()

                output = model(imgs)
                mse += criterion(output, labels.unsqueeze(1)).item()
                rmse += F.mse_loss(output, labels.unsqueeze(1)).item()
            
            rmse = math.sqrt(rmse)
            print(f"[Valid] MSE Loss: {mse:.4f} RMSE: {rmse:.4f}")
            model.train()
        
        return mse, rmse
    

if __name__ == "__main__":
    import wandb
    wandb.init()

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default=Task.Main, help=f"choose task among 'main', 'age', 'ageg', 'gender', 'mask' (default: {Task.Main})")
    parser.add_argument('--model-type', type=str, default=Config.VanillaEfficientNet, help=f'model type for train (default: {Config.VanillaEfficientNet})')
    parser.add_argument('--load-state-dict', type=dict, default=None, help=f'random seed (default: None)')
    parser.add_argument('--data-root', type=str, default=Config.Train, help=f"data directory for train (default: {Config.Train})")
    parser.add_argument('--transform-type', type=str, default=Config.BaseTransform, help=f'transform type for train (default: {Config.BaseTransform})')
    parser.add_argument('--epochs', type=int, default=Config.Epochs, help=f'number of epochs to train (default: {Config.Epochs})')
    parser.add_argument('--batch-size', type=int, default=Config.BatchSize, metavar='N', help=f'input batch size for training (default: {Config.BatchSize})')
    parser.add_argument('--optim-type', type=str, default=Config.Adam, help=f'optimizer type (default: {Config.Adam})')
    parser.add_argument('--lr', type=float, default=Config.LR, help=f'learning rate (default: {Config.LR})')
    parser.add_argument('--seed', type=int, default=Config.Seed, help=f'random seed (default: {Config.Seed})')
    parser.add_argument('--seed', type=int, default=Config.Seed, help=f'random seed (default: {Config.Seed})')
    parser.add_argument('--save-path', type=str, default=Config.ModelPath, help=f'random seed (default: {Config.ModelPath})')
    
    args = parser.parse_args()
    wandb.config.update(args) # adds all of the arguments as config variables

    train(args)
