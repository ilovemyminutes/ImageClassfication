import argparse
import os
import math
from numpy.lib.function_base import average
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score

import torch
from torch import nn
from torch.nn import functional as F

from model import load_model
from config import Config, Task
from dataset import get_dataloader
from utils import age2ageg, set_seed, get_timestamp
from optimizers import get_optim

import wandb


def train(
    task: str = Task.AgeC,  # 수행할 태스크(분류-메인 태스크, 마스크 상태, 연령대, 성별, 회귀-나이)
    model_type: str = Config.VanillaEfficientNet,  # 불러올 모델명
    load_state_dict: str = None,  # 학습 이어서 할 경우 저장된 파라미터 경로
    data_root: str = Config.Train,  # 데이터 경로
    transform_type: str = Config.BaseTransform,  # 적용할 transform
    epochs: int = Config.Epochs,
    batch_size: int = Config.BatchSize,
    optim_type: str = Config.Adam,
    lr: float = Config.LR,
    save_path: str = Config.ModelPath,
    seed: int = Config.Seed,
):
    set_seed(seed)
    trainloader = get_dataloader(task, "train", data_root, transform_type, batch_size)
    validloader = get_dataloader(
        task, "valid", data_root, transform_type, 1024, shuffle=False, drop_last=False
    )

    model = load_model(model_type, task, load_state_dict)
    model.cuda()
    model.train()

    optimizer = get_optim(model, optim_type_=optim_type, lr=lr)

    if task != Task.Age:  # classification(main, ageg, mask, gender)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            print(f"Epoch: {epoch}")

            # F1, ACC
            pred_list = []
            true_list = []

            # CE Loss
            total_loss = 0
            num_samples = 0

            for idx, (imgs, labels) in tqdm(enumerate(trainloader), desc="Train"):
                imgs = imgs.cuda()
                labels = labels.cuda()

                output = model(imgs)
                loss = criterion(output, labels)
                _, preds = torch.max(output.data, dim=1)

                pred_list.append(preds.data.cpu().numpy())
                true_list.append(labels.data.cpu().numpy())

                total_loss += loss
                num_samples += imgs.size(0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss = total_loss / num_samples

                pred_arr = np.hstack(pred_list)
                true_arr = np.hstack(true_list)
                train_acc = (true_arr == pred_arr).sum() / len(true_arr)
                train_f1 = f1_score(y_true=true_arr, y_pred=pred_arr, average="macro")

                # logs during one epoch
                wandb.log(
                    {
                        f"Ep{epoch:0>2d} Train F1": train_f1,
                        f"Ep{epoch:0>2d} Train Accuracy": train_acc,
                        f"Ep{epoch:0>2d} Train Loss": train_loss,
                    }
                )

                if idx != 0 and idx % 100 == 0:
                    valid_f1, valid_acc, valid_loss = validate(
                        task, model_type, model, validloader, criterion
                    )

                    print(
                        f"[Valid] F1: {valid_f1:.4f} ACC: {valid_acc:.4f} Loss: {valid_loss:.4f}"
                    )
                    print(
                        f"[Train] F1: {train_f1:.4f} ACC: {train_acc:.4f} Loss: {train_loss:.4f}"
                    )

                    wandb.log(
                        {
                            f"Ep{epoch:0>2d} Valid F1": valid_f1,
                            f"Ep{epoch:0>2d} Valid Accuracy": valid_acc,
                            f"Ep{epoch:0>2d} Valid Loss": valid_loss,
                        }
                    )

            # logs for one epoch in total
            wandb.log(
                {
                    "Train F1": train_f1,
                    "Valid F1": valid_f1,
                    "Train Accuracy": train_acc,
                    "Valid Accuracy": valid_acc,
                    "Train Loss": train_loss,
                    "Valid Loss": valid_loss,
                }
            )

            if save_path:
                name = f"{model_type}_task({task})ep({epoch:0>2d})f1({valid_f1:.4f})loss({valid_loss:.4f})lr({lr})trans({transform_type})optim({optim_type})seed({seed}).pth"
                torch.save(model.state_dict(), os.path.join(save_path, name))

    # regression(age)
    else:
        criterion = nn.MSELoss()
        # criterion = nn.SmoothL1Loss()
        for epoch in range(epochs):
            print(f"Epoch: {epoch}")

            pred_list = []
            true_list = []

            mse_raw = 0
            rmse_raw = 0
            num_samples = 0

            for idx, (imgs, labels) in tqdm(enumerate(trainloader), desc="Train"):
                imgs = imgs.cuda()

                # regression(age)
                labels_reg = labels.float().cuda()
                output = model(imgs)
                loss = criterion(output, labels_reg.unsqueeze(1))

                mse_raw += loss.item() * len(labels_reg)
                rmse_raw += loss.item() * len(labels_reg)
                num_samples += len(labels_reg)

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # classification(ageg)
                labels_clf = age2ageg(labels.data.numpy())
                preds_clf = age2ageg(output.data.cpu().numpy().flatten())
                pred_list.append(preds_clf)
                true_list.append(labels_clf)

                train_rmse = math.sqrt(rmse_raw / num_samples)
                train_mse = mse_raw / num_samples

                # eval for clf(ageg)
                pred_arr = np.hstack(pred_list)
                true_arr = np.hstack(true_list)

                train_acc = (true_arr == pred_arr).sum() / len(true_arr)
                train_f1 = f1_score(y_true=true_arr, y_pred=pred_arr, average="macro")

                # logs during one epoch
                wandb.log(
                    {
                        f"Ep{epoch:0>2d} Train F1": train_f1,
                        f"Ep{epoch:0>2d} Train ACC": train_acc,
                        f"Ep{epoch:0>2d} Train RMSE": train_rmse,
                        f"Ep{epoch:0>2d} Train MSE": train_mse,
                    }
                )

                if idx != 0 and idx % 100 == 0:
                    valid_f1, valid_acc, valid_rmse, valid_mse = validate(
                        task, model, validloader, criterion
                    )
                    print(
                        f"[Valid] F1: {valid_f1:.4f} ACC: {valid_acc:.4f} RMSE: {valid_rmse:.4f} MSE: {valid_mse:.4f}"
                    )
                    print(
                        f"[Train] F1: {train_f1:.4f} ACC: {train_acc:.4f} RMSE: {train_rmse:.4f} MSE: {train_mse:.4f}"
                    )
                    wandb.log(
                        {
                            "Valid F1": valid_f1,
                            "Valid ACC": valid_acc,
                            "Valid RMSE": valid_rmse,
                            "Valid MSE": valid_mse,
                        }
                    )
            wandb.log(
                {
                    "Train F1": train_f1,
                    "Valid F1": valid_f1,
                    "Train ACC": train_acc,
                    "Valid ACC": valid_acc,
                    "Train RMSE": train_rmse,
                    "Valid RMSE": valid_rmse,
                    "Train MSE": train_mse,
                    "Valid MSE": valid_mse,
                }
            )

            if save_path:
                name = f"{model_type}_task({task})ep({epoch:0>2d})f1({valid_f1:.4f})loss({valid_mse:.4f})lr({lr})trans({transform_type})optim({optim_type})seed({seed}).pth"
                torch.save(model.state_dict(), os.path.join(save_path, name))


def validate(task, model, validloader, criterion):

    # classification - main, mask, ageg, gender
    if task != Task.Age:

        pred_list = []
        true_list = []
        total_loss = 0

        with torch.no_grad():
            model.eval()
            for imgs, labels in tqdm(validloader, desc="Valid"):
                imgs = imgs.cuda()

                output = model(imgs)
                loss = criterion(output, labels).item()
                _, preds = torch.max(output, dim=1)

                pred_list.append(preds.data.cpu().numpy())
                true_list.append(labels.numpy())

                total_loss += loss

            pred_arr = np.hstack(pred_list)
            true_arr = np.hstack(true_list)

            valid_loss = total_loss / len(true_arr)
            valid_acc = (true_arr == pred_arr).sum() / len(true_arr)
            valid_f1 = f1_score(y_true=true_arr, y_pred=pred_arr, average="macro")
            model.train()

        return valid_f1, valid_acc, valid_loss

    # regression - age
    else:

        # evaluation for clf(ageg)
        pred_list = []
        true_list = []

        # evaluation for reg(age)
        mse_raw = 0
        rmse_raw = 0
        num_samples = 0

        with torch.no_grad():
            model.eval()
            for imgs, labels in tqdm(validloader, desc="Valid"):
                imgs = imgs.cuda()
                output = model(imgs)

                # regression(age)
                labels_reg = labels.float().cuda()
                mse_raw += criterion(output, labels_reg.unsqueeze(1)).item() * len(
                    labels_reg
                )
                rmse_raw += F.mse_loss(output, labels_reg.unsqueeze(1)).item() * len(
                    labels_reg
                )
                num_samples += len(labels_reg)

                # classification(ageg)
                labels_clf = age2ageg(labels.data.numpy())
                preds_clf = age2ageg(output.data.cpu().numpy().flatten())
                pred_list.append(preds_clf)
                true_list.append(labels_clf)

            # eval/loss for reg(age)
            valid_rmse = math.sqrt(rmse_raw / num_samples)  # loss
            valid_mse = mse_raw / num_samples  # eval

            # eval for clf(ageg)
            pred_arr = np.hstack(pred_list)
            true_arr = np.hstack(true_list)

            valid_acc = (true_arr == pred_arr).sum() / len(true_arr)
            valid_f1 = f1_score(y_true=true_arr, y_pred=pred_arr, average="macro")

            model.train()

        return valid_f1, valid_acc, valid_rmse, valid_mse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( "--task", type=str, default=Task.AgeC, help=f"choose task among 'main', 'age', 'ageg', 'gender', 'mask', 'all' (default: {Task.Main})")
    parser.add_argument( "--model-type", type=str, default=Config.VanillaEfficientNet, help=f"model type for train (default: {Config.VanillaEfficientNet})")
    parser.add_argument( "--load-state-dict", type=str, default=None, help=f"(optional) state dict path for continuous train (default: None)")
    parser.add_argument( "--data-root", type=str, default=Config.Train, help=f"data directory for train (default: {Config.Train})")
    parser.add_argument( "--transform-type", type=str, default=Config.BaseTransform, help=f"transform type for train (default: {Config.BaseTransform})")
    parser.add_argument("--epochs",type=int,default=Config.Epochs,help=f"number of epochs to train (default: {Config.Epochs})")
    parser.add_argument("--batch-size",type=int,default=Config.BatchSize,metavar="N",help=f"input batch size for training (default: {Config.BatchSize})")
    parser.add_argument("--optim-type",type=str,default=Config.Adam,help=f"optimizer type (default: {Config.Adam})")
    parser.add_argument("--lr",type=float,default=Config.LR,help=f"learning rate (default: {Config.LR})")
    parser.add_argument("--seed",type=int,default=Config.Seed,help=f"random seed (default: {Config.Seed})")
    parser.add_argument("--save-path",type=str,default=Config.ModelPath,help=f"random seed (default: {Config.ModelPath})")

    args = parser.parse_args()
    name = args.model_type + '_' + args.task + '_' + get_timestamp()
    run = wandb.init(project="pstage-imageclf", name=name, reinit=True)
    
    wandb.config.update(args)  # adds all of the arguments as config variables
    print("=" * 100)
    print(args)
    print("=" * 100)
    train(**vars(args))
    run.finish()