import argparse
import os
import math
from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
import wandb
from model import load_model
from config import Config, get_optim, Task, get_class_num
from dataset import get_dataloader
from utils import set_seed, get_timestamp


def train(
    task: str = Task.Age, # 수행할 태스크(분류-메인 태스크, 마스크 상태, 연령대, 성별, 회귀-나이)
    model_type: str = Config.VanillaEfficientNet, # 불러올 모델명
    load_state_dict: str = None, # 학습 이어서 할 경우 저장된 파라미터 경로
    data_root: str = Config.Train, # 데이터 경로
    transform_type: str = Config.BaseTransform, # 적용할 transform
    epochs: int = Config.Epochs,
    batch_size: int = Config.BatchSize,
    optim_type: str = Config.Adam,
    lr: float = Config.LR,
    save_path: str = Config.ModelPath,
    seed: int = Config.Seed,
):
    set_seed(seed)
    trainloader = get_dataloader(task, "train", data_root, transform_type, batch_size)
    validloader = get_dataloader(task, "valid", data_root, transform_type, 1024)

    n_classes = get_class_num(task)
    model = load_model(model_type, n_classes, load_state_dict)
    model.cuda()
    model.train()

    optimizer = get_optim(model, optim_type_=optim_type, lr=lr)

    # classification(main, ageg, mask, gender)
    if task != Task.Age:  
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

                avg_loss = total_loss / num_samples
                avg_acc = total_corrects / num_samples

                # logs during one epoch
                wandb.log(
                    {
                        f"Ep{epoch:0>2d} Train Accuracy": avg_acc,
                        f"Ep{epoch:0>2d} Train Loss": avg_loss,
                    }
                )

                if idx != 0 and idx % 100 == 0:
                    val_loss, val_eval = validate(task, model, validloader, criterion)
                    print(f"[Train] Avg Loss: {avg_loss:.4f} Avg Acc: {avg_acc:.4f}")
                    wandb.log(
                        {
                            f"Ep{epoch:0>2d} Valid Accuracy": val_eval,
                            f"Ep{epoch:0>2d} Valid Loss": val_loss,
                        }
                    )

            # logs for one epoch in total
            wandb.log(
                {
                    "Train Accuracy": avg_acc,
                    "Valid Accuracy": val_eval,
                    "Train Loss": avg_loss,
                    "Valid Loss": val_loss,
                }
            )

            if save_path:
                name = f"{model_type}_task{task}_epoch{epoch:0>2d}_lr{lr}_transform{transform_type}_optim{optim_type}_loss{val_loss:.4f}_eval{val_eval:.4f}_seed{seed}.pth"
                torch.save(model.state_dict(), os.path.join(save_path, name))

    # regression(age)
    else:
        criterion = nn.MSELoss()
        # criterion = nn.SmoothL1Loss() 
        for epoch in range(epochs):
            print(f"Epoch: {epoch}")

            mse_raw = 0
            rmse_raw = 0
            num_samples = 0

            for idx, (imgs, labels) in tqdm(enumerate(trainloader), desc="Train"):
                imgs = imgs.cuda()
                labels = labels.float().cuda()

                output = model(imgs)
                loss = criterion(output, labels.unsqueeze(1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                mse_raw += criterion(output, labels.unsqueeze(1)).item() * len(labels)
                rmse_raw += F.mse_loss(output, labels.unsqueeze(1)).item() * len(labels)
                num_samples += len(labels)

                mse = mse_raw / num_samples
                rmse = math.sqrt(rmse_raw / num_samples)

                # logs during one epoch
                wandb.log(
                    {
                        f"Ep{epoch:0>2d} Train RMSE": math.sqrt(rmse),
                        f"Ep{epoch:0>2d} Train MSE Loss": mse,
                    }
                )

                if idx != 0 and idx % 100 == 0:
                    val_mse, val_rmse = validate(task, model, validloader, criterion)
                    print(
                        f"[Train] MSE Loss: {mse:.4f} RMSE: {rmse:.4f}"
                    )
                    wandb.log(
                    {
                        "Valid RMSE": val_rmse,
                        "Valid MSE Loss": val_mse,
                    }
                )

            wandb.log(
                {
                    "Train RMSE": rmse,
                    "Valid RMSE": val_rmse,
                    "Train MSE Loss": mse,
                    "Valid MSE Loss": val_mse,
                }
            )

            if save_path:
                name = f"{model_type}_task{task}_epoch{epoch:0>2d}_lr{lr}_transform{transform_type}_optim{optim_type}_loss{val_mse:.4f}_eval{val_rmse:.4f}_seed{seed}.pth"
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

    else:  # main, mask, ageg, gender
        mse_raw = 0
        rmse_raw = 0
        num_samples = 0

        with torch.no_grad():
            model.eval()
            for imgs, labels in tqdm(validloader, desc="Valid"):
                imgs = imgs.cuda()
                labels = labels.float().cuda()
                
                output = model(imgs)

                mse_raw += criterion(output, labels.unsqueeze(1)).item() * len(labels)
                rmse_raw += F.mse_loss(output, labels.unsqueeze(1)).item() * len(labels)
                num_samples += len(labels)

            mse = mse_raw / num_samples
            rmse = math.sqrt(rmse_raw / num_samples)
            print(f"[Valid] MSE Loss: {mse:.4f} RMSE: {rmse:.4f}")
            model.train()

        return mse, rmse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default=Task.Main,
        help=f"choose task among 'main', 'age', 'ageg', 'gender', 'mask' (default: {Task.Main})",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=Config.VanillaEfficientNet,
        help=f"model type for train (default: {Config.VanillaEfficientNet})",
    )
    parser.add_argument(
        "--load-state-dict",
        type=str,
        default=None,
        help=f"(optional) state dict path for continuous train (default: None)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=Config.Train,
        help=f"data directory for train (default: {Config.Train})",
    )
    parser.add_argument(
        "--transform-type",
        type=str,
        default=Config.BaseTransform,
        help=f"transform type for train (default: {Config.BaseTransform})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=Config.Epochs,
        help=f"number of epochs to train (default: {Config.Epochs})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=Config.BatchSize,
        metavar="N",
        help=f"input batch size for training (default: {Config.BatchSize})",
    )
    parser.add_argument(
        "--optim-type",
        type=str,
        default=Config.Adam,
        help=f"optimizer type (default: {Config.Adam})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=Config.LR,
        help=f"learning rate (default: {Config.LR})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=Config.Seed,
        help=f"random seed (default: {Config.Seed})",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=Config.ModelPath,
        help=f"random seed (default: {Config.ModelPath})",
    )
    args = parser.parse_args()
    name = args.model_type + '_' + args.task + '_' + get_timestamp()
    run = wandb.init(project="pstage-imageclf", name=name, reinit=True)
    
    wandb.config.update(args)  # adds all of the arguments as config variables
    print("=" * 100)
    print(args)
    print("=" * 100)
    train(**vars(args))
    run.finish()
