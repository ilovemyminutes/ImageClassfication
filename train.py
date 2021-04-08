import argparse
from augmentation import configure_transform
import os
import math
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from model import load_model
from config import Config, Task, Loss, Aug
from dataset import get_dataloader, TrainDataset
from utils import age2ageg, set_seed, get_timestamp
from loss import get_criterion
from optims import get_optim, get_scheduler
import wandb


VALID_CYCLE = 250


def train(
    task: str = Task.AgeC,  # 수행할 태스크(분류-메인 태스크, 마스크 상태, 연령대, 성별, 회귀-나이)
    model_type: str = Config.VanillaEfficientNet,  # 불러올 모델명
    load_state_dict: str = None,  # 학습 이어서 할 경우 저장된 파라미터 경로
    train_root: str = Config.TrainS,  # 데이터 경로
    valid_root: str = Config.ValidS,
    transform_type: str = Aug.BaseTransform,  # 적용할 transform
    epochs: int = Config.Epochs,
    batch_size: int = Config.Batch32,
    optim_type: str = Config.Adam,
    loss_type: str = Loss.CE,
    lr: float = Config.LRBase,
    lr_scheduler: str = Config.CosineScheduler,
    save_path: str = Config.ModelPath,
    seed: int = Config.Seed,
):
    set_seed(seed)
    trainloader = get_dataloader(task, "train", train_root, transform_type, batch_size)
    validloader = get_dataloader(
        task, "valid", valid_root, transform_type, 1024, shuffle=False, drop_last=False
    )

    model = load_model(model_type, task, load_state_dict)
    model.cuda()
    model.train()

    optimizer = get_optim(model, optim_type=optim_type, lr=lr)
    criterion = get_criterion(loss_type=loss_type, task=task)

    if lr_scheduler is not None:
        scheduler = get_scheduler(scheduler_type=lr_scheduler, optimizer=optimizer)

    best_f1 = 0

    if task != Task.Age:  # classification(main, ageg, mask, gender)
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
                if lr_scheduler is not None:
                    scheduler.step()

                train_loss = total_loss / num_samples

                pred_arr = np.hstack(pred_list)
                true_arr = np.hstack(true_list)
                train_acc = (true_arr == pred_arr).sum() / len(true_arr)
                train_f1 = f1_score(y_true=true_arr, y_pred=pred_arr, average="macro")

                if epoch == 0:  # logs during just first epoch

                    wandb.log(
                        {
                            f"Ep{epoch:0>2d} Train F1": train_f1,
                            f"Ep{epoch:0>2d} Train ACC": train_acc,
                            f"Ep{epoch:0>2d} Train Loss": train_loss,
                        }
                    )

                if idx != 0 and idx % VALID_CYCLE == 0:
                    valid_f1, valid_acc, valid_loss = validate(
                        task, model, validloader, criterion
                    )

                    print(
                        f"[Valid] F1: {valid_f1:.4f} ACC: {valid_acc:.4f} Loss: {valid_loss:.4f}"
                    )
                    print(
                        f"[Train] F1: {train_f1:.4f} ACC: {train_acc:.4f} Loss: {train_loss:.4f}"
                    )
                    if epoch == 0:
                        # logs during one epoch
                        wandb.log(
                            {
                                f"Ep{epoch:0>2d} Valid F1": valid_f1,
                                f"Ep{epoch:0>2d} Valid ACC": valid_acc,
                                f"Ep{epoch:0>2d} Valid Loss": valid_loss,
                            }
                        )

            # logs for one epoch in total
            wandb.log(
                {
                    "Train F1": train_f1,
                    "Valid F1": valid_f1,
                    "Train ACC": train_acc,
                    "Valid ACC": valid_acc,
                    "Train Loss": train_loss,
                    "Valid Loss": valid_loss,
                }
            )

            if save_path and valid_f1 >= best_f1:
                name = f"{model_type}_task({task})ep({epoch:0>2d})f1({valid_f1:.4f})bs({batch_size})loss({valid_loss:.4f})lr({lr})trans({transform_type})optim({optim_type})crit({loss_type})seed({seed}).pth"
                best_f1 = valid_f1
                torch.save(model.state_dict(), os.path.join(save_path, name))

    # regression(age)
    else:
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
                scheduler.step()

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
                # wandb.log(
                #     {
                #         f"Ep{epoch:0>2d} Train F1": train_f1,
                #         f"Ep{epoch:0>2d} Train ACC": train_acc,
                #         f"Ep{epoch:0>2d} Train RMSE": train_rmse,
                #         f"Ep{epoch:0>2d} Train MSE": train_mse,
                #     }
                # )

                if idx != 0 and idx % VALID_CYCLE == 0:
                    valid_f1, valid_acc, valid_rmse, valid_mse = validate(
                        task, model, validloader, criterion
                    )
                    print(
                        f"[Valid] F1: {valid_f1:.4f} ACC: {valid_acc:.4f} RMSE: {valid_rmse:.4f} MSE: {valid_mse:.4f}"
                    )
                    print(
                        f"[Train] F1: {train_f1:.4f} ACC: {train_acc:.4f} RMSE: {train_rmse:.4f} MSE: {train_mse:.4f}"
                    )
                    # wandb.log(
                    #     {
                    #         "Valid F1": valid_f1,
                    #         "Valid ACC": valid_acc,
                    #         "Valid RMSE": valid_rmse,
                    #         "Valid MSE": valid_mse,
                    #     }
                    # )
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
                name = f"{model_type}_task({task})ep({epoch:0>2d})f1({valid_f1:.4f})bs({batch_size})loss({valid_mse:.4f})lr({lr})trans({transform_type})optim({optim_type})crit({loss_type})seed({seed}).pth"
                torch.save(model.state_dict(), os.path.join(save_path, name))


def train_cv(
    task: str = Task.AgeC,  # 수행할 태스크(분류-메인 태스크, 마스크 상태, 연령대, 성별, 회귀-나이)
    model_type: str = Config.VanillaEfficientNet,  # 불러올 모델명
    load_state_dict: str = None,  # 학습 이어서 할 경우 저장된 파라미터 경로
    train_root: str = Config.TrainS,  # 데이터 경로
    valid_root: str = Config.ValidS,
    transform_type: str = Aug.BaseTransform,  # 적용할 transform
    age_filter: int=58,
    epochs: int = Config.Epochs,
    cv: int = 5,
    batch_size: int = Config.Batch32,
    optim_type: str = Config.Adam,
    loss_type: str = Loss.CE,
    lr: float = Config.LRBase,
    lr_scheduler: str = Config.CosineScheduler,
    save_path: str = Config.ModelPath,
    seed: int = Config.Seed,
):
    if save_path:
        kfold_dir = f"kfold_{model_type}_" + get_timestamp()
        if kfold_dir not in os.listdir(save_path):
            os.mkdir(os.path.join(save_path, kfold_dir))
        print(f'Models will be saved in {os.path.join(save_path, kfold_dir)}.')

    set_seed(seed)
    transform = configure_transform(phase="train", transform_type=transform_type)
    trainset = TrainDataset(root=train_root, transform=transform, task=task, age_filter=age_filter, meta_path=Config.Metadata)
    validloader = get_dataloader(
        task, "valid", valid_root, transform_type, 1024, shuffle=False, drop_last=False
    )

    kfold = KFold(n_splits=cv, shuffle=True)

    for fold_idx, (train_indices, _) in enumerate(
        kfold.split(trainset)
    ):  # 앙상블이 목적이므로 test 인덱스는 따로 사용하지 않고, validloader를 통해 성능 검증
        if fold_idx == 0 or fold_idx == 1 or fold_idx == 2 or fold_idx == 3: continue
        print(f"Train Fold #{fold_idx}")
        train_sampler = SubsetRandomSampler(train_indices)
        trainloader = DataLoader(
            trainset, batch_size=batch_size, sampler=train_sampler, drop_last=True
        )

        model = load_model(model_type, task, load_state_dict)
        model.cuda()
        model.train()

        optimizer = get_optim(model, optim_type=optim_type, lr=lr)
        criterion = get_criterion(loss_type=loss_type, task=task)

        if lr_scheduler is not None:
            scheduler = get_scheduler(scheduler_type=lr_scheduler, optimizer=optimizer)

        best_f1 = 0

        if task != Task.Age:  # classification(main, ageg, mask, gender)
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
                    if lr_scheduler is not None:
                        scheduler.step()

                    train_loss = total_loss / num_samples

                    pred_arr = np.hstack(pred_list)
                    true_arr = np.hstack(true_list)
                    train_acc = (true_arr == pred_arr).sum() / len(true_arr)
                    train_f1 = f1_score(
                        y_true=true_arr, y_pred=pred_arr, average="macro"
                    )

                    if epoch == 0:  # logs during just first epoch

                        wandb.log(
                            {
                                f"Fold #{fold_idx} Ep{epoch:0>2d} Train F1": train_f1,
                                f"Fold #{fold_idx} Ep{epoch:0>2d} Train ACC": train_acc,
                                f"Fold #{fold_idx} Ep{epoch:0>2d} Train Loss": train_loss,
                            }
                        )

                    if idx != 0 and idx % VALID_CYCLE == 0:
                        valid_f1, valid_acc, valid_loss = validate(
                            task, model, validloader, criterion
                        )

                        print(
                            f"[Valid] F1: {valid_f1:.4f} ACC: {valid_acc:.4f} Loss: {valid_loss:.4f}"
                        )
                        print(
                            f"[Train] F1: {train_f1:.4f} ACC: {train_acc:.4f} Loss: {train_loss:.4f}"
                        )
                        if epoch == 0:
                            # logs during one epoch
                            wandb.log(
                                {
                                    f"Fold #{fold_idx} Ep{epoch:0>2d} Valid F1": valid_f1,
                                    f"Fold #{fold_idx} Ep{epoch:0>2d} Valid ACC": valid_acc,
                                    f"Fold #{fold_idx} Ep{epoch:0>2d} Valid Loss": valid_loss,
                                }
                            )

                # logs for one epoch in total
                wandb.log(
                    {
                        f"Fold #{fold_idx} Train F1": train_f1,
                        f"Fold #{fold_idx} Valid F1": valid_f1,
                        f"Fold #{fold_idx} Train ACC": train_acc,
                        f"Fold #{fold_idx} Valid ACC": valid_acc,
                        f"Fold #{fold_idx} Train Loss": train_loss,
                        f"Fold #{fold_idx} Valid Loss": valid_loss,
                    }
                )

                if save_path and valid_f1 >= best_f1:
                    name = f"Fold{fold_idx:0>2d}_{model_type}_task({task})ep({epoch:0>2d})f1({valid_f1:.4f})bs({batch_size})loss({valid_loss:.4f})lr({lr})trans({transform_type})optim({optim_type})crit({loss_type})seed({seed}).pth"
                    best_f1 = valid_f1
                    torch.save(
                        model.state_dict(), os.path.join(save_path, kfold_dir, name)
                    )

        # regression(age)
        else:
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
                    scheduler.step()

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
                    train_f1 = f1_score(
                        y_true=true_arr, y_pred=pred_arr, average="macro"
                    )

                    if idx != 0 and idx % VALID_CYCLE == 0:
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
                        f"Fold #{fold_idx} Train F1": train_f1,
                        f"Fold #{fold_idx} Valid F1": valid_f1,
                        f"Fold #{fold_idx} Train ACC": train_acc,
                        f"Fold #{fold_idx} Valid ACC": valid_acc,
                        f"Fold #{fold_idx} Train RMSE": train_rmse,
                        f"Fold #{fold_idx} Valid RMSE": valid_rmse,
                        f"Fold #{fold_idx} Train MSE": train_mse,
                        f"Fold #{fold_idx} Valid MSE": valid_mse,
                    }
                )

                if save_path:
                    name = f"Fold{fold_idx:0>2d}_{model_type}_task({task})ep({epoch:0>2d})f1({valid_f1:.4f})bs({batch_size})loss({valid_mse:.4f})lr({lr})trans({transform_type})optim({optim_type})crit({loss_type})seed({seed}).pth"
                    torch.save(
                        model.state_dict(), os.path.join(save_path, kfold_dir, name)
                    )
        model.cpu()


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
                loss = criterion(output, labels.cuda()).item()
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
    LOAD_STATE_DICT = None

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=Task.Main)
    parser.add_argument("--model-type", type=str, default=Config.VanillaEfficientNet)
    parser.add_argument("--load-state-dict", type=str, default=LOAD_STATE_DICT)
    parser.add_argument("--train-root", type=str, default=Config.Train)
    parser.add_argument("--valid-root", type=str, default=Config.Valid)
    parser.add_argument("--transform-type", type=str, default=Aug.Random)
    parser.add_argument("--age-filter", type=int, default=58)
    parser.add_argument("--epochs", type=int, default=Config.Epochs)
    parser.add_argument("--cv", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=Config.Batch32)
    parser.add_argument("--optim-type", type=str, default=Config.Adam)
    parser.add_argument("--loss-type", type=str, default=Loss.LS)
    parser.add_argument("--lr", type=float, default=Config.LRSlower)
    parser.add_argument("--lr-scheduler", type=str, default=Config.CosineScheduler)
    parser.add_argument("--seed", type=int, default=Config.Seed)
    parser.add_argument("--save-path", type=str, default=Config.ModelPath)

    args = parser.parse_args()
    name = args.model_type + "_" + args.task + "_" + get_timestamp()
    run = wandb.init(project="pstage-imageclf", name=name, reinit=True)

    wandb.config.update(args)  # adds all of the arguments as config variables
    print("=" * 100)
    print(args)
    print("=" * 100)

    if args.cv is not None:
        print(
            "Welcome to K-Fold CV Train! If you check it works, then you can go to sleep...😴"
        )
        train_cv(**vars(args))
    else:
        train(**vars(args))

    run.finish()
