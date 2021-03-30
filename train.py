import os
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.nn import functional as F
import fire
from model import load_model
from config import Config
from dataset import get_dataloader
from utils import set_seed


def train(
    model_type: str = Config.VanillaEfficientNet,
    data_root: str = Config.Train,
    transform_type: str = Config.BaseTransform,
    epochs: int = Config.Epochs,
    batch_size: int = Config.BatchSize,
    lr: float = Config.LR,
    load_state_dict: str = None,
    save_path: str = Config.ModelPath,
    seed: int = Config.Seed,
):
    print("============Settings============")
    print(
        f"Model: {model_type}, Load: {load_state_dict}, Transform Type: {transform_type}, Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr}, Seed: {seed}"
    )
    print("================================")

    set_seed(seed)
    trainloader = get_dataloader("train", data_root, transform_type, batch_size)
    validloader = get_dataloader("valid", data_root, transform_type, batch_size)

    model = load_model(model_type, load_state_dict)
    model.cuda()
    model.train()

    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        for idx, (imgs, labels) in tqdm(enumerate(trainloader), desc="Train"):
            imgs = imgs.cuda()
            labels = labels.cuda()

            output = model(imgs)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx != 0 and idx % 100 == 0:
                avg_loss, avg_acc = validate(model, validloader, criterion)

        if save_path:
            name = f"{model_type}_epoch{epoch:0>2d}_lr{lr}_transform{transform_type}_loss{avg_loss:.4f}_acc{avg_acc:.4f}_seed{seed}.pth"
            torch.save(model.state_dict(), os.path.join(save_path, name))


def validate(model, validloader, criterion):
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


if __name__ == "__main__":
    fire.Fire({"run": train})
