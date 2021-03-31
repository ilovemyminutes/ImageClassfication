import os
from tqdm import tqdm
import pandas as pd
import torch
from torch.nn import functional as F
import fire
from dataset import get_dataloader
from model import load_model
from config import Config


LOAD_STATE_DICT = "./saved_models/VanillaResNet_epoch26_lr0.00025_transformbase_loss0.0037_acc0.8436_seed42.pth"

    
def predict(
    model_type: str = Config.VanillaResNet,
    load_state_dict: str = LOAD_STATE_DICT,
    transform_type: str = Config.BaseTransform,
    data_root: str = Config.Test,
    save_path: str = Config.Inference,
):
    model = load_model(model_type, load_state_dict)
    model.cuda()
    model.eval()

    dataloader = get_dataloader(
        phase="test",
        data_root=data_root,
        transform_type=transform_type,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )

    with torch.no_grad():
        pred_list = []
        true_list = []
        for imgs, labels in tqdm(dataloader, desc="Inference"):
            imgs = imgs.cuda()
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)

            pred_list.append(preds.item())
            true_list.append(labels.item())

    if save_path:
        model_name = os.path.basename(load_state_dict)
        if model_name not in os.listdir(save_path):
            os.mkdir(os.path.join(save_path, model_name))

        result = pd.DataFrame(dict(y_pred=pred_list, y_true=true_list))
        result.to_csv(
            os.path.join(save_path, model_name, "prediction.csv"), index=False
        )


def submit(
    model_type: str = Config.VanillaResNet,
    load_state_dict: str = LOAD_STATE_DICT,
    transform_type: str = Config.BaseTransform,
    data_root: str = Config.Eval,
    save_path: str = Config.Inference,
):
    model = load_model(model_type, load_state_dict)
    model.cuda()
    model.eval()

    dataloader = get_dataloader(
        phase="eval",
        data_root=data_root,
        transform_type=transform_type,
        batch_size=1,
        shuffle=False,
        drop_last=False
    )

    with torch.no_grad():
        id_list = []
        pred_list = []
        for img_id, img in tqdm(dataloader, desc="Inference"):
            img = img.cuda()
            output = model(img)
            _, pred = torch.max(output, 1)
            id_list.append(img_id[0])
            pred_list.append(pred.item())

    prediction = pd.DataFrame(dict(ImageID=id_list, ans=pred_list))
    if save_path:
        model_name = os.path.basename(load_state_dict)
        if model_name not in os.listdir(save_path):
            os.mkdir(os.path.join(save_path, model_name))
        prediction.to_csv(
            os.path.join(save_path, model_name, "submission.csv"), index=False
        )


if __name__ == "__main__":
    fire.Fire({"eval": submit, "pred": predict})
