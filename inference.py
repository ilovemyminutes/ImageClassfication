import os
from tqdm import tqdm
import pandas as pd
import torch
from torch.nn import functional as F
import fire
from dataset import get_dataloader
from model import VanillaResNet
from config import Config


LOAD_STATE_DICT = './saved_models/vanillaresnet_epoch04_lr0.0005_transformbase_loss0.0050_acc0.8092_seed42.pth'

def test_predict(model_type: str=Config.VanillaResNet, load_state_dict: str=LOAD_STATE_DICT, transform_type: str=Config.BaseTransform, data_root: str=Config.Test, save_path: str=Config.Inference):
    if model_type == "VanillaResNet":
        model = VanillaResNet()
    else:
        raise NotImplementedError()

    if load_state_dict:
        model.load_state_dict(torch.load(load_state_dict))
    
    model.cuda()
    model.eval()

    dataloader = get_dataloader(phase='test', data_root=data_root, transform_type=transform_type, batch_size=1, shuffle=False, drop_last=False)

    with torch.no_grad():
        pred_list = []
        true_list = []
        for imgs, labels in tqdm(dataloader, desc='Inference'):
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
        result.to_csv(os.path.join(save_path, model_name, 'prediction.csv'), index=False)
    
    

def eval_predict(model_type: str=Config.VanillaResNet, load_state_dict: str=LOAD_STATE_DICT, transform_type: str=Config.BaseTransform, data_root: str=Config.Eval, save_path: str=Config.Inference):
    info = pd.read_csv(Config.Info)

    if model_type == "VanillaResNet":
        model = VanillaResNet()
    else:
        raise NotImplementedError()

    if load_state_dict:
        model.load_state_dict(torch.load(load_state_dict))

    model.cuda()
    model.eval()

    dataloader = get_dataloader(phase='eval', data_root=data_root, transform_type=transform_type, batch_size=1, shuffle=False, drop_last=False)
    pred_dict = {}
    for name, img in tqdm(dataloader, desc='Inference'):
        name, img = name[0], img.cuda()
        outputs = model(img)
        _, pred_label = torch.max(outputs, 1)
        pred_dict[name] = int(pred_label.item())

    prediction_raw = pd.Series(pred_dict).to_frame('ans')
    prediction_raw = prediction_raw.reset_index().rename({'index':'ImageID'}, axis=1)
    prediction = info[['ImageID']].merge(prediction_raw, how='left', on='ImageID')

    if save_path:
        model_name = os.path.basename(load_state_dict)
        if model_name not in os.listdir(save_path):
            os.mkdir(os.path.join(save_path, model_name))
        prediction.to_csv(os.path.join(save_path, model_name, 'submission.csv'), index=False)


if __name__ == '__main__':
    fire.Fire({'eval': eval_predict, 'test': test_predict})

