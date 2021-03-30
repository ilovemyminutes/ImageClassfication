import os
from tqdm import tqdm
import pandas as pd
import torch
from torch.nn import functional as F
import fire
from dataset import get_dataloader
from model import VanillaResNet
from config import Config


LOAD_STATE_DICT = './saved_models/vanillaresnet_epoch29_transformbase_loss0.0036_acc0.8503_seed42.pth'

def predict(model_type: str=Config.VanillaResNet, load_state_dict: str=LOAD_STATE_DICT, transform_type: str=Config.BaseTransform, data_root: str=Config.Eval, save_path: str=Config.Inference):
    info = pd.read_csv(Config.Info)

    # if model_type == "vanillaresnet":
    model = VanillaResNet()
    # else:
    #     raise NotImplementedError()

    if load_state_dict:
        model.load_state_dict(torch.load(load_state_dict))

    model.cuda()
    model.eval()

    pred_dict ={}
    
    testloader = get_dataloader(phase='test', data_root=data_root, transform_type=transform_type, batch_size=1, shuffle=False, drop_last=False)

    for name, img in tqdm(testloader, desc='Inference'):
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
    fire.Fire(predict)

