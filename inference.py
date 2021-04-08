import argparse
import os
from glob import glob
from tqdm import tqdm
from functools import reduce
import pandas as pd
import numpy as np
import torch
from torch.nn import functional as F
import fire
from dataset import get_dataloader, LabelEncoder
from model import load_model
from config import Config, Task, Aug
from utils import get_timestamp, set_seed

LOAD_STATE_DICT = "./saved_models/main/VanillaEfficientNet_task(main)ep(14)f1(0.9306)bs(32)loss(0.0012)lr(0.001)trans(random)optim(adam)crit(labelsmoothingLoss)seed(42).pth"
ENSEMBLE = './saved_models/kfold-ensemble-VanillaEfficientNet_20210407'

def predict(
    task: str = Task.Main,
    model_type: str = Config.VanillaEfficientNet,
    load_state_dict: str = None,
    transform_type: str = Aug.BaseTransform,
    data_root: str = Config.Valid,
    save_path: str = Config.Inference,
):
    '''주어진 Train Data에 대한 inference. True Label과 Pred Label의 두가지 컬럼을 구성된 csv파일을 export
    '''

    # load phase
    if load_state_dict is None:
        load_state_dict = LOAD_STATE_DICT

    model = load_model(model_type, task, load_state_dict)
    model.cuda()
    model.eval()

    dataloader = get_dataloader(
        task=task,
        phase="test",
        data_root=data_root,
        transform_type=transform_type,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )

    # inference phase
    with torch.no_grad():
        pred_list = []
        true_list = []
        for imgs, labels in tqdm(dataloader, desc="Inference"):
            imgs = imgs.cuda()
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)

            pred_list.append(preds.item())
            true_list.append(labels.item())

    # export phase
    if save_path:
        model_name = os.path.basename(load_state_dict)
        if model_name not in os.listdir(save_path):
            os.mkdir(os.path.join(save_path, model_name))

        result = pd.DataFrame(dict(y_pred=pred_list, y_true=true_list))
        result.to_csv(
            os.path.join(save_path, model_name, "prediction.csv"), index=False
        )


def export_submission(
    task: str = Task.Main,
    model_type: str = Config.VanillaEfficientNet,
    load_state_dict: str = LOAD_STATE_DICT,
    transform_type: str = Aug.Random,
    data_root: str = Config.Eval,
    save_path: str = Config.Inference,
):
    # Inference by singular model
    if task != Task.MultiLabel:
        model = load_model(model_type, task, load_state_dict)
        model.cuda()
        model.eval()

        dataloader = get_dataloader(
            phase="eval",
            data_root=data_root,
            transform_type=transform_type,
            batch_size=1024,
            shuffle=False,
            drop_last=False,
        )

        with torch.no_grad():
            id_list = []
            pred_list = []

            for img_ids, imgs in tqdm(dataloader, desc="Inference"):
                imgs = imgs.cuda()
                output = model(imgs)
                _, preds = torch.max(output, 1)
                id_list.extend(img_ids)
                pred_list.extend(preds.data.cpu().numpy().flatten())

        prediction = pd.DataFrame(dict(ImageID=id_list, ans=pred_list))
        if save_path:
            model_name = os.path.basename(load_state_dict)
            if model_name not in os.listdir(save_path):
                os.mkdir(os.path.join(save_path, model_name))
            prediction.to_csv(
                os.path.join(save_path, model_name, "submission.csv"), index=False
            )

    # Inference by predicting multi-label(ageg, mask, gender)
    # TODO: 각 task별 파라미터를 불러오는 과정을 좀더 효율적으로 구성할 수는 없을까?
    else:
        if save_path:
            submission_dir = "all" + get_timestamp().replace("/", "-")
            if submission_dir not in os.listdir(save_path):
                os.mkdir(os.path.join(save_path, submission_dir))

        TASKS = ["mask", "gender", "ageg"]

        param_dict = dict()
        param_dict[
            "mask"
        ] = "VanillaEfficientNet_task(mask)ep(09)f1(0.9845)bs(64)loss(0.0007)lr(0.005)trans(base)optim(adam)crit(labelsmoothingLoss)seed(42).pth"
        param_dict[
            "gender"
        ] = "VanillaEfficientNet_task(gender)ep(08)f1(0.9736)bs(32)loss(0.0006)lr(0.001)trans(random)optim(adamp)crit(labelsmoothingLoss)seed(42).pth"
        param_dict[
            "ageg"
        ] = "VanillaEfficientNet_task(ageg)ep(02)f1(0.8562)loss(0.0008)lr(0.001)trans(random)optim(adamp)crit(labelsmoothingLoss)seed(42).pth"

        preds_dict = dict()  # Task별 예측값이 담길 딕셔너리
        for (
            t
        ) in (
            TASKS
        ):  # 불러온 파라미터 파일명에 dependent하므로 파일명 형식 유지 TODO: json 파일로 메타정보를 저장해두면 좋을 것 같다
            load_state_dict = os.path.join(Config.ModelPath, t, param_dict[t])
            model_type = param_dict[t].split("_")[0]
            transform_type = param_dict[t].split("trans(")[-1].split(")")[0]
            model = load_model(
                model_type=model_type, task=t, load_state_dict=load_state_dict
            )
            model.cuda()
            model.eval()

            dataloader = get_dataloader(
                phase="eval",
                data_root=data_root,
                transform_type=transform_type,
                batch_size=1024,
                shuffle=False,
                drop_last=False,
            )

            with torch.no_grad():
                id_list = []
                pred_list = []

                for img_ids, imgs in tqdm(dataloader, desc="Inference"):
                    imgs = imgs.cuda()
                    output = model(imgs)
                    _, preds = torch.max(output, 1)
                    id_list.extend(img_ids)
                    pred_list.extend(preds.data.cpu().numpy().flatten())

            pred_df = pd.DataFrame(dict(ImageID=id_list, ans=pred_list))
            preds_dict[t] = pred_df
            pred_df.to_csv(
                os.path.join(save_path, submission_dir, f"submission_{t}.csv"),
                index=False,
            )

        # make submission by integrating three tasks
        label_enc = LabelEncoder()
        submission = pd.DataFrame(dict(ImageID=id_list))
        for t in TASKS:
            submission[t] = preds_dict[t]["ans"]

        submission["ans"] = submission.apply(
            lambda x: label_enc.transform(tuple(x.iloc[1:]), task="main"), axis=1
        )
        submission[["ImageID", "ans"]].to_csv(
            os.path.join(save_path, submission_dir, "submission_main.csv"), index=False
        )

def export_submission_kfold_ensemble(
    task: str=Task.Main,
    root: str='./saved_models/kfold-ensemble-VanillaEfficientNet_20210407',
    transform_type: str = Aug.Random,
    data_root: str = Config.Eval,
    top_k: int=3, # fold별로 성능이 가장 높은 모델을 몇개씩 활용할지
    method: str='hard', 
    save_path: str = Config.Inference,
    tta: int = 1
):
    '''
    KFold CV를 통해 학습한 모델들 일부를 불러와 앙상블을 진행하는 함수
    '''
    set_seed(Config.Seed)
    if save_path is None:
        raise ValueError("Input 'save_path' to save submission files")

    # set export path
    save_dir = f'({method})-{os.path.basename(root)}'
    if save_dir not in os.listdir(save_path):
        os.mkdir(os.path.join(save_path, save_dir))
        save_path = os.path.join(save_path, save_dir)

    # load data
    if tta == 1:
        dataloader = get_dataloader(
                phase="eval",
                data_root=data_root,
                transform_type=transform_type,
                batch_size=1024,
                shuffle=False,
                drop_last=False,
            )
            
    elif tta > 1:
        transform_type = Aug.TTA
        dataloader = get_dataloader(
                phase="eval",
                data_root=data_root,
                transform_type=transform_type,
                batch_size=1024,
                shuffle=False,
                drop_last=False,
            )


    folds = glob(os.path.join(root))
    load_state_dict_list = []
    for fold in folds:
        model_params = glob(os.path.join(fold, '*'))
        model_params.sort(key=lambda x: float(x.split('f1(')[-1].split(')')[0]), reverse=True)
        load_state_dict_list.extend(model_params[:top_k])
    
    pred_list = []
    weight_list = []

    for load_state_dict in load_state_dict_list:
        fname = os.path.basename(load_state_dict)
        model_type = fname.split('_')[-1].split('_')[0]
        weight = float(fname.split('f1(')[-1].split(')')[0]) # for weighted ensemble

        model = load_model(model_type=model_type, task=task, load_state_dict=load_state_dict)
        model.cuda()
        model.eval()

        for i in range(tta):
            if tta > 1:
                print(f'TTA Iter: {i}')

            weight_list.append(weight)

            # inference phase
            with torch.no_grad():
                id_list = []
                fold_preds = []

                for img_ids, imgs in tqdm(dataloader, desc="Inference"):
                    imgs = imgs.cuda()
                    output = model(imgs)
                    preds = F.softmax(output, dim=1) # transform to probability
                    
                    if method == 'hard':
                        preds = torch.argmax(preds, dim=1) # get hard label

                    id_list.extend(img_ids)
                    fold_preds.append(preds.data.cpu().numpy())

            if method == 'hard':
                fold_preds = np.hstack(fold_preds)
                pred_df = pd.DataFrame(dict(ImageID=id_list, ans=fold_preds))

            elif method == 'soft':
                fold_preds = np.vstack(fold_preds)
                pred_id = pd.DataFrame(dict(ImageID=id_list))
                pred_df = pd.concat([pred_id, pd.DataFrame(fold_preds)], axis=1)
            
            # export each inference of singular model
            save_name = f"{fname.split('bs')[0]}_{i}.csv"
            pred_df.to_csv(os.path.join(save_path, save_name), index=False)
            pred_list.append(pred_df.drop('ImageID', axis=1))


    if method == 'hard':
        ensemble_hardvoting = pd.concat(pred_list, axis=1)
        ans_hardvoting = ensemble_hardvoting.apply(lambda x: x.value_counts().index[0], axis=1)
        submission_hardvoting = pd.DataFrame(dict(ImageID=id_list, ans=ans_hardvoting))
        submission_hardvoting.to_csv(os.path.join(save_path, 'submission_ens_hardvoting.csv'), index=False)


    elif method == 'soft':
        num_ensembles = len(pred_list)
        weights = map(lambda x: x / sum(weight_list), weight_list) # for weighted avg ensemble
        
        # get ensemble result by 3 methods
        ensemble_arithmetic = sum(pred_list) / num_ensembles
        ensemble_geometric = reduce(lambda x, y: x*y, pred_list) ** (num_ensembles**(-1))
        ensemble_weighted_avg = sum([w*pred for w, pred in zip(weights, pred_list)])

        # get hard label
        ans_arithmetic = ensemble_arithmetic.apply(lambda x: np.argmax(x), axis=1)
        ans_geometric = ensemble_geometric.apply(lambda x: np.argmax(x), axis=1)
        ans_weighted_avg = ensemble_weighted_avg.apply(lambda x: np.argmax(x), axis=1)

        # submission
        submission_arithmetic = pd.DataFrame(dict(ImageID=id_list, ans=ans_arithmetic))
        submission_geometric = pd.DataFrame(dict(ImageID=id_list, ans=ans_geometric))
        submission_weighted_avg = pd.DataFrame(dict(ImageID=id_list, ans=ans_weighted_avg))

        submission_arithmetic.to_csv(os.path.join(save_path, 'submission_ens_arithmetic.csv'), index=False)
        submission_geometric.to_csv(os.path.join(save_path, 'submission_ens_geometric.csv'), index=False)
        submission_weighted_avg.to_csv(os.path.join(save_path, 'submission_ens_weighted_avg.csv'), index=False)


if __name__ == "__main__":
    fire.Fire({"submit_singular": export_submission, "valid": predict, 'submit_ensemble': export_submission_kfold_ensemble})
