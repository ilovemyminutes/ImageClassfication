import os
from tqdm import tqdm
import pandas as pd
import torch
from torch.nn import functional as F
import fire
from dataset import get_dataloader, LabelEncoder
from model import load_model
from config import Config, Task, get_class_num, Aug
from utils import get_timestamp

LOAD_STATE_DICT = "./saved_models/main/VanillaEfficientNet_task(main)ep(14)f1(0.9306)bs(32)loss(0.0012)lr(0.001)trans(random)optim(adam)crit(labelsmoothingLoss)seed(42).pth"


def predict(
    task: str = Task.Main,
    model_type: str = Config.VanillaEfficientNet,
    load_state_dict: str = None,
    transform_type: str = Aug.BaseTransform,
    data_root: str = Config.Valid,
    save_path: str = Config.Inference,
):
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


def predict_submission(
    task: str = Task.Main,
    model_type: str = Config.VanillaEfficientNet,
    load_state_dict: str = LOAD_STATE_DICT,
    transform_type: str = Aug.Random,
    data_root: str = Config.Eval,
    save_path: str = Config.Inference,
):
    if task != Task.All:
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

    elif task == Task.All:
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
        )  # 미래의 지형아 미안하다
        submission[["ImageID", "ans"]].to_csv(
            os.path.join(save_path, submission_dir, "submission_main.csv"), index=False
        )

    elif task == "ensemble":
        raise NotImplementedError()


if __name__ == "__main__":
    fire.Fire({"submit": predict_submission, "pred": predict})
