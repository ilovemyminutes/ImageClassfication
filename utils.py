import json
import random
import numpy as np
import torch


def save_json(path: str, f: object) -> None:
    with open(path, "w") as json_path:
        json.dump(
            f,
            json_path,
        )


def load_json(path: str) -> dict:
    with open(path, "r") as json_file:
        output = json.load(json_file)
    return output


def set_seed(seed: int = 42, contain_cuda: bool = False):
    random.seed(seed)
    np.random.seed(seed)

    if contain_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Seed set as {seed}")
