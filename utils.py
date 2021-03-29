import json

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