from pathlib import Path

import dill

BASE_DIR = Path(__file__).parent


def save_model(model: dict, name: str, dir: str = BASE_DIR / "src" / "models/"):
    # print(dir.resolve().exists())
    # file_dir = Path(dir) if dir else
    file = Path(dir) / f"{name}.pkl"
    with open(file, "wb") as f:
        dill.dump(model, f)


def load_model(name: str, dir: str = BASE_DIR / "src" / "models/"):
    file = Path(dir) / f"{name}.pkl"
    with open(file, "rb") as f:
        model = dill.load(f)
    return model


def raise_error(condition, message):
    """Raise an error if the condition is True."""
    if condition:
        raise ValueError(message)
