import json
import pickle

import numpy as np


def dump_pickle(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def load_pickle(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def dump_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_array(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def load_array(path):
    return np.load(path)
