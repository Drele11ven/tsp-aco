# src/tsp_utils.py
import numpy as np
import json
import os
from datetime import datetime

def euclidean_distance_matrix(coords):
    coords = np.array(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    return dist

def random_euclidean_instance(n_nodes=20, seed=None, bounds=(0, 100)):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(bounds[0], bounds[1], size=(n_nodes, 2))
    return xs

def ensure_results_dir(base="results"):
    os.makedirs(base, exist_ok=True)
    return base

def make_run_folder(base="results"):
    ensure_results_dir(base)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(base, ts)
    os.makedirs(folder, exist_ok=False)
    return folder

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
