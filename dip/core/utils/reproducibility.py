import glob
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int = 42):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def load_checkpoint(save_dir):
    answers_files = glob.glob(f"{save_dir}/answers_*.csv")
    indexes = [
        int(answer_file.split("_")[-1].split(".")[0]) for answer_file in answers_files
    ]
    return max(indexes) if indexes else 0


def save_results(
    results: dict, aspect_names: list[str], class_names: list[str], save_path: str
):
    results = {
        aspect_names[i]: {
            class_names[j]: round(results[i][j], 4) for j in results[i].keys()
        }
        for i in results.keys()
    }

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)
