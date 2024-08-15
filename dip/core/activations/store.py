import glob
import os

import torch as t
from tqdm import tqdm

ACTS_BATCH_SIZE = 25


def collect_acts(
    directory,
    layer,
    center=True,
    scale=False,
    device="cpu",
    acts_batch_size: int = ACTS_BATCH_SIZE,
):
    """
    Collects activations from a dataset of statements, returns as a tensor of shape [n_activations, activation_dimension].
    """
    activation_files = glob.glob(os.path.join(directory, f"layer_{layer}_*.pt"))
    acts = []
    for i in tqdm(
        range(0, acts_batch_size * len(activation_files), acts_batch_size),
        desc=f"Collecting activations from layer {layer}",
    ):
        acts.append(t.load(os.path.join(directory, f"layer_{layer}_{i}.pt")).to(device))
    acts = t.cat(acts, dim=0).to(device)
    if center:
        acts = acts - t.mean(acts, dim=0)
    if scale:
        acts = acts / t.std(acts, dim=0)
    return acts
