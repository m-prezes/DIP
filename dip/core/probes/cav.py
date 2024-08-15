from pathlib import Path

import torch
from sklearn.metrics import accuracy_score


class CAV:
    def __init__(self, probe_class, device, verbose=True):
        self.probe_class = probe_class
        self.device = device
        self.verbose = verbose
        self.cav = None

    def fit(self, dataset, activations, aspect):
        labels = dataset[aspect]
        labels = torch.tensor(labels).to(self.device)
        labels = labels.type(torch.float32)

        activations = activations.type(torch.float32)
        activations = activations.to(self.device)

        self.cav = self.learn_cav(activations, labels, aspect)
        self.cav = self.cav / self.cav.norm()
        self.cav = self.cav.to(self.device)

    def learn_cav(self, activations, labels, aspect):
        probe = self.probe_class.from_data(activations, labels, device=self.device)
        accuracy = accuracy_score(
            labels.cpu().detach().numpy(),
            probe.pred(activations).cpu().detach().numpy(),
        )

        cav = probe.cav

        if self.verbose:
            print(f"Learned CAV for concept: {aspect}")
            print(f"\t{cav[:2]}...{cav[-2:]}")
            print(f"\tAccuracy: {accuracy * 100:.1f}%")
            print()
        return cav


def save_cav(cav, path, name):
    Path(path).mkdir(parents=True, exist_ok=True)
    torch.save(cav, f"{path}/{name}.pt")


def load_cav(path, name):
    return torch.load(f"{path}/{name}.pt")
