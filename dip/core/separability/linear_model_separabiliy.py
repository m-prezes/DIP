from pathlib import Path

import seaborn as sns
import torch
from master_thesis.core.activations.store import collect_acts
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score


class LinearModelSeparability:
    def __init__(
        self,
        probe_class,
        device,
        activations_dir,
        layers,
        dataset,
        aspect,
        base_accuracy=None,
    ):
        self.probe_class = probe_class
        self.device = device
        self.activations_dir = activations_dir
        self.layers = layers
        self.base_accuracy = base_accuracy
        self.dataset = dataset
        self.aspect = aspect

    def get_accuracy(self, activations):
        labels = self.dataset[self.aspect]
        labels = torch.tensor(labels).to(self.device)
        labels = labels.type(torch.float32)

        activations = activations.type(torch.float32)
        activations = activations.to(self.device)

        probe = self.probe_class.from_data(activations, labels, device=self.device)
        accuracy = accuracy_score(
            labels.cpu().detach().numpy(),
            probe.pred(activations).cpu().detach().numpy(),
        )
        return accuracy

    def plot_separability_score(self, show=False, save_path=None):
        separability_scores = []
        for layer in self.layers:
            activations = collect_acts(
                self.activations_dir,
                layer=layer,
                center=False,
                scale=False,
            )
            separability_scores.append(self.get_accuracy(activations))

        first_best_layer = self.layers[
            separability_scores.index(max(separability_scores))
        ]

        sns.set(style="whitegrid")
        fig, ax = plt.subplots()
        ax.plot(self.layers, separability_scores)
        if self.base_accuracy:
            ax.axhline(y=self.base_accuracy, color="r", linestyle="--")

        # add horizontal line for best layer
        ax.axvline(x=first_best_layer, color="g", linestyle="--")
        ax.plot(first_best_layer, separability_scores[first_best_layer - 1], "go")
        ax.text(
            first_best_layer,
            separability_scores[first_best_layer - 1],
            f"Layer: {first_best_layer}",
            fontsize=10,
        )

        ax.legend(["Accuracy", "Base Accuracy"])

        ax.set_xlabel("Layer")
        ax.set_ylabel("Accuracy")
        ax.set_title(
            f"Linear Separability of {self.aspect} ({self.probe_class.__name__})"
        )

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
        if show:
            plt.show()
