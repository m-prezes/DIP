from pathlib import Path

import seaborn as sns
from master_thesis.core.activations.store import collect_acts
from matplotlib import pyplot as plt
from sklearn.metrics import calinski_harabasz_score


class CalinskiHarabaszIndex:
    def __init__(
        self,
        activations_dir,
        layers,
        dataset,
        aspect,
    ):
        self.activations_dir = activations_dir
        self.layers = layers
        self.dataset = dataset
        self.aspect = aspect

    def get_score(self, activations):
        labels = self.dataset[self.aspect]
        score = calinski_harabasz_score(activations, labels)
        return score

    def plot_separability_score(self, show=False, save_path=None):
        separability_scores = []
        for layer in self.layers:
            activations = collect_acts(
                self.activations_dir,
                layer=layer,
                center=False,
                scale=False,
            )
            separability_scores.append(self.get_score(activations))

        first_best_layer = self.layers[
            separability_scores.index(max(separability_scores))
        ]

        sns.set(style="whitegrid")
        fig, ax = plt.subplots()
        ax.plot(self.layers, separability_scores)

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
        ax.set_title(f"Separability of {self.aspect} (Calinski Harabasz Index)")

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
        if show:
            plt.show()
