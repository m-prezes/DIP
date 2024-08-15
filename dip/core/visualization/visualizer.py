import torch as t
from plotly import express as px


class Visualizer:
    def __init__(self, df):
        self.df = df

    def plot(
        self,
        dimensions,
        layer=None,
        color_label=None,
        **kwargs,
    ):
        acts = self.df["activation"].tolist()
        acts = t.stack(acts, dim=0).cuda()
        pcs, preserved_variance = self.get_pcs(acts, dimensions)

        proj = t.mm(acts, pcs)
        for dim in range(dimensions):
            self.df[f"PC{dim+1}"] = proj[:, dim].tolist()

        # plot using plotly
        if dimensions == 2:
            fig = px.scatter(
                self.df,
                x="PC1",
                y="PC2",
                hover_name="sentence",
                color_continuous_scale="Bluered_r",
                color=color_label,
                **kwargs,
            )
        elif dimensions == 3:
            fig = px.scatter_3d(
                self.df,
                x="PC1",
                y="PC2",
                z="PC3",
                hover_name="sentence",
                color_continuous_scale="Bluered_r",
                color=color_label,
                **kwargs,
            )
        else:
            raise ValueError("Dimensions must be 2 or 3")

        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
        )

        fig.update_layout(
            coloraxis_showscale=False,
        )

        fig.update_layout(
            title={
                "text": f"Warstwa {layer} - zachowana wariancja {preserved_variance:.2f}",
                "x": 0.5,
                "xanchor": "center",
            }
        )

        fig.update_traces(marker=dict(size=3))
        fig.update_traces(opacity=0.5)

        return fig

    def get_pcs(self, X, k=2):
        """
        Performs Principal Component Analysis (PCA) on the n x d data matrix X.
        Returns the k principal components, the corresponding eigenvalues and the projected data.
        """
        X = X - t.mean(X, dim=0)
        cov_mat = t.mm(X.t(), X) / (X.size(0) - 1)
        eigenvalues, eigenvectors = t.linalg.eigh(cov_mat)
        sorted_indices = t.argsort(eigenvalues, descending=True)
        eigenvectors = eigenvectors[:, sorted_indices]
        eigenvectors = eigenvectors[:, :k]
        total_variance = t.sum(eigenvalues)
        sorted_eigenvalues = t.sort(eigenvalues, descending=True).values
        preserved_variance = t.sum(sorted_eigenvalues[:k]) / total_variance
        return eigenvectors, preserved_variance
