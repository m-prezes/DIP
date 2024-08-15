import numpy as np
import torch as t
from sklearn.linear_model import SGDClassifier


class SVMProbe(t.nn.Module):
    def __init__(self, net) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: t.Tensor, iid=None) -> t.Tensor:
        return t.from_numpy(self.net.predict(x.cpu().numpy()))

    def pred(self, x: t.Tensor, iid=None) -> t.Tensor:
        return self(x, iid)

    def from_data(acts: t.Tensor, labels: t.Tensor, **kwargs) -> "SVMProbe":
        probe = SGDClassifier(alpha=0.01, max_iter=1000, tol=1e-3)
        probe.fit(acts.cpu().numpy(), labels.cpu().numpy())
        return SVMProbe(probe)

    @property
    def cav(self) -> t.Tensor:
        cav = self.net.coef_[0]
        cav = cav / np.linalg.norm(cav)
        return t.from_numpy(cav)
