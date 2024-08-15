import torch as t
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class LDAProbe(t.nn.Module):
    def __init__(self, net) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: t.Tensor, iid=None) -> t.Tensor:
        return t.from_numpy(self.net.predict(x.cpu().numpy()))

    def pred(self, x: t.Tensor, iid=None) -> t.Tensor:
        return self(x, iid)

    def from_data(acts: t.Tensor, labels: t.Tensor, **kwargs) -> "LDAProbe":
        probe = LinearDiscriminantAnalysis()
        probe.fit(acts.cpu().numpy(), labels.cpu().numpy())
        return LDAProbe(probe)

    @property
    def cav(self) -> t.Tensor:
        return t.from_numpy(self.net.coef_[0])
