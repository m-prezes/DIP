import torch as t


class MMProbe(t.nn.Module):
    def __init__(
        self,
        cav: t.Tensor,
        covariance: t.Tensor = None,
        inv: t.Tensor = None,
        atol: float = 1e-3,
    ) -> None:
        super().__init__()
        self.cav = t.nn.Parameter(cav, requires_grad=False)
        if inv is None:
            self.inv = t.nn.Parameter(
                t.linalg.pinv(covariance, hermitian=True, atol=atol),
                requires_grad=False,
            )
        else:
            self.inv = t.nn.Parameter(inv, requires_grad=False)

    def forward(self, x: t.Tensor, iid: bool = False) -> t.Tensor:
        if iid:
            return t.nn.Sigmoid()(x @ self.inv @ self.cav)
        else:
            return t.nn.Sigmoid()(x @ self.cav)

    def pred(self, x: t.Tensor, iid: bool = False) -> t.Tensor:
        return self(x, iid=iid).round()

    def from_data(
        acts: t.Tensor, labels: t.Tensor, device: str = "cpu", atol: float = 1e-3
    ) -> "MMProbe":
        acts, labels = acts.to(device), labels.to(device)
        pos_acts, neg_acts = acts[labels == 1], acts[labels == 0]
        pos_mean, neg_mean = pos_acts.mean(0), neg_acts.mean(0)
        cav = pos_mean - neg_mean

        centered_data = t.cat([pos_acts - pos_mean, neg_acts - neg_mean], 0)
        covariance = centered_data.t() @ centered_data / acts.shape[0]

        probe = MMProbe(cav, covariance=covariance).to(device)

        return probe
