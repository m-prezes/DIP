import torch as t


class LRProbe(t.nn.Module):
    def __init__(self, d_in: int) -> None:
        super().__init__()
        self.net = t.nn.Sequential(t.nn.Linear(d_in, 1, bias=False), t.nn.Sigmoid())

    def forward(self, x: t.Tensor, iid=None) -> t.Tensor:
        return self.net(x).squeeze(-1)

    def pred(self, x: t.Tensor, iid=None) -> t.Tensor:
        return self(x).round()

    def from_data(
        acts: t.Tensor,
        labels: t.Tensor,
        lr: float = 0.001,
        weight_decay: float = 0.1,
        epochs: int = 1000,
        device: str = "cpu",
    ) -> "LRProbe":
        acts, labels = acts.to(device), labels.to(device)
        probe = LRProbe(acts.shape[-1]).to(device)

        opt = t.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(epochs):
            opt.zero_grad()
            loss = t.nn.BCELoss()(probe(acts), labels)
            loss.backward()
            opt.step()

        return probe

    @property
    def cav(self) -> t.Tensor:
        return self.net[0].weight.data[0]
