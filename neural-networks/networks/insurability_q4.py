import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(3, 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.linear_out = nn.Linear(2, 3, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.linear_out(x)
        x = self.softmax(x)
        return x


class MomentumOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9):
        super(MomentumOptimizer, self).__init__(params, defaults={"lr": lr})
        self.momentum = momentum
        self.state = dict()
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p] = dict(mom=torch.zeros_like(p.data))

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p not in self.state:
                    self.state[p] = dict(mom=torch.zeros_like(p.data))
                mom = self.state[p]["mom"]
                mom = self.momentum * mom - group["lr"] * p.grad.data
                p.data += mom
