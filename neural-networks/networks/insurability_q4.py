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
