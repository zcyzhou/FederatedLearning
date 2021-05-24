import torch
import torch.nn as nn
import torch.nn.functional as F


class MLR(nn.Module):
    """
    Multinomial logistic regression model
    """
    def __init__(self):
        super(MLR, self).__init__()
        self.fc1 = nn.Linear(784, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output
