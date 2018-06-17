import os
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
# https://github.com/ZijunDeng/pytorch-semantic-segmentation/blob/master/utils/misc.py
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss((1 - F.softmax(inputs)) ** self.gamma * F.log_softmax(inputs), targets)
        
if __name__ == '__main__':
    loss = FocalLoss()
    input = Variable(torch.randn(3, 5), requires_grad=True)
    target = Variable(torch.LongTensor(3).random_(5))
    output = loss(input, target)
    output.backward()

    print(input)
    print(target)
    print(output)