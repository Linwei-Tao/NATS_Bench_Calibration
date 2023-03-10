
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class InverseFocalLoss(nn.Module):
    def __init__(self, gamma=0, size_average=False):
        super(InverseFocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input,-1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        loss = -1 * (1+pt)**self.gamma * logpt

        if self.size_average: return loss.mean()
        else:
            return loss.sum()