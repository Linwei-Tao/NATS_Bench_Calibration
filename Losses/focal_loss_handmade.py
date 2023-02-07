'''
Implementation of Focal Loss.
Reference:
[1]  T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, Focal loss for dense object detection.
     arXiv preprint arXiv:1708.02002, 2017.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# improved focal loss
class FocalLossHandMade1(nn.Module):
    # -(1-p_k+p_j)^\gamma * log(p_k)
    def __init__(self, gamma=0, size_average=False):
        super(FocalLossHandMade1, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logp_k = F.log_softmax(input, -1)

        softmax_logits = logp_k.exp()
        logp_k = logp_k.gather(1, target)
        logp_k = logp_k.view(-1)
        p_k = logp_k.exp()  # p_k: probility at target label
        p_j_mask = torch.lt(softmax_logits, p_k.reshape(p_k.shape[0], 1)) * 1  # mask all logit larger and equal than p_k
        p_j = torch.topk(p_j_mask * softmax_logits, 1)[0].squeeze()

        loss = -1 * (1 - p_k + p_j) ** self.gamma * logp_k
        focal_loss = -1 * (1 - p_k) ** self.gamma * logp_k

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

# focal loss adding a constant
class FocalLossHandMade2(nn.Module):
    # -(1-p_k+c)^\gamma * log(p_k)
    def __init__(self, gamma=0, size_average=False, c=None):
        super(FocalLossHandMade2, self).__init__()
        self.gamma = gamma
        self.c = c
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logp_k = F.log_softmax(input, -1)

        softmax_logits = logp_k.exp()
        logp_k = logp_k.gather(1, target)
        logp_k = logp_k.view(-1)
        p_k = logp_k.exp()  # p_k: probility at target label
        p_j_mask = torch.lt(softmax_logits, p_k.reshape(p_k.shape[0], 1)) * 1  # mask all logit larger and equal than p_k
        p_j = torch.topk(p_j_mask * softmax_logits, 1)[0].squeeze()

        loss = -1 * (1 - p_k + self.c) ** self.gamma * logp_k
        focal_loss = -1 * (1 - p_k) ** self.gamma * logp_k
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

# focal loss adding a next n_lower largest probility
class FocalLossHandMade3(nn.Module):
    # -(1-p_k+p_j)^\gamma * log(p_k)
    def __init__(self, gamma=0, size_average=False, n_lower=1):
        super(FocalLossHandMade3, self).__init__()
        self.gamma = gamma
        self.n_lower = n_lower
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logp_k = F.log_softmax(input, -1)

        softmax_logits = logp_k.exp()
        logp_k = logp_k.gather(1, target)
        logp_k = logp_k.view(-1)
        p_k = logp_k.exp()  # p_k: probility at target label
        p_j_mask = torch.lt(softmax_logits, p_k.reshape(p_k.shape[0], 1)) * 1  # mask all logit larger and equal than p_k
        p_j = torch.topk(p_j_mask * softmax_logits, self.n_lower)[0][:, self.n_lower-1].squeeze()

        loss = -1 * (1 - p_k + p_j) ** self.gamma * logp_k

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
