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


class FocalLossHardnessClibrated(nn.Module):
    def __init__(self, gamma=0, lamda=0.01, size_average=False):
        super(FocalLossHardnessClibrated, self).__init__()
        self.gamma = gamma
        self.lamda = lamda
        self.size_average = size_average

    def forward(self, input, target):

        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)
        target_onehot = torch.squeeze(F.one_hot(target))
        num_class = torch.tensor(target_onehot.shape[-1])
        logpt = F.log_softmax(input, -1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()  # probility for target label
        po_mask = torch.ones_like(target_onehot).cuda() - target_onehot
        po = F.softmax(input, -1) * po_mask
        po_normlized = po / (1 - pt).reshape(-1, 1).tile((1, 10))
        po_entropy = po_normlized * torch.log(po_normlized)
        po_entropy[po_entropy != po_entropy] = 0

        entorpy_normalizer = torch.log(1 / (num_class - 1))

        hardness_calibrated_term = po_entropy.sum(-1) / entorpy_normalizer
        hardness_calibrated_term = hardness_calibrated_term.detach()
        loss = -1 * (1 - pt - self.lamda * hardness_calibrated_term) ** self.gamma * logpt
        if torch.isnan(loss.sum()):
            print('test')
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

