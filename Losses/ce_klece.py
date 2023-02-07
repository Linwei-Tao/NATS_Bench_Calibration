'''
Implementation of the classwise calibration regularization term.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import namedtuple, deque


class KLECE(nn.Module):
    """
    Computes KLECE loss.
    """
    def __init__(self, n_bins=15, n_classes=10, confidence_memory_size=50000):
        super(KLECE, self).__init__()
        self.n_bins = n_bins
        self.n_classes = n_classes
        self.confidence_memory_size = confidence_memory_size
        self.confidence_table = torch.zeros(size=(n_classes, n_bins))
        self.confidence_memory = deque(maxlen=confidence_memory_size)
        self.label_memory = deque(maxlen=confidence_memory_size)
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]


    def get_ground_truth_confidence(self, logits, table):
        # probs = F.softmax(torch.randn(256,10))
        # table = torch.randn(10, 15)
        bin_ind = torch.div(logits, (1.0 / 15), rounding_mode='floor').to(torch.int32)
        ground_truth_confidence = torch.zeros_like(logits)
        for i in range(logits.shape[0]):
            for k in range(logits.shape[1]):
                ground_truth_confidence[i][k] = table[k][bin_ind[i][k]]
        return ground_truth_confidence

    def update_confidence_table(self, logits, labels):
        self.confidence_memory.append(logits)
        self.label_memory.append(F.one_hot(labels))
        softmaxes = torch.cat(tuple(self.confidence_memory))
        labels = torch.cat(tuple(self.label_memory))

        num_classes = int((torch.max(labels) + 1).item())

        for i in range(num_classes):
            class_confidences = softmaxes[:, i]
            labels_in_class = labels.eq(i)  # one-hot vector of all positions where the label belongs to the class i

            bin_index = 0
            for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
                in_bin = class_confidences.gt(bin_lower.item()) * class_confidences.le(bin_upper.item())
                prop_in_bin = in_bin.float().mean()

                if prop_in_bin.item() > 0:
                    accuracy_in_bin = labels_in_class[in_bin].float().mean()
                    # avg_confidence_in_bin = class_confidences[in_bin].mean()
                    # class_sce += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                    self.confidence_table[i][bin_index] = accuracy_in_bin

                bin_index = bin_index + 1

        return self.confidence_table



    def forward(self, input, target):
        batch_size = input.shape[0]
        predicted_logits = F.softmax(input, dim=1)
        self.update_confidence_table(predicted_logits, target)
        ground_truth_confidence = self.get_ground_truth_confidence(predicted_logits, self.confidence_table)

        # to-do: can give each sample a weight
        klece = nn.MSELoss()(predicted_logits, ground_truth_confidence)
        return klece
