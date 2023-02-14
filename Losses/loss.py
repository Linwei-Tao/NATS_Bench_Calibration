'''
Implementation of the following loss functions:
1. Cross Entropy
2. Focal Loss
3. Cross Entropy + MMCE_weighted
4. Cross Entropy + MMCE
5. Brier Score
'''

from torch.nn import functional as F
from Losses.focal_loss import FocalLoss
from Losses.inverse_focal_loss import InverseFocalLoss
from Losses.adafocal import AdaFocal
from Losses.focal_loss_handmade import FocalLossHandMade1, FocalLossHandMade2, FocalLossHandMade3
from Losses.focal_loss_hc import FocalLossHardnessClibrated
from Losses.focal_loss_adaptive_gamma import FocalLossAdaptive
from Losses.mmce import MMCE, MMCE_weighted
from Losses.ce_klece import KLECE
from Losses.brier_score import BrierScore


def cross_entropy(logits, targets, **kwargs):
    return F.cross_entropy(logits, targets, label_smoothing=kwargs['label_smoothing'], reduction='sum')


def focal_loss(logits, targets, **kwargs):
    return FocalLoss(gamma=kwargs['gamma'])(logits, targets)


def adafocal(logits, targets, **kwargs):
    return AdaFocal(gamma=kwargs['gamma'], device=kwargs['device'],
                    prev_epoch_adabin_dict=kwargs['prev_epoch_adabin_dict'], gamma_lambda=kwargs['gamma_lambda'],
                    adafocal_start_epoch=kwargs['adafocal_start_epoch'], epoch=kwargs['epoch'])(logits, targets)


def inverse_focal_loss(logits, targets, **kwargs):
    return InverseFocalLoss(gamma=kwargs['gamma'])(logits, targets)


def focal_loss_handmade1(logits, targets, **kwargs):
    return FocalLossHandMade1(gamma=kwargs['gamma'])(logits, targets)


def focal_loss_handmade2(logits, targets, **kwargs):
    return FocalLossHandMade2(gamma=kwargs['gamma'], c=kwargs['c'])(logits, targets)


def focal_loss_handmade3(logits, targets, **kwargs):
    return FocalLossHandMade3(gamma=kwargs['gamma'], n_lower=kwargs['n_lower'])(logits, targets)


def focal_loss_adaptive(logits, targets, **kwargs):
    return FocalLossAdaptive(gamma=kwargs['gamma'],
                             device=kwargs['device'])(logits, targets)


def focal_loss_hc(logits, targets, **kwargs):
    return FocalLossHardnessClibrated(gamma=kwargs['gamma'], lamda=kwargs['lamda'])(logits, targets)


def mmce(logits, targets, **kwargs):
    ce = F.cross_entropy(logits, targets)
    mmce = MMCE(kwargs['device'])(logits, targets)
    return ce + (kwargs['lamda'] * mmce)


def mmce_weighted(logits, targets, **kwargs):
    ce = F.cross_entropy(logits, targets)
    mmce = MMCE_weighted(kwargs['device'])(logits, targets)
    return ce + (kwargs['lamda'] * mmce)


def brier_score(logits, targets, **kwargs):
    return BrierScore()(logits, targets)


def ce_klece(logits, targets, **kwargs):
    ce = F.cross_entropy(logits, targets)
    klece = KLECE()(logits, targets)
    return ce + (kwargs['lamda'] * klece)
