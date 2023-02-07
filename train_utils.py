'''
This module contains methods for training models with different loss functions.
'''

import torch
from torch.nn import functional as F
from torch import nn
import os

from Losses.loss import cross_entropy, focal_loss, focal_loss_adaptive
from Losses.loss import mmce, mmce_weighted
from Losses.loss import brier_score
from Losses.loss import ce_klece
from Losses.loss import focal_loss_hc
from Losses.loss import focal_loss_handmade1, focal_loss_handmade2, focal_loss_handmade3
from Losses.loss import inverse_focal_loss


loss_function_dict = {
    'cross_entropy': cross_entropy,
    'focal_loss': focal_loss,
    'focal_loss_adaptive': focal_loss_adaptive,
    'mmce': mmce,
    'mmce_weighted': mmce_weighted,
    'brier_score': brier_score,
    'ce_klece': ce_klece,
    'focal_loss_hc': focal_loss_hc,
    'focal_loss_handmade1': focal_loss_handmade1,
    'focal_loss_handmade2': focal_loss_handmade2,
    'focal_loss_handmade3': focal_loss_handmade3,
    "inverse_focal_loss": inverse_focal_loss,
}


def train_single_epoch(epoch,
                       model,
                       train_loader,
                       optimizer,
                       device,
                       loss_function='cross_entropy',
                       gamma=1.0,
                       lamda=1.0,
                       label_smoothing=0.0,
                       start_step=0,
                       scheduler=None,
                       args=None,):
    '''
    Util method for training a model for a single epoch.
    '''
    model.train()
    train_loss = 0
    num_samples = 0
    for step, (data, labels) in enumerate(train_loader):
        if step < start_step:
            continue
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        (feature, logits) = model(data)
        if ('mmce' in loss_function):
            loss = (len(data) * loss_function_dict[loss_function](logits, labels, gamma=gamma, lamda=lamda, device=device))
        elif ('ce_klece' in loss_function):
            loss = loss_function_dict[loss_function](logits, labels, gamma=gamma, lamda=lamda, device=device)
        else:
            loss = loss_function_dict[loss_function](logits, labels, gamma=gamma, lamda=lamda, label_smoothing=label_smoothing, device=device)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        train_loss += loss.item()
        optimizer.step()
        num_samples += len(data)
        if scheduler is not None:
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'step': step + 1
            }
            torch.save(state, os.path.join(args.save_loc, 'latest.pt'))
            print("Epoch {}/{} Step {}/{}: Loss: {:.4f}".format(epoch, args.epoch, step + 1, len(train_loader),
                                                                 loss.item()))

    return train_loss / (num_samples+1e-8)



def test_single_epoch(epoch,
                      model,
                      test_val_loader,
                      device,
                      loss_function='cross_entropy',
                      gamma=1.0,
                      lamda=1.0):
    '''
    Util method for testing a model for a single epoch.
    '''
    model.eval()
    loss = 0
    num_samples = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_val_loader):
            data = data.to(device)
            labels = labels.to(device)

            logits = model(data)
            if ('mmce' in loss_function):
                loss += (len(data) * loss_function_dict[loss_function](logits, labels, gamma=gamma, lamda=lamda, device=device).item())
            else:
                loss += loss_function_dict[loss_function](logits, labels, gamma=gamma, lamda=lamda, device=device).item()
            num_samples += len(data)

    print('======> Test set loss: {:.4f}'.format(
        loss / num_samples))
    return loss / num_samples