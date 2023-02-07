import os
import sys
import numpy as np
cwd = os.getcwd()
module_path = "/".join(cwd.split('/')[0:-1])
if module_path not in sys.path:
    sys.path.append(module_path)
import torch
import torch.nn as nn
import utils
import random
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn

# Import dataloaders
import data.cifar10 as cifar10
import data.cifar10_c as cifar10_c
import data.svhn as svhn
from temperature_scaling import ModelWithTemperature



# Import network models
from module.resnet import resnet50, resnet110
from module.resnet_tiny_imagenet import resnet50 as resnet50_ti
from module.wide_resnet import wide_resnet_cifar
from module.densenet import densenet121

# Import metrics to compute
from metrics.ood_test_utils import get_roc_auc

# Import plot related libraries
import seaborn as sb
import matplotlib.pyplot as plt

# Dataset params
dataset_num_classes = {
    'cifar10': 10,
    'svhn': 10
}

dataset_loader = {
    'cifar10': cifar10,
    'svhn': svhn,
    'cifar10_c': cifar10_c
}

# Mapping model name to model function
models = {
    'resnet50': resnet50,
    'resnet110': resnet110,
    'wide_resnet': wide_resnet_cifar,
    'densenet121': densenet121,
}

# Checking if GPU is available
cuda = False
if (torch.cuda.is_available()):
    cuda = True

# Setting additional parameters
torch.manual_seed(1)
device = torch.device("cuda" if cuda else "cpu")

def model_train(train_queue, model, criterion, optimizer):
    # set model to training model
    model.train()
    # create metrics
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    # training loop
    total_steps = len(train_queue)
    for step, (x, target) in enumerate(train_queue):
        n = x.size(0)
        # data to CUDA
        x = x.to('cuda').requires_grad_(False)
        target = target.to('cuda', non_blocking=True).requires_grad_(False)
        # update model weight
        # forward
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, target)
        # backward
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        # update metrics
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)
    # return average metrics
    return objs.avg, top1.avg, top5.avg

class args:
    data_aug = True
    gpu = device == "cuda"
    train_batch_size = 128
    test_batch_size = 128


dataset = 'cifar10'
ood_dataset = 'svhn'


num_classes = dataset_num_classes[dataset]
train_loader, val_loader = dataset_loader[dataset].get_train_valid_loader(
    batch_size=args.train_batch_size,
    augment=args.data_aug,
    random_seed=1,
    pin_memory=args.gpu,
    data_dir='../../datasets'
)

test_loader = dataset_loader[dataset].get_test_loader(
    batch_size=args.test_batch_size,
    pin_memory=args.gpu,
    data_dir='../../datasets'
)


ood_test_loader = dataset_loader[ood_dataset].get_test_loader(
    batch_size=args.test_batch_size,
    pin_memory=args.gpu,
)



# Taking input for the model
model_name_suffix = [
                        'brier_score_350.model',
                        'cross_entropy_350.model',
                        'cross_entropy_smoothed_smoothing_0.05_350.model',
                        'focal_loss_adaptive_53_350.model',
                        'focal_loss_gamma_3.0_350.model',
                        'mmce_weighted_lamda_2.0_350.model',
                     ]
model_name = 'densenet121'

for suffix in model_name_suffix:
    saved_model_name = '{}_{}'.format(model_name, suffix)

    model = models[model_name]

    net = model(num_classes=num_classes, temp=1.0)
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
    net.load_state_dict(torch.load('../../weights/cifar10/focal_calibration/' + str(saved_model_name)))

    (fpr_entropy, tpr_entropy, thresholds_entropy), (fpr_confidence, tpr_confidence, thresholds_confidence), pre_T_auc_entropy, auc_confidence = get_roc_auc(net, test_loader, ood_test_loader, device)

    scaled_model = ModelWithTemperature(net, False)
    scaled_model.set_temperature(val_loader, cross_validate='ece')
    (fpr_entropy, tpr_entropy, thresholds_entropy), (fpr_confidence, tpr_confidence, thresholds_confidence), post_T_auc_entropy, auc_confidence = get_roc_auc(scaled_model, test_loader, ood_test_loader, device)

    clrs = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    plt.figure()
    plt.rcParams["figure.figsize"] = (10, 8)
    sb.set_style('whitegrid')
    plt.plot(fpr_entropy, tpr_entropy, color=clrs[0], linewidth=5, label='ROC')

    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.xlabel('FPR', fontsize=30)
    plt.ylabel('TPR', fontsize=30)
    plt.legend(fontsize=28)

    # plt.show()
    print('Model Name: ' + str(model_name))
    print('OOD dataset: ' + str(ood_dataset))
    print('saved_model_name: ' + str(saved_model_name))
    print('Pre T AUROC entropy: ' + str(pre_T_auc_entropy))
    print('Post T AUROC entropy: ' + str(post_T_auc_entropy))
    print('_________')