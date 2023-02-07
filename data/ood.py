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
model_name = 'wide_resnet'
combination = torch.tensor([301, 277, 151, 323])
ftlr = 1e-4


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


model = models[model_name]

weight_root = "../../weights/{}/{}".format(dataset, model_name)
net = model(num_classes=num_classes, temp=1.0, weight_root=weight_root)
net.ops = np.array(list(range(350)))
net.cuda()
cudnn.benchmark = True
net.load_combination(combination)
criterion = nn.CrossEntropyLoss(reduction='sum')
criterion = criterion.to('cuda')
# use SGD to optimize the model (optimize model.parameters())
optimizer = torch.optim.SGD(
    net.parameters(),
    lr=ftlr,
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=False
)

model_train(train_loader, net, criterion, optimizer)

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
print('Pre T AUROC entropy: ' + str(pre_T_auc_entropy))
print('Post T AUROC entropy: ' + str(post_T_auc_entropy))


