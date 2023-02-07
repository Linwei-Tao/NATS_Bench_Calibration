import os

import numpy as np
import pandas as pd
import math
import torch
import numpy as np
from nats_bench import create
from torch import nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn import functional as F
import xautodl
from xautodl.datasets.DownsampledImageNet import ImageNet16
from xautodl.datasets.SearchDatasetWrap import SearchDataset
from xautodl.config_utils import load_config
import torch
import torchvision.models as M
from torch.optim.lr_scheduler import CosineAnnealingLR
from xautodl.models import get_cell_based_tiny_net

from datatest import get_valid_test_loader, ECELoss, AdaptiveECELoss, ClasswiseECELoss, get_logits_labels
from torchvision.models import resnet50, ResNet50_Weights
from torch import nn, optim
from torch.nn import functional as F

def get_train_loader(dataset,  batch):
    data_dir = '/media/linwei/disk1/NATS-Bench/cifar.python'

    if dataset == "cifar10":
        normalize = transforms.Normalize(
            mean=[x / 255 for x in [125.3, 123.0, 113.9]],
            std=[x / 255 for x in [63.0, 62.1, 66.7]],
        )

        # define transform
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
        ])
        data = datasets.CIFAR10(
            root=data_dir, train=True,
            download=False, transform=transform,
        )

        num_train = len(data)
        indices = list(range(num_train))
        split = int(np.floor(0.1 * num_train))

        np.random.seed(777)
        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(
            data, batch_size=batch, sampler=train_sampler,
            num_workers=1, pin_memory=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            data, batch_size=batch, sampler=valid_sampler,
            num_workers=1, pin_memory=True,
        )


    return train_loader, valid_loader

def get_test_loader(batch_size,
                    shuffle=True,
                    num_workers=1,
                    pin_memory=True):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    data_dir = '/media/linwei/disk1/NATS-Bench/cifar.python'
    dataset = datasets.CIFAR10(
        root=data_dir, train=False,
        download=False, transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader

def get_logits_labels2(data_loader, net):
    logits_list = []
    labels_list = []
    net.eval()
    #with torch.no_grad():
    for data, label in data_loader:
        data = data.cuda(0)
        #data = data.cpu()

        #
        #print(data)
        #print(data.shape)
        logits = net(data)
        #print(logits)
        #print(logits[1].shape)
        logits_list.append(logits)
        labels_list.append(label)
        '''
        logits = torch.cat(logits_list).cuda(0)
        labels = torch.cat(labels_list).cuda(0)
        '''
        logits = torch.cat(logits_list).cpu()
        labels = torch.cat(labels_list).cpu()
    return logits, labels

torch.manual_seed(777)
torch.cuda.manual_seed(777)
url='/media/linwei/disk1/NATS-Bench/350cifar10/'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
dset='cifar10'
#trainloader = get_train_loader(dset, '/media/linwei/disk1/NATS-Bench/cifar.python', batch=256)
testloader = get_test_loader(batch_size=128)
trainloader, valloader= get_train_loader(dset, batch=128)
#model = M.resnet50(pretrained=True)
'''
model = M.resnet50()
dim_in = model.fc.in_features
model.fc =nn.Linear(in_features=dim_in, out_features=10, bias=True)
model = model.cuda(0)
'''
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

starter.record()
idx=6111
api = create(r"/media/linwei/disk1/NATS-Bench/NATS-tss-v1_0-3ffb9-full/NATS-tss-v1_0-3ffb9-full", 'tss', fast_mode=True, verbose=True)
config = api.get_net_config(idx, dset)
model = get_cell_based_tiny_net(config)
params = api.get_net_param(idx, dset, seed=777, hp='200')
model.load_state_dict(params)
model = model.cuda(0)
epochs=150
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9, nesterov=True)
#scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.1)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

lossfunction =nn.CrossEntropyLoss(reduction='sum')

best_val_acc = 0
best_model_dict={}
ep_num=0

for epoch in range(epochs):

    print(epoch+200)
    for input, label in trainloader:
        model.train()
        input = input.cuda(0)
        label = label.cuda(0)
        logits = model(input)
        softmaxes = F.softmax(logits[1], dim=1)
        #print(logits)
        #loss = lossfunction(logits, label)
        loss = lossfunction(softmaxes, label)
        #print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 2)
        optimizer.step()
    scheduler.step()

    model.eval()
    with torch.no_grad():
        logits, labels = get_logits_labels(valloader, model)
        softmaxes = F.softmax(logits, dim=1)
        _, predictions = torch.max(softmaxes, 1)
        acc = predictions.eq(labels).float().mean().item()
            #print(acc)
        if acc > best_val_acc:
            best_model_dict = model.state_dict()
            ep_num=epoch
            best_val_acc=acc
        print("valacc:{}".format(acc))

        logits, labels = get_logits_labels(testloader, model)
        softmaxes = F.softmax(logits, dim=1)
        _, predictions = torch.max(softmaxes, 1)
        acc = predictions.eq(labels).float().mean().item()
        # print(acc)

        print("testacc:{}".format(acc))
    '''
    if epoch%10==9:
        print(epoch)
        model.eval()
        with torch.no_grad():
            logits, labels = get_logits_labels2(valloader, model)
            softmaxes = F.softmax(logits, dim=1)
            #_, predictions = torch.max(logits, 1)
            _, predictions = torch.max(softmaxes, 1)
            acc = predictions.eq(labels).float().mean().item()
            print(acc)
    '''

modelurl = url+'idx'+str(idx)+'+epoch'+str(ep_num)+'-200to350beta.model'

torch.save(best_model_dict, modelurl)

ender.record()

model.load_state_dict(torch.load(modelurl))
model = model.cuda(0)
model.eval()
with torch.no_grad():
    info=[]
    logits, labels = get_logits_labels(testloader, model)
    softmaxes = F.softmax(logits, dim=1)
    _, predictions = torch.max(softmaxes, 1)
    acc= predictions.eq(labels).float().mean()
    info.append(acc)
    ece_criterion = ECELoss().cuda(0)
    ece = ece_criterion(logits, labels).item()

    info.append(ece)


print(starter.elapsed_time(ender))
print(info)

