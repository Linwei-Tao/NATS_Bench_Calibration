import argparse
import wandb
import os
from pathlib import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils
from nats_bench import create

from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision
from nats_bench import create
from xautodl.models import get_cell_based_tiny_net
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch import nn, optim
from torch.nn import functional as F
from datatest import get_valid_test_loader, ECELoss, AdaptiveECELoss, ClasswiseECELoss, get_logits_labels

# hfai
#import hfai
#import hfai_env

#hfai_env.set_env('lfs')

# retrain
# Import dataloaders

def get_train_loader(dataset,  batch):
    data_dir = '/data'

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
        print("num_train:{}".format(num_train))
        indices = list(range(num_train))
        split = int(np.floor(0.1 * num_train))
        #split = 0

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
        mean=[x / 255 for x in [125.3, 123.0, 113.9]],
        std=[x / 255 for x in [63.0, 62.1, 66.7]],
    )

    # define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    data_dir = '/data'
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

def cross_entropy(logits, targets, **kwargs):
    return F.cross_entropy(logits, targets, reduction='sum')


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Calibration Dataset Test")
    parser.add_argument('--structure', type=int, default=6111)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--scheduler', type=str, default='cos')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--seed', type=int, default=777, help='random seed')
    parser.add_argument('--wandb_mode', type=str, default="offline")  # wandb

    args, unknown_args = parser.parse_known_args()
    args.wandb_dir = "./wandb_logs"
    Path(args.wandb_dir).mkdir(parents=True, exist_ok=True)


    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cudnn.enabled = True
    np.random.seed(args.seed)  # set random seed: numpy
    torch.manual_seed(args.seed)  # set random seed: torch
    torch.cuda.manual_seed(args.seed)  # set random seed: torch.cuda

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    os.environ['WANDB_MODE'] = args.wandb_mode
    #932e9085b298a0639cf29bc1ea769e970af390ba
    wandb.login(key="932e9085b298a0639cf29bc1ea769e970af390ba")
    args.wandb_dir = "./wandb"
    Path(args.wandb_dir).mkdir(parents=True, exist_ok=True)
    wandb.init(project="Calibration Dataset", entity="nuke", config=args, id="{}-{}-{}-{}".format(args.structure, args.scheduler, args.seed, args.device), dir=args.wandb_dir)

    print("wandb.run.dir", wandb.run.dir)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cudnn.enabled = True
    np.random.seed(args.seed)  # set random seed: numpy
    torch.manual_seed(args.seed)  # set random seed: torch
    torch.cuda.manual_seed(args.seed)  # set random seed: torch.cuda

    dset='cifar10'
    testloader = get_test_loader(batch_size=args.batch_size)
    trainloader, valloader = get_train_loader(dset, batch=args.batch_size)

    api = create("NATS-tss-v1_0-3ffb9-simple", 'tss',
                 fast_mode=True, verbose=True)
    config = api.get_net_config(args.structure, dset)
    model = get_cell_based_tiny_net(config)
    model = model.cuda(0)

    epochs = 350
    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0005, momentum=0.9, nesterov=True)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs*len(trainloader))
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)

    best_val_acc = 0
    best_model_dict = {}
    ep_num = 0

    for epoch in range(epochs):



        if args.scheduler == "cos":
            for i, (input, label) in enumerate(trainloader):
                model.train()

                input = input.cuda(0)
                label = label.cuda(0)
                optimizer.zero_grad()

                logits = model(input)

                loss = cross_entropy(logits[1], label)

                loss.backward()

                #torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
                optimizer.step()
                scheduler.step()
        else:
            for i, (input, label) in enumerate(trainloader):
                model.train()

                input = input.cuda(0)
                label = label.cuda(0)
                optimizer.zero_grad()

                logits = model(input)

                loss = cross_entropy(logits[1], label)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
                optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            logits, labels = get_logits_labels(valloader, model)
            softmaxes = F.softmax(logits, dim=1)
            _, predictions = torch.max(softmaxes, 1)
            val_acc = predictions.eq(labels).float().mean().item()
            # print(acc)
            if val_acc > best_val_acc:
                best_model_dict = model.state_dict()
                ep_num = epoch
                best_val_acc = val_acc


            logits, labels = get_logits_labels(testloader, model)
            softmaxes = F.softmax(logits, dim=1)
            _, predictions = torch.max(softmaxes, 1)
            test_acc = predictions.eq(labels).float().mean().item()


        wandb.log({
            "val_acc": val_acc*100, "test_acc": test_acc*100
        }, step=epoch)

    model1 = get_cell_based_tiny_net(config)
    model1 = model1.cuda(0)
    model1.load_state_dict(best_model_dict)
    model1.eval()

    with torch.no_grad():

        logits, labels = get_logits_labels(testloader, model1)
        softmaxes = F.softmax(logits, dim=1)
        _, predictions = torch.max(softmaxes, 1)
        best_test_acc = predictions.eq(labels).float().mean()

        ece_criterion = ECELoss().cuda(0)
        ece = ece_criterion(logits, labels).item()

        aece_criterion = AdaptiveECELoss().cuda(0)
        aece = aece_criterion(logits, labels).item()

        cece_criterion = ClasswiseECELoss().cuda(0)
        cece = cece_criterion(logits, labels).item()

    wandb.log({
        "best_test_acc": best_test_acc*100,
        "ece": ece*100,
        "aece": aece*100,
        "cece": cece*100
    })


