import math
import torch
import numpy as np
from torch import nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn import functional as F
import xautodl
from xautodl.datasets.DownsampledImageNet import ImageNet16
from xautodl.datasets.SearchDatasetWrap import SearchDataset
from xautodl.config_utils import load_config


def get_valid_test_loader(dataset, config_root, batch):
    data_dir = '/media/linwei/disk1/NATS-Bench/cifar.python'

    if dataset == "cifar10":
        normalize = transforms.Normalize(
            mean=[x / 255 for x in [125.3, 123.0, 113.9]],
            std=[x / 255 for x in [63.0, 62.1, 66.7]],
        )

        # define transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        data1 = datasets.CIFAR10(
            root=data_dir, train=True,
            download=False, transform=transform,
        )
        data2 = datasets.CIFAR10(
            root=data_dir, train=False,
            download=False, transform=transform,
        )

        cifar_split = load_config("{:}/cifar-split.txt".format(config_root), None, None)
        train_split, valid_split = (
            cifar_split.train,
            cifar_split.valid,
        )
        valid_loader = torch.utils.data.DataLoader(
            data1,
            batch_size=batch,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(valid_split),
            num_workers=1,

            pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            data2, batch_size=batch, shuffle=False,
            num_workers=1, pin_memory=True,
        )
    elif dataset == "cifar100":
        normalize = transforms.Normalize(
            mean=[x / 255 for x in [129.3, 124.1, 112.4]],
            std=[x / 255 for x in [68.2, 65.4, 70.4]],
        )

        # define transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        data = datasets.CIFAR100(
            root=data_dir, train=False,
            download=False, transform=transform,
        )
        cifar100_test_split = load_config(
            "{:}/cifar100-test-split.txt".format(config_root), None, None
        )
        valid_loader = torch.utils.data.DataLoader(
            data,
            batch_size=batch,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                cifar100_test_split.xvalid
            ),
            num_workers=1,
            pin_memory=True,
        )
        test_loader = torch.utils.data.DataLoader(
            data,
            batch_size=batch,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                cifar100_test_split.xtest
            ),
            num_workers=1,
            pin_memory=True,
        )

    else:
        normalize = transforms.Normalize(
            mean=[x / 255 for x in [122.68, 116.66, 104.01]],
            std=[x / 255 for x in [63.22, 61.26, 65.09]],
        )
        test_transform = transforms.Compose(
            [transforms.ToTensor(), normalize]
        )
        data = ImageNet16('/media/linwei/disk1/NATS-Bench/cifar.python/ImageNet16', False, test_transform, 120)
        imagenet_test_split = load_config(
            "{:}/imagenet-16-120-test-split.txt".format(config_root), None, None
        )
        valid_loader = torch.utils.data.DataLoader(
            data,
            batch_size=batch,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                imagenet_test_split.xvalid
            ),
            num_workers=1,
            pin_memory=True,
        )
        test_loader = torch.utils.data.DataLoader(
            data,
            batch_size=batch,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                imagenet_test_split.xtest
            ),
            num_workers=1,
            pin_memory=True,
        )

    return valid_loader, test_loader

def get_test_loader_cifar10(batch_size,
                    shuffle=True,
                    num_workers=1,
                    pin_memory=False):

    normalize = transforms.Normalize(
        mean=[x / 255 for x in [125.3, 123.0, 113.9]],
        std=[x / 255 for x in [63.0, 62.1, 66.7]],
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

def get_test_loader_cifar100(batch_size,
                    shuffle=True,
                    num_workers=1,
                    pin_memory=False):

    normalize = transforms.Normalize(
        mean=[x / 255 for x in [129.3, 124.1, 112.4]],
        std=[x / 255 for x in [68.2, 65.4, 70.4]],
    )

    # define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    data_dir = '/media/linwei/disk1/NATS-Bench/cifar.python'
    dataset = datasets.CIFAR100(
        root=data_dir, train=False,
        download=True, transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader

def get_test_loader_imagenet(batch_size,
                    shuffle=True,
                    num_workers=1,
                    pin_memory=False):
    normalize = transforms.Normalize(
        mean=[x / 255 for x in [122.68, 116.66, 104.01]],
        std=[x / 255 for x in [63.22, 61.26, 65.09]],
    )
    test_transform = transforms.Compose(
        [transforms.ToTensor(), normalize]
    )
    test_data = ImageNet16('/media/linwei/disk1/NATS-Bench/cifar.python/ImageNet16', False, test_transform , 120)
    data_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return data_loader



def get_logits_labels(data_loader, net):
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
        logits_list.append(logits[1])
        labels_list.append(label)
        '''
        logits = torch.cat(logits_list).cuda(0)
        labels = torch.cat(labels_list).cuda(0)
        '''
        logits = torch.cat(logits_list).cpu()
        labels = torch.cat(labels_list).cpu()
    return logits, labels
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

class ECELoss(nn.Module):
    '''
    Compute ECE (Expected Calibration Error)
    '''
    def __init__(self, n_bins=15):
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

class ClasswiseECELoss(nn.Module):
    '''
    Compute Classwise ECE
    '''
    def __init__(self, n_bins=15):
        super(ClasswiseECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        num_classes = int((torch.max(labels) + 1).item())
        softmaxes = F.softmax(logits, dim=1)
        per_class_sce = None

        for i in range(num_classes):
            class_confidences = softmaxes[:, i]
            class_sce = torch.zeros(1, device=logits.device)
            labels_in_class = labels.eq(i) # one-hot vector of all positions where the label belongs to the class i

            for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
                in_bin = class_confidences.gt(bin_lower.item()) * class_confidences.le(bin_upper.item())
                prop_in_bin = in_bin.float().mean()
                if prop_in_bin.item() > 0:
                    accuracy_in_bin = labels_in_class[in_bin].float().mean()
                    avg_confidence_in_bin = class_confidences[in_bin].mean()
                    class_sce += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            if (i == 0):
                per_class_sce = class_sce
            else:
                per_class_sce = torch.cat((per_class_sce, class_sce), dim=0)

        sce = torch.mean(per_class_sce)
        return sce

class AdaptiveECELoss(nn.Module):
    '''
    Compute Adaptive ECE
    '''
    def __init__(self, n_bins=15):
        super(AdaptiveECELoss, self).__init__()
        self.nbins = n_bins

    def histedges_equalN(self, x):
        npt = len(x)
        return np.interp(np.linspace(0, npt, self.nbins + 1),
                     np.arange(npt),
                     np.sort(x))
    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        n, bin_boundaries = np.histogram(confidences.cpu().detach(), self.histedges_equalN(confidences.cpu().detach()))
        #print(n,confidences,bin_boundaries)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

