import torch
import torch.nn.functional as F
from sklearn import metrics
import os
import pandas as pd
import numpy as np
from nats_bench import create
import xautodl
from xautodl.models import get_cell_based_tiny_net
from datatest import get_logits_labels, get_valid_test_loader, ECELoss, AdaptiveECELoss, ClasswiseECELoss, \
    get_logits_labels2
import sys
sys.path.append('/home/haolan/NATS_Bench_Calibration')

from temprature import ModelWithTemperature

import data.cifar10 as cifar10
import data.cifar10_c as cifar10_c
import data.svhn as svhn
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from Metrics.ood_test_utils import get_roc_auc

# Checking if GPU is available
cuda = False
if (torch.cuda.is_available()):
    cuda = True
device = torch.device("cuda" if cuda else "cpu")
print(device)

def confidence(net_output):
    p = F.softmax(net_output, dim=1)
    confidence, _ = torch.max(p, dim=1)
    return confidence

def entropy(net_output):
    p = F.softmax(net_output, dim=1)
    # print(p.shape)
    logp = F.log_softmax(net_output, dim=1)
    plogp = p * logp
    entropy = - torch.sum(plogp, dim=1)
    return entropy


def get_roc_auc(net, test_loader, ood_test_loader, device, temped):
    bin_labels_entropies = None
    bin_labels_confidences = None
    entropies = None
    confidences = None

    net.eval()
    with torch.no_grad():
        # Getting entropies for in-distribution data
        for i, (data, label) in enumerate(test_loader):
            data = data.to(device)
            label = label.to(device)

            bin_label_entropy = torch.zeros(label.shape).to(device)
            bin_label_confidence = torch.ones(label.shape).to(device)

            net_output = net(data)
            # print(len(net_output))

            if temped:
                entrop = entropy(net_output)
                conf = confidence(net_output)
                # print(entrop, conf)
            else:
                entrop = entropy(net_output[1])
                conf = confidence(net_output[1])


            if (i == 0):
                bin_labels_entropies = bin_label_entropy
                bin_labels_confidences = bin_label_confidence
                entropies = entrop
                confidences = conf
            else:
                bin_labels_entropies = torch.cat((bin_labels_entropies, bin_label_entropy))
                bin_labels_confidences = torch.cat((bin_labels_confidences, bin_label_confidence))
                entropies = torch.cat((entropies, entrop))
                confidences = torch.cat((confidences, conf))

        # Getting entropies for OOD data
        for i, (data, label) in enumerate(ood_test_loader):
            data = data.to(device)
            label = label.to(device)

            bin_label_entropy = torch.ones(label.shape).to(device)
            bin_label_confidence = torch.zeros(label.shape).to(device)

            net_output = net(data)

            if temped:
                entrop = entropy(net_output)
                conf = confidence(net_output)
                # print(entrop, conf)
            else:
                entrop = entropy(net_output[1])
                conf = confidence(net_output[1])


            bin_labels_entropies = torch.cat((bin_labels_entropies, bin_label_entropy))
            bin_labels_confidences = torch.cat((bin_labels_confidences, bin_label_confidence))
            entropies = torch.cat((entropies, entrop))
            confidences = torch.cat((confidences, conf))

    fpr_entropy, tpr_entropy, thresholds_entropy = metrics.roc_curve(bin_labels_entropies.cpu().numpy(), entropies.cpu().numpy())
    fpr_confidence, tpr_confidence, thresholds_confidence = metrics.roc_curve(bin_labels_confidences.cpu().numpy(), confidences.cpu().numpy())
    auc_entropy = metrics.roc_auc_score(bin_labels_entropies.cpu().numpy(), entropies.cpu().numpy())
    auc_confidence = metrics.roc_auc_score(bin_labels_confidences.cpu().numpy(), confidences.cpu().numpy())

    return (fpr_entropy, tpr_entropy, thresholds_entropy), (fpr_confidence, tpr_confidence, thresholds_confidence), auc_entropy, auc_confidence




def get_train_loader(dataset="cifar10",  batch=128):
    data_dir = '/home/../../media/linwei/disk1/NATS-Bench/cifar.python'

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
        #print("num_train:{}".format(num_train))
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

class args:
    data_aug = True
    gpu = device == "cuda"
    train_batch_size = 128
    test_batch_size = 128

dataset_loader = {
    'cifar10': cifar10,
    'svhn': svhn,
    'cifar10_c': cifar10_c
}

dataset = 'cifar10'
ood_dataset = 'cifar10_c'


test_loader = dataset_loader[dataset].get_test_loader(
    batch_size=args.test_batch_size,
    pin_memory=args.gpu,
    data_dir='/home/../../data'
)


ood_test_loader_cifar10_c = dataset_loader['cifar10_c'].get_test_loader(
    batch_size=args.test_batch_size,
    pin_memory=args.gpu
)

ood_test_loader_svhn = dataset_loader['svhn'].get_test_loader(
    batch_size=args.test_batch_size,
    pin_memory=args.gpu
)


for dset in ['cifar10']:
    #for dset in ['ImageNet16-120']:
        # valloader, testloader = get_valid_test_loader(dset, '/home/../../media/linwei/disk1/NATS-Bench/cifar.python', batch=256)
        _, valloader = get_train_loader()
        idx_list = []
        acc_list = []
        # ece_bef_list = []
        # ece_aft_list = []
        # aece_bef_list = []
        # aece_aft_list = []
        # cece_bef_list = []
        # cece_aft_list = []
        pre_auc_cifar10 = []
        pre_auc_svhn = []
        post_auc_cifar10 = []
        post_auc_svhn = []
        #for idx in range(15625):
        #for idx in range(100):

        for idx in range(15625):
            api = create(r"/home/../../media/linwei/disk1/NATS-Bench/NATS-tss-v1_0-3ffb9-full/NATS-tss-v1_0-3ffb9-full", 'tss',
                         fast_mode=True, verbose=True)

            idx_list.append(idx)
            testacc = api.get_more_info(idx, dset, hp='200', is_random=False)["test-accuracy"]
            acc_list.append(testacc)

            config = api.get_net_config(idx, dset)
            network = get_cell_based_tiny_net(config)
            params = api.get_net_param(idx, dset, None, hp='200')
            network.load_state_dict(next(iter(params.values())))

            with torch.no_grad():
                net1 = network.to('cuda')
                # logits, labels = get_logits_labels(test_loader, net1)
                # x=[]
                # y=[]
                # z=[]
                # for bins in [5,10,15,20]:
                #     ece_criterion = ECELoss(n_bins=bins).cuda(1)
                #     ece = ece_criterion(logits, labels).item()
                #     x.append(ece)
                #     aece_criterion = AdaptiveECELoss(n_bins=bins).cuda(1)
                #     aece = aece_criterion(logits, labels).item()
                #     y.append(aece)
                #     cece_criterion = ClasswiseECELoss(n_bins=bins).cuda(1)
                #     cece = cece_criterion(logits, labels).item()
                #     z.append(cece)
                # ece_bef_list.append(x)
                # aece_bef_list.append(y)
                # cece_bef_list.append(z)
                pre_auc_cifar10.append(get_roc_auc(net1, test_loader, ood_test_loader_cifar10_c, 0, False)[-2])
                pre_auc_svhn.append(get_roc_auc(net1, test_loader, ood_test_loader_svhn, 0, False)[-2])
                print(pre_auc_cifar10, pre_auc_svhn)

            scaled_model = ModelWithTemperature(network)
            scaled_model.set_temperature(valloader)

            with torch.no_grad():
                net2 = scaled_model.to('cuda')
                # logits, labels = get_logits_labels2(test_loader, net2)
                # p = []
                # q = []
                # r = []
                # bin_temp = []
                # bin_temp.append(net2.get_temp())
                # for bins in [5, 10, 15, 20]:
                #     ece_criterion = ECELoss(n_bins=bins).cuda(1)
                #     ece = ece_criterion(logits, labels).item()
                #     p.append(ece)
                #
                #     aece_criterion = AdaptiveECELoss(n_bins=bins).cuda(1)
                #     aece = aece_criterion(logits, labels).item()
                #     q.append(aece)
                #
                #     cece_criterion = ClasswiseECELoss(n_bins=bins).cuda(1)
                #     cece = cece_criterion(logits, labels).item()
                #     r.append(cece)
                #
                #
                # ece_aft_list.append(p)
                # aece_aft_list.append(q)
                # cece_aft_list.append(r)

                post_auc_cifar10.append(get_roc_auc(net2, test_loader, ood_test_loader_cifar10_c, 0, True)[-2])
                # bin_temp.append(net2.get_temp())
                post_auc_svhn.append(get_roc_auc(net2, test_loader, ood_test_loader_svhn, 0, True)[-2])
                # bin_temp.append(net2.get_temp())
                print(post_auc_cifar10, post_auc_svhn)
                # print(bin_temp)




            if (idx % 200 == 199) | (idx == 15624):

                # ece_bef_list = np.array(ece_bef_list)
                # ece_aft_list = np.array(ece_aft_list)
                # aece_bef_list = np.array(aece_bef_list)
                # aece_aft_list = np.array(aece_aft_list)
                # cece_bef_list = np.array(cece_bef_list)
                # cece_aft_list = np.array(cece_aft_list)
                df = pd.DataFrame(zip(pre_auc_cifar10, post_auc_cifar10, pre_auc_svhn, post_auc_svhn))

                idx_list = []
                acc_list = []
                # ece_bef_list = []
                # ece_aft_list = []
                # aece_bef_list = []
                # aece_aft_list = []
                # cece_bef_list = []
                # cece_aft_list = []
                pre_auc_cifar10 = []
                pre_auc_svhn = []
                post_auc_cifar10 = []
                post_auc_svhn = []
                # df.to_csv('/media/linwei/disk1/NATS-Bench/NATS-details/'+dset+'/to'+str(idx)+'.csv')
                path = '/home/haolan/NATS_Bench_Calibration/summary/' + dset + '/to' + str(idx) + '.csv'
                print(path)
                df.to_csv(path)


