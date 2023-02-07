'''
Script for training models.
'''

import os
from datetime import datetime
import socket
import numpy as np
from torch import optim
from pathlib import Path
import torch
import torch.utils.data
import argparse
import torch.backends.cudnn as cudnn
import random
import wandb


# Import dataloaders
import data.cifar10 as cifar10
import data.cifar100 as cifar100
import data.tiny_imagenet as tiny_imagenet

# Import network models
from Net.resnet import resnet50, resnet110
from Net.resnet_tiny_imagenet import resnet50 as resnet50_ti
from Net.wide_resnet import wide_resnet_cifar
from Net.densenet import densenet121


# Import loss functions
from Losses.loss import cross_entropy, focal_loss, focal_loss_adaptive
from Losses.loss import mmce, mmce_weighted
from Losses.loss import brier_score

# Import train and validation utilities
from train_utils import train_single_epoch

# Import validation metrics
from Metrics.metrics import test_classification_net
from datetime import datetime

# Import NATS_BENCH
from nats_bench import create
from xautodl.models import get_cell_based_tiny_net

dataset_num_classes = {
    'cifar10': 10,
    'cifar100': 100,
    'tiny_imagenet': 200,
    'imagenet': 1000
}

dataset_loader = {
    'cifar10': cifar10,
    'cifar100': cifar100,
    'tiny_imagenet': tiny_imagenet
}

models = {
    'resnet50': resnet50,
    'resnet50_ti': resnet50_ti,
    'resnet110': resnet110,
    'wide_resnet': wide_resnet_cifar,
    'densenet121': densenet121
}


def parseArgs():
    default_dataset = 'cifar10'
    dataset_root = './datasets'
    train_batch_size = 128
    test_batch_size = 128
    learning_rate = 0.1
    momentum = 0.9
    optimiser = "sgd"
    loss = "cross_entropy"
    gamma = 1.0
    gamma2 = 1.0
    gamma3 = 1.0
    lamda = 0.1
    weight_decay = 5e-4
    log_interval = 50
    save_interval = 50
    save_loc = './run/'
    load_loc = './'
    model = "nats_bench"
    epoch = 350
    first_milestone = 150  # Milestone for change in lr
    second_milestone = 250  # Milestone for change in lr
    gamma_schedule_step1 = 100
    gamma_schedule_step2 = 250

    parser = argparse.ArgumentParser(
        description="Training for calibration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, default=default_dataset,
                        dest="dataset", help='dataset to train on')
    parser.add_argument("--dataset-root", type=str, default=dataset_root,
                        dest="dataset_root", help='root path of the dataset (for tiny imagenet)')
    parser.add_argument("--wandb_mode", type=str, default="disabled")
    parser.add_argument("--data-aug", action="store_true", dest="data_aug")
    parser.add_argument("--parallel", action="store_true", default=False)
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.set_defaults(data_aug=True)

    parser.add_argument("-g", action="store_true", dest="gpu",
                        help="Use GPU")
    parser.set_defaults(gpu=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--save_all", action="store_true")
    parser.set_defaults(save_all=False)
    parser.add_argument("--load", action="store_true")
    parser.set_defaults(load=False)
    parser.add_argument("-b", type=int, default=train_batch_size,
                        dest="train_batch_size", help="Batch size")
    parser.add_argument("-tb", type=int, default=test_batch_size,
                        dest="test_batch_size", help="Test Batch size")
    parser.add_argument("--epoch", type=int, default=epoch, dest="epoch",
                        help='Number of training epochs')
    parser.add_argument("--lr", type=float, default=learning_rate,
                        dest="learning_rate", help='Learning rate')
    parser.add_argument("--mom", type=float, default=momentum,
                        dest="momentum", help='Momentum')
    parser.add_argument("--nesterov", action="store_true", dest="nesterov",
                        help="Whether to use nesterov momentum in SGD")
    parser.set_defaults(nesterov=False)
    parser.add_argument("--decay", type=float, default=weight_decay,
                        dest="weight_decay", help="Weight Decay")
    parser.add_argument("--opt", type=str, default=optimiser,
                        dest="optimiser",
                        help='Choice of optimisation algorithm')

    # loss function
    parser.add_argument("--loss", type=str, default=loss, dest="loss_function",
                        help="Loss function to be used for training")
    parser.add_argument("--gamma", type=float, default=gamma,
                        dest="gamma", help="Gamma for focal components")
    parser.add_argument("--gamma2", type=float, default=gamma2,
                        dest="gamma2", help="Gamma for different focal components")
    parser.add_argument("--gamma3", type=float, default=gamma3,
                        dest="gamma3", help="Gamma for different focal components")
    parser.add_argument("--lamda", type=float, default=lamda,
                        dest="lamda", help="Regularization factor")
    parser.add_argument("--gamma-schedule", type=int, default=0,
                        dest="gamma_schedule", help="Schedule gamma or not")
    parser.add_argument("--gamma-schedule-step1", type=int, default=gamma_schedule_step1,
                        dest="gamma_schedule_step1", help="1st step for gamma schedule")
    parser.add_argument("--gamma-schedule-step2", type=int, default=gamma_schedule_step2,
                        dest="gamma_schedule_step2", help="2nd step for gamma schedule")

    # log
    parser.add_argument("--log-interval", type=int, default=log_interval,
                        dest="log_interval", help="Log Interval on Terminal")
    parser.add_argument("--save-interval", type=int, default=save_interval,
                        dest="save_interval", help="Save Interval on Terminal")
    parser.add_argument("--save-path", type=str, default=save_loc,
                        dest="save_loc",
                        help='Path to export the model')
    parser.add_argument("--load-path", type=str, default=load_loc,
                        dest="load_loc",
                        help='Path to load the model from')

    parser.add_argument("--model", type=str, default=model, dest="model",
                        help='Model to train')
    parser.add_argument("--first-milestone", type=int, default=first_milestone,
                        dest="first_milestone", help="First milestone to change lr")
    parser.add_argument("--second-milestone", type=int, default=second_milestone,
                        dest="second_milestone", help="Second milestone to change lr")

    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--platform", type=str, default="local")


    # NATS_BENCH
    parser.add_argument("--arch_index", type=int, default=0)
    parser.add_argument("--arch_load_epoch", type=str, default=None, help="values in 01 12 90 200")

    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgs()

    if args.platform=="local" or args.platform=="gadi":
        os.environ["MARSV2_NB_NAME"] = args.dataset+"-"+args.model+"-"+str(args.arch_index)+"-"+args.loss_function+"-"+str(1)
        args.save_loc = 'checkpoints/{}'.format(os.environ["MARSV2_NB_NAME"])
    elif args.platform=="hfai":
        args.save_loc = 'checkpoints/{}-{}'.format(os.environ["MARSV2_NB_NAME"], args.device)


    args.save_path = Path(args.save_loc)
    args.save_path.mkdir(exist_ok=True, parents=True)
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        if args.deterministic:
            print('deterministic')
            torch.backends.cudnn.deterministic = True


    setup_seed(args.seed)
    # set current device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    os.environ['WANDB_MODE'] = args.wandb_mode

    cuda = False
    if (torch.cuda.is_available() and args.gpu):
        cuda = True
    device = torch.device("cuda" if cuda else "cpu")
    print("CUDA set: " + str(cuda))
    print(args)

    config = args
    config.hostname = socket.gethostname()
    if args.dataset == 'tiny_imagenet':
        args.epoch = 100
        args.first_milestone = 40
        args.second_milestone = 60
        args.train_batch_size = 64
        args.dataset_root = "./datasets/tiny-imagenet-200"
    elif args.dataset == 'imagenet':
        args.epoch = 90
        args.learning_rate = 0.01
        args.first_milestone = 30
        args.second_milestone = 60

    wandb.init(project="improved_focal_loss for Calibration", entity="linweitao", config=config, id = "{}-{}".format(os.environ["MARSV2_NB_NAME"], args.device))

    num_classes = dataset_num_classes[args.dataset]

    # Choosing the model to train
    if args.model == "nats_bench":
        # use nats_bench
        if args.platform == "local":
            # api = create(r"/media/linwei/disk1/NATS-Bench/NATS-tss-v1_0-3ffb9-full/NATS-tss-v1_0-3ffb9-full", 'tss', fast_mode=True, verbose=True)
            api = create("NATS_Bench/NATS-tss-v1_0-3ffb9-simple", 'tss',
                         fast_mode=True, verbose=True)
        elif args.platform == "hfai":
            api = create("NATS_Bench/NATS-tss-v1_0-3ffb9-simple", 'tss',
                         fast_mode=True, verbose=True)
        config = api.get_net_config(args.arch_index, args.dataset)
        net = get_cell_based_tiny_net(config)
        if args.arch_load_epoch:
            params = api.get_net_param(args.arch_index, args.dataset, seed=777, hp=str(args.arch_load_epoch))
            net.load_state_dict(params)
    else:
        net = models[args.model](num_classes=num_classes)


    if args.gpu is True:
        net.cuda()

    start_epoch = 0
    num_epochs = args.epoch

    if args.optimiser == "sgd":
        opt_params = net.parameters()
        optimizer = optim.SGD(opt_params,
                              lr=args.learning_rate,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay,
                              nesterov=args.nesterov)
    elif args.optimiser == "adam":
        opt_params = net.parameters()
        optimizer = optim.Adam(opt_params,
                               lr=args.learning_rate,
                               weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.first_milestone, args.second_milestone],
                                               gamma=0.1)

    if (args.dataset == 'tiny_imagenet'):
        train_loader = dataset_loader[args.dataset].get_data_loader(
            root=args.dataset_root,
            split='train',
            batch_size=args.train_batch_size,
            pin_memory=args.gpu,
            data_dir=args.dataset_root,)

        val_loader = dataset_loader[args.dataset].get_data_loader(
            root=args.dataset_root,
            split='val',
            batch_size=args.test_batch_size,
            pin_memory=args.gpu,
            data_dir=args.dataset_root,)

        test_loader = dataset_loader[args.dataset].get_data_loader(
            root=args.dataset_root,
            split='val',
            batch_size=args.test_batch_size,
            pin_memory=args.gpu,
            data_dir=args.dataset_root)
    else:
        train_loader, val_loader = dataset_loader[args.dataset].get_train_valid_loader(
            batch_size=args.train_batch_size,
            augment=args.data_aug,
            random_seed=1,
            pin_memory=args.gpu,
            data_dir=args.dataset_root,
        )

        test_loader = dataset_loader[args.dataset].get_test_loader(
            batch_size=args.test_batch_size,
            pin_memory=args.gpu,
            data_dir=args.dataset_root,
        )


    if (args.save_path / 'latest.pt').exists():
        print("Loading from ", args.save_path)
        ckpt = torch.load(args.save_path / 'latest.pt', map_location='cpu')
        net.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch']
        print("*********** Successfully continue training form epoch ", start_epoch, "***********")


    for epoch in range(start_epoch, num_epochs):
        if (args.loss_function == 'focal_loss' and args.gamma_schedule == 1):
            if (epoch < args.gamma_schedule_step1):
                gamma = args.gamma
            elif (epoch >= args.gamma_schedule_step1 and epoch < args.gamma_schedule_step2):
                gamma = args.gamma2
            else:
                gamma = args.gamma3
        else:
            gamma = args.gamma

        epoch_start_time = datetime.now().replace(microsecond=0)
        train_loss = train_single_epoch(epoch,
                                        net,
                                        train_loader,
                                        optimizer,
                                        device,
                                        loss_function=args.loss_function,
                                        gamma=gamma,
                                        lamda=args.lamda)



        val_acc, pre_val_nll, pre_val_ece, test_acc, pre_test_ece, pre_test_adaece, pre_test_cece, pre_test_nll, T_opt, post_test_ece, post_test_adaece, \
        post_test_cece, post_test_nll = test_classification_net(net, test_loader, val_loader, device)

        print(f'[{datetime.now().strftime("%H:%M:%S")}] [{args.device}] Epoch {epoch}: Acc: {test_acc}, Pre_ECE: {pre_test_ece}, Post_ECE: {post_test_ece}')

        scheduler.step()


        wandb.log({
            "current epoch": epoch, "val_acc": val_acc, "pre_val_nll": pre_val_nll, "pre_val_ece": pre_val_ece,
            "test_acc": test_acc, "pre_test_ece": pre_test_ece, "pre_test_adaece": pre_test_adaece,
            "pre_test_cece": pre_test_cece, "pre_test_nll": pre_test_nll, "T_opt": T_opt,
            "post_test_ece": post_test_ece,
            "post_test_adaece": post_test_adaece, "post_test_cece": post_test_cece, "post_test_nll": post_test_nll
        })


        state = {
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch + 1,
        }
        torch.save(state, os.path.join(args.save_loc, 'latest.pt'))
        if args.save_all:
            torch.save(state, os.path.join(args.save_loc, '{}.pt'.format(epoch + 1)))
