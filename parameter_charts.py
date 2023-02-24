from Net.resnet import resnet50, resnet110
from Net.wide_resnet import wide_resnet_cifar
from Net.densenet import densenet121

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

models = {
    'resnet50': resnet50(),
    'resnet110': resnet110(),
    'wide-resnet': wide_resnet_cifar(),
    'densenet121': densenet121()
}

param_list = {
    'resnet50': 0,
    'resnet110': 0,
    'wide-resnet': 0,
    'densenet121': 0
}
for name, model in models.items():

    count = 0

    for x in model.parameters():
        count += x.numel()

    param_list[name] = count

df = pd.DataFrame([param_list['resnet50'], param_list['resnet110'],
                  param_list['wide-resnet'], param_list['densenet121']])

ece_result = pd.read_csv('normal_model_result_2.csv')

cifar10_fl_list = []
cifar10_ce_list = []
cifar100_fl_list = []
cifar100_ce_list = []
for idx, (loss, dataset) in enumerate(zip(ece_result['loss_func'], ece_result['dataset'])):
    if dataset == 'cifar10':
        if loss == 'FL':
            cifar10_fl_list.append(idx)
        else:
            cifar10_ce_list.append(idx)

    else:
        if loss == 'FL':
            cifar100_fl_list.append(idx)
        else:
            cifar100_ce_list.append(idx)

total_list = [cifar10_fl_list, cifar10_ce_list, cifar100_fl_list, cifar100_ce_list]


cifar10_FL_df = ece_result.loc[cifar10_fl_list].sort_values(by="val_acc", inplace=False, ascending=True)
cifar10_CE_df = ece_result.loc[cifar10_ce_list].sort_values(by="val_acc", inplace=False, ascending=True)
cifar100_FL_df = ece_result.loc[cifar100_fl_list].sort_values(by="val_acc", inplace=False, ascending=True)
cifar100_CE_df = ece_result.loc[cifar100_ce_list].sort_values(by="val_acc", inplace=False, ascending=True)

total_df = [cifar10_FL_df, cifar10_CE_df, cifar100_FL_df, cifar100_CE_df]

cifar10_fl_dict = {'resnet50': {'val_acc': 0, 'pre_val_ece': 0, 'param': 0},
                   'wide-resnet': {'val_acc': 0, 'pre_val_ece': 0, 'param': 0},
                   'resnet110': {'val_acc': 0, 'pre_val_ece': 0, 'param': 0},
                   'densenet121': {'val_acc': 0, 'pre_val_ece': 0, 'param': 0},
                   }
cifar10_ce_dict = {'resnet50': {'val_acc': 0, 'pre_val_ece': 0, 'param': 0},
                   'wide-resnet': {'val_acc': 0, 'pre_val_ece': 0, 'param': 0},
                   'resnet110': {'val_acc': 0, 'pre_val_ece': 0, 'param': 0},
                   'densenet121': {'val_acc': 0, 'pre_val_ece': 0, 'param': 0},
                   }
cifar100_fl_dict = {'resnet50': {'val_acc': 0, 'pre_val_ece': 0, 'param': 0},
                    'wide-resnet': {'val_acc': 0, 'pre_val_ece': 0, 'param': 0},
                    'resnet110': {'val_acc': 0, 'pre_val_ece': 0, 'param': 0},
                    'densenet121': {'val_acc': 0, 'pre_val_ece': 0, 'param': 0},
                    }

cifar100_ce_dict = {'resnet50': {'val_acc': 0, 'pre_val_ece': 0, 'param': 0},
                    'wide-resnet': {'val_acc': 0, 'pre_val_ece': 0, 'param': 0},
                    'resnet110': {'val_acc': 0, 'pre_val_ece': 0, 'param': 0},
                    'densenet121': {'val_acc': 0, 'pre_val_ece': 0, 'param': 0},
                    }
total_dict = [cifar10_fl_dict, cifar10_ce_dict,
              cifar100_fl_dict, cifar100_ce_dict]

for curr_df, curr_list, curr_dict in zip(total_df, total_list, total_dict):
    for idx in curr_list:
        name, val_acc, pre_val_ece = curr_df.loc[idx]['model'], curr_df.loc[idx]['val_acc'], curr_df.loc[idx]['pre_val_ece']
        curr_dict[name]['val_acc'] = val_acc
        curr_dict[name]['pre_val_ece'] = pre_val_ece
        curr_dict[name]['param'] = param_list[name]

plt_title = ['CIFAR10-FL', 'CIFAR10-CE', 'CIFAR100-FL', 'CIFAR100-CE']

plt.figure(figsize=(16, 16/4), linewidth=0.75)

plt.subplots_adjust(left=0.11, bottom=0.25, right=0.96,
                    top=0.91, hspace=0.1, wspace=0.45)
plt.rc('font', family='Times New Roman')
matplotlib.rcParams.update({'font.size': 12})
colors = np.random.rand(len(plt_title))
for idx, dic in enumerate(total_dict):
    model_list = []
    val_acc_list = []
    pre_val_ece_list = []
    param_lists = []
    for model_name, values in dic.items():
        model_list.append(model_name)
        val_acc_list.append(values['val_acc']*100)
        pre_val_ece_list.append(values['pre_val_ece']*100)
        param_lists.append(values['param']*0.000005)
    ax1 = plt.subplot(1, 4, idx+1)
    ax1.set_xlabel("Accuracy")
    ax1.set_ylabel("pre_ECE")
    ax1.set_title(plt_title[idx])
    ax1.scatter(x=val_acc_list,
                y=pre_val_ece_list,
                s=param_lists,
                c=colors,
                alpha=0.6
                )
    for idx, (x, y) in enumerate(zip(val_acc_list, pre_val_ece_list)):
        ax1.text(x, y, model_list[idx], ha='center', va='bottom')

plt.show()