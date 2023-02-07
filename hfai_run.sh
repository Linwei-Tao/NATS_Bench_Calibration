#!/bin/bash
#cifar10_best8acc_idx = [13714, 6111, 9930, 3731, 5111, 81, 1459, 5292]
#cifar10_best8acc = [94.37333,94.37333,94.36333,94.34333,94.30667,94.30000,94.29667,94.29333]

python hfai_train.py --device=0 --arch_index=13714 --platform=hfai&
python hfai_train.py --device=1 --arch_index=6111 --platform=hfai&
python hfai_train.py --device=2 --arch_index=9930 --platform=hfai&
python hfai_train.py --device=3 --arch_index=3731 --platform=hfai&
python hfai_train.py --device=4 --arch_index=5111 --platform=hfai&
python hfai_train.py --device=5 --arch_index=81 --platform=hfai&
python hfai_train.py --device=6 --arch_index=1459 --platform=hfai&
python hfai_train.py --device=7 --arch_index=5292 --platform=hfai&
# hfai bash hfai_run.sh -- -n 1 --force --no_diff --name cifar10-best8acc-350
