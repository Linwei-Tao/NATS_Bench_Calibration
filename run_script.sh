# nohup /home/jinyu/miniconda3/envs/natscalibration/bin/python /home/jinyu/NATS_Bench_Calibration/train.py --arch_index=6111 --device=1 > ./runlog/train_6111_9_Feb_2023.log  &
# nohup /home/jinyu/miniconda3/envs/natscalibration/bin/python /home/jinyu/NATS_Bench_Calibration/nats_bench_train_tss_my.py --mode=specific-resnet50 --datasets=cifar10 > ./runlog/train_nats_10_Feb_2023_2.log  &

# Train the single model with Nats Setting
# --mode=specific-resnet50, just modify resnet50, such as --mode=specific-6111
# The result runlog will output at ./runlog/
nohup python nats_bench_train_tss.py --mode=specific-resnet50 --datasets=cifar10 > ./runlog/train_nats_10_Feb_2023_2.log  &