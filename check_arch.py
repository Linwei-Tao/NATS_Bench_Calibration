import pandas as pd
import numpy as np
url='./Figure_data/cifar10_ecesum.csv'
df=pd.read_csv(url)
print(df.head())


# Import NATS_BENCH
from nats_bench import create
from xautodl.models import get_cell_based_tiny_net

# Create NATS_BENCH
api = create("/home/../../media/linwei/disk1/NATS-Bench/NATS-tss-v1_0-3ffb9-full/NATS-tss-v1_0-3ffb9-full", 'tss', fast_mode=True, verbose=True)
config = api.get_net_config(6111, "cifar10")
# get the info of architecture of the 6111-th model on CIFAR-10
api.query_by_arch(6111,'12')
print(config)