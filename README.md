# Natsdataset  、
## 模型的metrics信息可以用get_info直接读出来而不用加载权重。读取时注意seed和hp的选取，不同的数值会导致读取出来的数值不同。

### figure文件夹里面是画图的代码   

### Figure_data文件夹里面是三个数据集的ece全部信息

### summary忽略

### hfai_train.py训练代码文件  

### utils.py存储计算需要的函数  

### hfai_run在hf平台上的执行

### resnet模型resnet

### temperature温度调整函数，默认训练50epochs

### load_uqique_arch加载异构代码（同构模型只存在empirical error）

### 200-350train.py里面有加载模型权重相关的代码，仅供参考
