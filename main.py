import argparse
import layers
import torch
import net
from utils import DataLoaderS

parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, default='exchange_rate.txt',help='location of the data file')
gc = layers.static_graph_constructor(nnodes=16,dim=1, device='cpu',out_channels=1)
x = torch.randint(0, 15,size=(1,16)).squeeze()
# print(x.shape)
# for para in gc.named_parameters():
#     print(para[0],para[1].size)
# for p in gc.parameters():
#     print(p)

args = parser.parse_args()
# device = torch.device(args.device)
device = torch.device('cpu')
data_dir = "data/" + args.data

Data = DataLoaderS (data_dir, 0.6, 0.2, device, 3, 5, 2)
# print(Data.train[0].shape)
net = net.Net(device, 3, 8, 5, 1, 0.5, 1)
ngc = layers.dynamic_graph_constructor(3,8,5)

for X, Y in Data.get_batches(Data.train[0], Data.train[1], 4, True):
    # 增加通道维度
    X = torch.unsqueeze (X, dim=1)
    # 转置
    X = X.transpose (2, 3)
    # print (X.shape)
    # gc (X)

    pre = net(X)
    scale = Data.scale.expand (pre.size (0), Data.m)
    print (scale.shape,Data.m)
    print(Y.shape, pre.shape)
    # print(Y.shape)