import argparse
import time

import layers
import torch
from net import Net
from utils import DataLoaderS,Optim
import numpy as np
import math
import torch.nn as nn

def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size, last_test=False):
    model.eval ()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None
    for X, Y in data.get_batches (X, Y, batch_size, False):
        X = torch.unsqueeze (X, dim=1)
        X = X.transpose (2, 3)
        with torch.no_grad ():
            output= model (X)
        output = torch.squeeze (output)
        if len (output.shape) == 1:
            output = output.unsqueeze (dim=0)
        scale = data.scale.expand (output.size (0), data.m)
        if predict is None:
            predict = output
            test = Y
        else:
            predict = torch.cat ((predict, output))
            test = torch.cat ((test, Y))
        total_loss += evaluateL2 (output * scale, Y * scale).item ()
        total_loss_l1 += evaluateL1 (output * scale, Y * scale).item ()
        n_samples += (output.size (0) * data.m)

    rse = math.sqrt (total_loss / n_samples) / data.rse
    rae = (total_loss_l1 / n_samples) / data.rae
    mae = total_loss_l1 / n_samples
    mse = total_loss / n_samples

    predict = predict.data.cpu ().numpy ()
    Ytest = test.data.cpu ().numpy ()
    sigma_p = (predict).std (axis=0)
    sigma_g = (Ytest).std (axis=0)
    mean_p = predict.mean (axis=0)
    mean_g = Ytest.mean (axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean (axis=0) / (sigma_p * sigma_g)
    correlation = (correlation [index]).mean ()



    return rse, mae, correlation, rae, mse


def train(data, X, Y, model, criterion, optim, batch_size):
    # 设置模型为训练模式
    model.train()
    # 初始化总损失为0
    total_loss = 0
    # 初始化样本数量为0
    n_samples = 0
    # 初始化迭代次数为0
    iter = 0

    # 获取批量数据
    for X, Y in data.get_batches(X, Y, batch_size, True):
        # 将梯度清零
        model.zero_grad()
        # 增加通道维度
        X = torch.unsqueeze(X, dim=1)
        # 转置
        X = X.transpose(2, 3)
        pre = model(X)
        scale = data.scale.expand (pre.size (0), data.m)
        # print(scale.shape)
        # if iter % args.step_size == 0:
        #     perm = np.random.permutation(range(args.num_nodes))
        # num_sub = int(args.num_nodes / args.num_split)

        loss = criterion (pre * scale, Y * scale)
        loss.backward ()
        total_loss += loss.item ()
        n_samples += (pre.size (0) * data.m)

        # 每隔100次迭代打印损失
        if iter % 100 == 0:
            print('iter:{:3d} | loss: {:.3f}'.format(iter, loss.item() / (pre.size(0) * data.m)))
            # 如果设置了break则退出循环
            # break
        # 增加迭代次数
        iter += 1

    # 返回平均损失
    return total_loss / n_samples

parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, default='sea_breeze/sea_breeze1.txt',
                    help='location of the data file')
parser.add_argument('--log_interval', type=int, default=2000, metavar='N', help='report interval')
parser.add_argument('--save', type=str, default='model/model.pt',
                    help='path to save the final model')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--gcn_depth',type=int,default=2,help='graph convolution depth')
parser.add_argument('--num_nodes',type=int,default=16,help='number of nodes/variables')
# parser.add_argument('--num_nodes',type=int,default=40,help='number of nodes/variables')
# parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
# parser.add_argument('--subgraph_size',type=int,default=20,help='k')
parser.add_argument('--node_dim',type=int,default=40,help='dim of nodes')
# parser.add_argument('--conv_channels',type=int,default=16,help='convolution channels')
parser.add_argument('--scale_channels',type=int,default=32,help='scale channels')
parser.add_argument('--end_channels',type=int,default=64,help='end channels')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--seq_in_len',type=int,default=24*7,help='input sequence length')
parser.add_argument('--seq_out_len',type=int,default=1,help='output sequence length')
parser.add_argument('--horizon', type=int, default=3)
parser.add_argument('--layers',type=int,default=3,help='number of layers')
parser.add_argument('--batch_size',type=int,default=4,help='batch size')
parser.add_argument('--lr',type=float,default=0.0001,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.00001,help='weight decay rate')
parser.add_argument('--clip',type=int,default=5,help='clip')
parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
parser.add_argument('--tanhalpha',type=float,default=3,help='tanh alpha')
parser.add_argument('--epochs',type=int,default=1,help='')
parser.add_argument('--num_split',type=int,default=1,help='number of splits for graphs')
parser.add_argument('--step_size',type=int,default=100,help='step_size')


args = parser.parse_args()
# device = torch.device(args.device)
device = torch.device('cpu')
torch.set_num_threads(3)


def main(params):
    alpha = params ['alpha']
    k = params ['k']
    beta = params ['beta']
    data_dir = "data/" + args.data

    Data = DataLoaderS (data_dir, 0.6, 0.2, device, args.horizon, args.seq_in_len, args.normalize)

    model = Net(device, args.horizon, args.num_nodes, args.seq_in_len, dim=1, alpha=alpha, k=k,beta = beta)
    model = model.to (device)

    print (args)

    nParams = sum ([p.nelement () for p in model.parameters ()])
    print ('Number of model parameters is', nParams, flush=True)


    if args.L1Loss:
        criterion = nn.L1Loss (size_average=False).to (device)
    else:
        criterion = nn.MSELoss (size_average=False).to (device)
    evaluateL2 = nn.MSELoss (size_average=False).to (device)
    evaluateL1 = nn.L1Loss (size_average=False).to (device)

    best_val = 10000000
    optim = Optim (model.parameters (), args.optim, args.lr, args.clip, lr_decay=args.weight_decay)

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        print ('begin training')
        for epoch in range (1, args.epochs + 1):
            epoch_start_time = time.time ()
            train_loss = train (Data, Data.train [0], Data.train [1], model, criterion, optim, args.batch_size)
            val_loss, val_rae, val_corr, val_mae, val_rmse = evaluate (Data, Data.valid [0], Data.valid [1], model,
                                                                       evaluateL2, evaluateL1,
                                                                       args.batch_size)

            print (
                '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f} | valid mae  {:5.4f} | valid rmse  {:5.4f}'.format (
                    epoch, (time.time () - epoch_start_time), train_loss, val_loss, val_rae, val_corr, val_mae,
                    val_rmse), flush=True)
            # Save the model if the validation loss is the best we've seen so far.

            if val_loss < best_val:
                with open (args.save, 'wb') as f:
                    torch.save (model, f)
                best_val = val_loss
            if epoch % 5 == 0:
                test_acc, test_rae, test_corr, test_mae, test_rmse = evaluate (Data, Data.test [0], Data.test [1],
                                                                               model, evaluateL2, evaluateL1,
                                                                               args.batch_size)
                print (
                    "test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f} | test mae  {:5.4f} | test rmse  {:5.4f}".format (
                        test_acc, test_rae, test_corr, test_mae, test_rmse), flush=True)


    except KeyboardInterrupt:
        print ('-' * 89)
        print ('Exiting from training early')


    # Load the best saved model.
    with open (args.save, 'rb') as f:
        model = torch.load (f)

    vtest_acc, vtest_rae, vtest_corr, vtest_mae, vtest_rmse = evaluate (Data, Data.valid [0], Data.valid [1], model,
                                                                        evaluateL2, evaluateL1,
                                                                        args.batch_size)
    test_acc, test_rae, test_corr, test_mae, test_rmse = evaluate (Data, Data.test [0], Data.test [1], model,
                                                                   evaluateL2, evaluateL1,
                                                                   args.batch_size)
    print (
        "final test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f} | test mae {:5.4f} | test mae  {:5.4f} | test rmse  {:5.4f}".format (
            test_acc, test_rae, test_corr, test_mae, test_mae, test_rmse))


    return test_acc

if __name__ == "__main__":
    k = 1
    beta = 0.6
    alpha = 0.5
    params = {}
    params ['alpha'] = alpha
    params ['k'] = k
    params ['beta'] = beta
    main(params)