import argparse
import torch
from torch import nn
from datetime import datetime
from datautil import loadmat
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import random
import numpy as np
import pandas as pd
from ptflops import get_model_complexity_info
from models import ResNet, CNNnet

USE_CUDA = torch.cuda.is_available()


def cnn_args():
    parser = argparse.ArgumentParser(description='CNN')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--batchsize', type=float, default=256)
    parser.add_argument('--weightdecay', type=float, default=0.01)
    parser.add_argument('--random_seed', type=int, default=10)  # 随机种子默认10
    parser.add_argument('--source1', type=str, default='.\GearData\Fea\Condition1.mat')
    parser.add_argument('--source2', type=str, default='.\GearData\Fea\Condition2.mat')
    parser.add_argument('--source3', type=str, default='.\GearData\Fea\Condition3.mat')
    parser.add_argument('--source4', type=str, default='.\GearData\Fea\Condition0.mat')
    args = parser.parse_args()
    return args


def get_scheduler(optimizer, args):
    if not True:
        return None
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))  # ** 乘方
    return scheduler


def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    args = cnn_args()
    device = args.device
    set_random_seed(args.random_seed)
    model = CNNnet().to(device)     # 对比ResNet和CNNNet之间的计算复杂度
    # model = ResNet().to(device)
    writer = SummaryWriter()
    dump_input = torch.rand(256, 1, 32).to(device)
    writer.add_graph(model, dump_input)
    macs, params = get_model_complexity_info(model, (1, 32), as_strings=False, print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    batchsize = args.batchsize
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weightdecay)
    sch = get_scheduler(optimizer, args)

    traindata1 = loadmat.TrainDataset(args.source1)
    dataloader1 = DataLoader(dataset=traindata1, batch_size=batchsize, pin_memory=True, shuffle=True, drop_last=False)
    traindata2 = loadmat.TrainDataset(args.source2)
    dataloader2 = DataLoader(dataset=traindata2, batch_size=batchsize, pin_memory=True, shuffle=True, drop_last=False)
    traindata3 = loadmat.TrainDataset(args.source3)
    dataloader3 = DataLoader(dataset=traindata3, batch_size=batchsize, pin_memory=True, shuffle=True, drop_last=False)
    testdata = loadmat.TrainDataset(args.source4)
    testdataloader = DataLoader(dataset=testdata, batch_size=batchsize, pin_memory=True, shuffle=True, drop_last=False)

    final_acc = 0
    filename = datetime.now().strftime('%d-%m-%y-%H_%M.pth')

    for i in range(1000):  # epoch
        model.train()
        train_iter = zip(dataloader1, dataloader2, dataloader3)
        total_loss = 0
        for j in range(dataloader1.__len__()):  # batch
            minibatch = [data for data in next(train_iter)]
            losses = torch.zeros(dataloader1.__len__()).to(device)
            q = torch.ones(dataloader1.__len__()).to(device)
            loss = 0
            for m in range(len(minibatch)):  # domain
                batch_data, batch_label = minibatch[m][0].cuda().float(), minibatch[m][1].cuda().long()
                data = torch.unsqueeze(batch_data, 1).to(device)
                data = torch.cat((data, data), 2)
                label = torch.squeeze(batch_label - 1, dim=1).to(device)
                outputs = model(data)
                losses[m] = criterion(outputs, label)
                q[m] *= (0.8 * losses[m].data).exp()
            q /= q.sum()
            loss = torch.dot(losses, q)  # 是否采用RDAWO trick
            # loss = torch.sum(losses)
            total_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sch.step()

        if i % 30 == 0:
            model.eval()
            pre_total = []
            act_total = []
            with torch.no_grad():
                total_num = 0
                for testbatch, (testX, testY) in enumerate(testdataloader):
                    testX = torch.unsqueeze(testX, dim=1).to(device)
                    testX = torch.cat((testX, testX), 2)
                    testY = torch.squeeze(testY - 1, dim=1).to(device)
                    predictY = torch.argmax(model(testX), -1)
                    num = testY.eq(predictY).sum().item()
                    total_num = total_num + num
                    pre_total = np.append(pre_total, predictY.cpu().numpy())
                    act_total = np.append(act_total, testY.cpu().numpy())
                acc = total_num / testdata.__len__()
                print('test accuracy: %f' % acc)
                writer.add_scalar('test accuracy:', acc, i)
                if final_acc < acc:  
                    # if i == 990:
                    final_acc = acc
                    torch.save(model, './model/' + filename)
                    lst = {"predict": pre_total, "actual": act_total}
                    save = pd.DataFrame(lst)
                    save.to_csv('./model/predict_label.csv')
        print('epoch: %d, loss: %f' % (i, total_loss))
        writer.add_scalar('training loss:', total_loss, i)
    writer.close()
