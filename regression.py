
#### training the CNN2 for regression, ( using gpu is time-saving )

import numpy as np
import torch
import time
from torch import nn
import matplotlib.pyplot as plt
from matlab import fft, ifft, fftshift, ifftshift
import random
#from model import ResNet

import torch as t
from torch import nn
from torch.nn import functional as F
from functions import *
### Defining the CNN2

import torch as t
from torch import nn
from torch.nn import functional as F

### Defining the CNN2

class ResidualBlock(nn.Module):
    """
    实现子 module：Residual Block
    """
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()

        # 由于 Residual Block 分为左右两部分，因此定义左右两边的 layer
        # 定义左边
        self.left = nn.Sequential(
            # Conv2d 参数：in_channel,out_channel,kernel_size,stride,padding
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            #(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
            )
        # 定义右边
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)  # 检测右边直连的情况
        out += residual
        return F.relu(out)
class ResNet(nn.Module):
    """
    实现主 module：ResNet34
    ResNet34 包含多个 layer，每个 layer 又包含多个 residual block
    用子 module 实现 residual block，用 _make_layer 函数实现 layer
    """

    def __init__(self, num_classes=2):
        super(ResNet, self).__init__()
        # 前几层图像转换
        fm = 16
        self.pre = nn.Sequential(
            nn.Conv2d(1, fm, 3, 1, 1, bias=False),
            nn.BatchNorm2d(fm), #nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),#kernel_size, stride=None, padding
        )

        # 重复的 layer 分别有 3，4，6，3 个 residual block

        self.layer1 = self._make_layer(fm, fm*2, 3)
        self.layer2 = self._make_layer(fm*2, fm*4, 4, stride=2)
        self.layer3 = self._make_layer(fm*4, fm*8, 6, stride=2)
        self.layer4 = self._make_layer(fm*8, fm*8, 3, stride=2)

        # 分类用的全连接
        self.fc = nn.Linear(fm*8, num_classes)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        """
        构建 layer，包含多个 residual block
        """
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(outchannel)
            )

        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)

        return self.fc(x)

####

data = np.load('data40000_tf1_wf01.npz')
Fb = 0.8
S=data['S']
#y = data['RC'];y1 = data['FC'];Y=data['CC'];
F = data['F'];a1 =data['A1'];a2 =data['A2'];
#RC=data['RC'];
RH = data['RH']
A12 =[];F_RH=[];
xRH =[];
for i in range(len(S)):
    if RH[i] ==1:#RC[i] ==1 and F[i]>Fb:
        xRH.append(S[i][:])
        A12.append([a1[i],a2[i]]);
        F_RH.append(F[i])

print('length of HR',len(F_RH))

xRH = np.concatenate(xRH);
xRH = np.reshape(xRH, (int(len(xRH)/160),160));

# formulating the database
x = xRH
y = A12

X1 = np.array(x); Y1 = np.array(y)
FF = np.array(F_RH)#[:4000]
occ = int(0.8*len(x)) # 80% ans training data
batch_size = 20

tra_x = X1[:occ]
tra_y = Y1[:occ]
tra_F = FF[:occ]

#print(tra_x.shape)
tst_x = X1[occ:]
tst_y = Y1[occ:]
tst_F = FF[occ:]
tra_num = tra_x.shape[0]
tst_num = tst_x.shape[0]

occ_posi = round(np.sum(tra_y)/tra_num,3)

batch_num = tra_num // batch_size
tst_batch_num = tst_num // batch_size
print('tra_batch_num',batch_num,"tst_batch_num",tst_batch_num)

tra_x,tst_x=for_cnn(tra_x,tst_x) # extending to 2d
#net = model.Conv.to(device)

tra_x = to_batch(tra_x, batch_size)  #
tra_y = to_batch(tra_y, batch_size)
tst_x = to_batch(tst_x, batch_size)
tst_y = to_batch(tst_y, batch_size)
tra_fidelity = to_batch(tra_F, batch_size)
tst_fidelity = to_batch(tst_F, batch_size)


#device ='cpu'# 'cuda' #若未配置cuda环境请将device改为'cpu'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# convert to tensor
tra_x = torch.tensor(tra_x).to(device).float()
tra_y = torch.tensor(tra_y).to(device).float()#long() for entropycross()
tst_x = torch.tensor(tst_x).to(device).float()
tst_y = torch.tensor(tst_y).to(device).float()




### start training


drop = 0
epochs = 30
learning_rate = 0.0001

# import AlexNet
# from model_ResNet import ResNet_34
# net = model.Conv1d.to(device)
# net = AlexNet().to(device)
net = ResNet().to(device)
# net = model.Conv.to(device)
print(tra_x.shape)
loss = torch.nn.MSELoss().to(device)  # 均方差计算loss
opt = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.001)

tra_loss_list = []
tst_loss_list = []
tra_acc_list = []
tst_acc_list = []
tst_F_list = []
tra_F_list = []

Epoch = []

Tra_loss_batch = np.zeros(shape=(epochs, batch_num))
Tst_loss_batch = np.zeros(shape=(epochs, tst_batch_num))

Tra_F_batch = np.zeros(shape=(epochs, batch_num))
Tst_F_batch = np.zeros(shape=(epochs, tst_batch_num))

F_org_tra = np.zeros(shape=(epochs, batch_num))
F_org_tst = np.zeros(shape=(epochs, tst_batch_num))
NN1 = batch_num // 10

for epoch in range(epochs):  # epochs
    # initialization
    Epoch.append(epoch);
    num = 0;
    total_loss = 0;
    j = random.randint(0, 3)
    start = time.time()
    jj = 0
    for batch in range(batch_num):
        net.train()
        logits = net(tra_x[batch])
        tra_loss = loss(logits, tra_y[batch])  # 每个batch的均方差的值[\Sigma(y_pre-y_real)^2]/batch_num
        tra_value = logits.cpu().detach().numpy()

        if batch % NN1 == 0:
            # batch<NN:

            ### fidelity
            F_org = tra_fidelity[batch]  # org fidelity
            prediction = logits
            y_pred = prediction.cpu().detach().numpy()

            # tst_loss = loss(prediction, tra_y[i])
            y_org = tra_y[batch].cpu().numpy()
            x_org = tra_x[batch].cpu().numpy()

            SS = x_org
            a1 = y_pred[j][0];
            a2 = y_pred[j][1];  # 在这个batch中取任意第j个作为这个batch的平均

            Si = SS[j][0][0] - SS[j][0][0][0] * 0.5
            F_pre = produce_fidelity(a1, a2, Si)  # 求出预测的fidelity

            Tra_F_batch[epoch][jj] = F_pre  #
            F_org_tra[epoch][jj] = F_org[j]
            jj += 1

        opt.zero_grad()  # 梯度清0
        tra_loss.backward()  # 自动对loss梯度
        opt.step()  # 优化参数
        total_loss += tra_loss.item()  # 把每个batch的平均loss累加

        Tra_loss_batch[epoch][batch] = tra_loss.item()

    # print(tra_loss.item())
    end = time.time()
    loss_ave = total_loss / (batch_num)  # average of every loss *batch_size
    tra_loss_list.append(loss_ave)

    F_ave = np.sum(np.abs(np.array(Tra_F_batch[epoch]) - np.array(F_org_tra[epoch]))) / (
        jj)  # average fidelity difference for every epoch
    tra_F_list.append(F_ave)

    net.eval()
    num = 0
    total_loss = 0

    NN2 = tst_batch_num // 10
    jj1 = 0

    for batch in range(tst_batch_num):

        tst_logits = net(tst_x[batch])
        tst_loss = loss(tst_logits, tst_y[batch])  # torch.tensor(tst_y[batch]).to(device).long())
        total_loss += tst_loss.item()

        if batch % NN2 == 0:
            # taking the first element of fiset 10 batch in every epoch for calculating the average vaule
            ### fidelity
            F_org = tst_fidelity[batch]  # org fidelity
            prediction = tst_logits
            y_pred = prediction.cpu().detach().numpy()

            # tst_loss = loss(prediction, tra_y[i])
            y_org = tst_y[batch].cpu().numpy()
            x_org = tst_x[batch].cpu().numpy()

            SS = x_org
            a1 = y_pred[j][0];
            a2 = y_pred[j][1];  # 在这个batch中取任意第j个作为这个batch的平均

            Si = SS[j][0][0] - SS[j][0][0][0] * 0.5
            F_pre = produce_fidelity(a1, a2, Si)  # 求出预测的fidelity
            # print(a1,a2,tst_y[batch][j],F_pre)

            Tst_F_batch[epoch][jj1] = F_pre  #
            F_org_tst[epoch][jj1] = F_org[j]
            jj1 += 1

        Tst_loss_batch[epoch][batch] = tst_loss.item()

    loss_ave_tes = total_loss / (tst_batch_num)  # *batch_size
    tst_loss_list.append(loss_ave_tes)

    F_ave = np.sum(np.abs(np.array(Tst_F_batch[epoch]) - np.array(F_org_tst[epoch]))) / (jj1)
    tst_F_list.append(F_ave)

    print('Epoch:', epoch, 'train_loss_average:', loss_ave, 'test_loss_average:', loss_ave_tes, 'tst fidelity_average:',
          F_ave)


plt.figure()
plt.plot(Epoch,tra_loss_list,label='Training loss')
plt.plot(Epoch,tst_loss_list,label='test loss')
plt.legend()
plt.yscale('log')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


np.savez('regression.npz',
         Tra_loss_batch=Tra_loss_batch,
         Tst_loss_batch=Tst_loss_batch,
         tra_loss_list=tra_loss_list,
         tst_loss_list=tst_loss_list,
         Tst_F_batch=Tst_F_batch,
         Tra_F_batch = Tra_F_batch,
         F_org_tra=F_org_tra,
         F_org_tst = F_org_tst,
         tra_F_list=tra_F_list,
         tst_F_list=tst_F_list,
         )
torch.save(net.state_dict(),"regression.pth")