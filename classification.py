import torch.nn as nn
from function_sets import *
import numpy as np
import torch
import time
from torch import nn
from model import mymodel
import matplotlib.pyplot as plt



data = np.load('./data40000_tf1_wf01.npz')
Fb = 0.9
x=data['S']
#y = data['RC'];y1 = data['FC'];Y=data['CC'];
F = data['F'];a1 =data['A1'];a2 =data['A2'];
RH = data['RH']
X1 = x
Y1 =RH


plt.scatter(np.linspace(0,len(F),len(F)),F,s = 0.1)


occ = int(0.8*len(X1))
batch_size = 200#100#

tra_x = X1[:occ]#X1[:26000]
tra_y = Y1[:occ]
tst_x = X1[occ:]
tst_y = Y1[occ:]

tra_num = tra_x.shape[0]
tst_num = tst_x.shape[0]

occ_posi = round(np.sum(tra_y)/tra_num,3)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
batch_num = tra_num // batch_size
tst_batch_num = tst_num // batch_size

# 使用卷积请运行下方三行代码
tra_x,tst_x=for_cnn(tra_x,tst_x)

tra_x = to_batch(tra_x, batch_size)  #
tra_y = to_batch(tra_y, batch_size)
tst_x = to_batch(tst_x, batch_size)
tst_y = to_batch(tst_y, batch_size)


tra_x = torch.tensor(tra_x).to(device).float()
tra_y = torch.tensor(tra_y).to(device).long()
tst_x = torch.tensor(tst_x).to(device).float()
#tst_y = torch.tensor(tst_y).to(device).long()# 设为long就全部失真了


# from model_ResNet34 import ResNet

epochs = 30

learning_rate = 0.0001


net = mymodel.Conv.to(device)  # .cuda()
loss = torch.nn.CrossEntropyLoss().to(device)
opt = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.001)
print(net)
tra_loss_list = []
tst_loss_list = []
tra_acc_list = []
tst_acc_list = []
Epoch = []
Tra_loss_batch = np.zeros(shape=(epochs, batch_num))
Tst_loss_batch = np.zeros(shape=(epochs, tst_batch_num))

Tra_acc_batch = np.zeros(shape=(epochs, batch_num))
Tst_acc_batch = np.zeros(shape=(epochs, tst_batch_num))

for epoch in range(epochs):
    Epoch.append(epoch)
    num = 0
    total_loss = 0
    start = time.time()
    for batch in range(batch_num):
        net.train()
        logits = net(tra_x[batch])
        tra_loss = loss(logits, tra_y[batch])

        tra_label = np.argmax(logits.cpu().detach().numpy(), axis=1)
        right_num = np.zeros([batch_size, ])
        right_num[tra_label == tra_y[batch].cpu().detach().numpy()] = 1
        right_num = np.sum(right_num)
        num += right_num

        Tra_acc = right_num / batch_size
        Tra_acc_batch[epoch][batch] = Tra_acc
        Tra_loss = tra_loss.item()
        Tra_loss_batch[epoch][batch] = Tra_loss

        opt.zero_grad()
        tra_loss.backward()
        '''
        for name, param in net.named_parameters():
            #print('层:', name, param.size())
            print('权值梯度', param.grad)
            #print('权值', param)
        '''

        opt.step()
        total_loss += tra_loss.item()
    end = time.time()
    # print('time for one epoach:',end-start,)
    tra_loss_list.append(total_loss / batch_num)
    total_loss / batch_num
    tra_acc = num / tra_num
    tra_acc_list.append(tra_acc)

    net.eval()
    num = 0
    total_loss = 0

    for batch in range(tst_batch_num):
        tst_logits = net(tst_x[batch])
        tst_label = np.argmax(tst_logits.cpu().detach().numpy(), axis=1)
        tst_loss = loss(tst_logits, torch.tensor(tst_y[batch]).to(device).long())

        right_num = np.zeros(tst_y[batch].shape)
        right_num[tst_label == tst_y[batch]] = 1
        right_num = np.sum(right_num)
        num += right_num
        total_loss += tst_loss.item()

        Tst_acc = right_num / batch_size  #
        Tst_acc_batch[epoch][batch] = Tst_acc
        Tst_loss = tst_loss.item()
        Tst_loss_batch[epoch][batch] = Tst_loss

    test_loss = total_loss / tst_batch_num
    tst_loss_list.append(test_loss)
    tst_acc = num / tst_num
    tst_acc_list.append(tst_acc)
    print('Epoch:', epoch, 'train_acc:', tra_acc, 'test_acc:', tst_acc, 'tra_loss:', total_loss / batch_num,
          'test_loss:', test_loss)
    # print('train_acc:', tra_acc, 'test_acc:', tst_acc)



plt.figure()
#plt.style.use('science')
plt.subplot(1,2,1)
plt.plot(Epoch,tra_loss_list, label='train loss', color='red')
plt.plot(Epoch,tst_loss_list, label='test loss', color='blue', linestyle=':')
plt.legend()
plt.xlabel(r'$epoch$',fontsize=15)
plt.ylabel(r'$Loss$',fontsize=15)
plt.subplot(1,2,2)
plt.plot(Epoch,tra_acc_list, label='train acc', color='red')
plt.plot(Epoch,tst_acc_list, label='test acc', color='blue', linestyle=':')
plt.legend()
plt.xlabel(r'$epoch$',fontsize=15)
plt.ylabel(r'$Accuracy$',fontsize=15)
plt.show()

np.savez('classification.npz',
         Tra_loss_batch=Tra_loss_batch,
         Tst_loss_batch=Tst_loss_batch,
         Tra_acc_batch=Tra_acc_batch,
         Tst_acc_batch=Tst_acc_batch,
         tra_loss_list=tra_loss_list,
         tst_loss_list=tst_loss_list,
         tra_acc_list=tra_acc_list,
         tst_acc_list=tst_acc_list)
torch.save(net.state_dict(),"classification.pth")