# verifying two trained network for 2000 database which is out of training and testing database#

import numpy as np
import torch
from function_sets import *
import matplotlib.pyplot as plt
from model import ResNet,mymodel


# verifying the classification

data =np.load('./data_2000_tst.npz')# np.load('./data_16000_wf08_08.npz')#
#data = np.load('./data_40000_08.npz')
Fb = 0.9
S=data['S']
#y = data['RC'];y1 = data['FC'];Y=data['CC'];
F = data['F'];a1 =data['A1'];a2 =data['A2'];
#RC=data['RC'];
Y = data['RH']
A12 =[];F_RH=[];
xRH =[];
for i in range(len(S)):
    if Y[i] ==1 and F[i]>Fb:#RC[i] ==1 and F[i]>Fb:
        xRH.append(S[i][:])
        A12.append([a1[i],a2[i]]);
        F_RH.append(F[i])

print('length of HR',len(F_RH))

xRH = np.concatenate(xRH);
xRH = np.reshape(xRH, (int(len(xRH)/160),160));

np.savez('HR.npz',xRH=xRH,A12=A12,FF = np.array(F_RH))

'''
data = np.load('./data_2000_test210.npz')
Fb = 0.9
x=data['S']
y = data['RC'];
y1 = data['FC'];
F = data['F'];Y=data['CC'];a1 =data['A1'];a2 =data['A2'];
HR =[];HuR=[];xHR =[];xHuR=[];F_HR=[];

for i in range(len(x)):
    if F[i]>Fb and y[i] ==1:
        HR.append([a1[i],a2[i]]);F_HR.append(F[i])
        xHR.append(x[i][:])

print('length of HR',len(HR))
xHR = np.concatenate(xHR);xHR = np.reshape(xHR, (int(len(xHR) / 160), 160));
np.savez('HR.npz',xHR=xHR,HR=HR,FF = np.array(F_HR))

#################verify the classification model


RH=[];
for i in range(len(Y)):
  if Y[i] == 0:
    RH.append(1)
  else:
    RH.append(0)
RH = np.array(RH)

Tst_x = to_matrix(x)
Tst_y = RH

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");print(device)

loss = torch.nn.CrossEntropyLoss().to(device)

Tst_x = torch.tensor(Tst_x).to(device).float()
Tst_y = torch.tensor(Tst_y).to(device).long()

net = mymodel.Conv.to(device)
net.load_state_dict(torch.load("classification.pth", map_location=device))
net.to(device)
prediction = net(Tst_x)
y_pred = tra_label = np.argmax(prediction.cpu().detach().numpy(), axis=1)

print('Accuracy of classification:', 1-np.sum(np.abs(y_pred-RH))/len(RH))
'''
S=S[:2000]
Y =Y[:2000]
Tst_x = to_matrix(S)
Tst_y = Y

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");print(device)

loss = torch.nn.CrossEntropyLoss().to(device)

Tst_x = torch.tensor(Tst_x).to(device).float()
Tst_y = torch.tensor(Tst_y).to(device).long()

net = mymodel.Conv.to(device)
net.load_state_dict(torch.load("Classification.pth", map_location=device))
net.to(device)
prediction = net(Tst_x)
y_pred = np.argmax(prediction.cpu().detach().numpy(), axis=1)

print('Accuracy of classification:', 1-np.sum(np.abs(y_pred-Y))/len(Y))



### verifying the regression

'''
data = np.load('./HR.npz')
x = data['xHR']#[:8000];
y =data['HR']#[:8000]
FF =data['FF']#[:8000]

x = x[:200]
y = y[:200]
FF = FF[:200]

Tst_x = to_matrix(x)
Tst_y = y
Tst_F = FF

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loss = torch.nn.MSELoss().to(device)# 均方差计算loss
Tst_x = torch.tensor(Tst_x).to(device).float()
Tst_y = torch.tensor(Tst_y).to(device).float()

model = ResNet()
model.load_state_dict(torch.load("regression.pth", map_location=device))
model.to(device)
prediction = model(Tst_x)

F_org = Tst_F  # original fidelity
y_pred = prediction.cpu().detach().numpy()

# tst_loss = loss(prediction, tra_y[i])
y_org = Tst_y.cpu().numpy()
x_org = Tst_x.cpu().numpy()

F_cell =[];S_cell = [];
for i in range(200):
  print(i)
  a1 = y_pred[i][0];
  a2 = y_pred[i][1]
  Si = x[i]#SS[i][0][0] - SS[i][0][0][0] * 0.5
  F_pre = produce_fidelity(a1, a2, Si)  # 求出预测的fidelity
  F_cell.append(F_pre)
  S_cell.append(i)
  #F_act = produce_fidelity(y[i][0], y[i][1], Si)
  #print(a1,a2,y[i][0],y[i][1])

  #print(F_pre,FF[i])#F_act,

np.savez('tst_regress.npz',F_pre=F_cell,F_org = FF)

plt.figure()
plt.scatter( S_cell,F_cell,color='red',marker='x')
plt.scatter( S_cell,FF,marker='o',facecolors='none',edgecolors='black')
plt.xlabel('S')
plt.xlabel('F')
np.savez('tst_regress.npz',F_pre=F_cell,F_org = FF)

print("Average fidelity difference is:",np.sum(np.abs(F_cell-FF))/200)

'''

### verifying the regression

# data = np.load('./HR.npz')
x = xRH  # data['xRH']#[:8000];
y = A12  # data['A12']#[:8000]
FF = F_RH  # data['FF']#[:8000]
NN = 200
x = x[:NN]
y = y[:NN]
FF = FF[:NN]

Tst_x = to_matrix(x)
Tst_y = y
Tst_F = FF

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loss = torch.nn.MSELoss().to(device)  # 均方差计算loss
Tst_x = torch.tensor(Tst_x).to(device).float()
Tst_y = torch.tensor(Tst_y).to(device).float()

model = ResNet()
model.load_state_dict(torch.load("regression1.pth", map_location=device))
model.to(device)
prediction = model(Tst_x)

F_org = Tst_F  # original fidelity
y_pred = prediction.cpu().detach().numpy()

# tst_loss = loss(prediction, tra_y[i])
y_org = Tst_y.cpu().numpy()
x_org = Tst_x.cpu().numpy()

F_cell = [];
S_cell = [];

for i in range(NN):
    a1 = y_pred[i][0];
    a2 = y_pred[i][1]
    Si = x[i]  # SS[i][0][0] - SS[i][0][0][0] * 0.5
    F_pre = produce_fidelity(a1, a2, Si)  # 求出预测的fidelity
    F_cell.append(F_pre)
    S_cell.append(i)
    print(i, 'fidelity diff:', np.abs(F_pre - F_org[i]), 'a1 diff', np.abs(a1 - y_org[i][0]), 'a2 diff',
          np.abs(a2 - y_org[i][1]))
    # F_act = produce_fidelity(y[i][0], y[i][1], Si)
    # print(a1,a2,y[i][0],y[i][1])

    # print(F_pre,FF[i])#F_act,

F_cell = np.array(F_cell)
FF = np.array(FF)

print("Average fidelity difference is:", np.sum(np.abs(F_cell - FF)) / NN)
np.savez('tst_regress.npz', F_pre=F_cell, F_org=FF)

plt.figure()
plt.scatter(S_cell, F_cell, color='red', marker='x')
plt.scatter(S_cell, FF, marker='o', facecolors='none', edgecolors='black')
plt.xlabel('S')
plt.xlabel('F')

plt.figure()
plt.plot(S_cell, FF, '-',lw = 2,color = 'grey')
plt.plot(S_cell, F_cell,'r-',lw = 0.5)

plt.xlabel('S')
plt.xlabel('F')

plt.figure()
plt.plot(S_cell, np.abs(FF-F_cell), '-',lw = 0.5,color = 'black')
#plt.plot(S_cell, F_cell,'r-',lw = 0.5)

plt.xlabel('S')
plt.xlabel('F')