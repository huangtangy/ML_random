import matplotlib.pyplot as plt
import numpy as np


###############################################


data = np.load('classification1.npz')#('classification.npz')#
Tra_loss_batch=data['Tra_loss_batch']
Tst_loss_batch=data['Tst_loss_batch']
Tra_acc_batch=data['Tra_acc_batch']
Tst_acc_batch=data['Tst_acc_batch']
tra_loss_list=data['tra_loss_list']
tst_loss_list=data['tst_loss_list']
tra_acc_list=data['tra_acc_list']
tst_acc_list=data['tst_acc_list']

Epoch= np.linspace(0,30,30)
plt.figure()
plt.style.use('science')
plt.subplot(1,3,1)
plt.title('classification')
for i in range(len(Epoch)):
    epp = []
    epp1 = []
    for j in range(len(Tra_loss_batch[i])):
        epp.append(i)
    for j in range(len(Tst_loss_batch[i])):
        epp1.append(i)

    # print(len(Tra_loss_batch[i]),len(epp))
    plt.plot(epp, Tra_acc_batch[i], lw=5,color='lightskyblue')
    plt.plot(epp1, Tst_acc_batch[i], lw=5, color='lightpink',alpha = 0.6)#'lightsalmon'#lightpink
plt.plot(Epoch, tra_acc_list, label='training',color='blue',lw=1.5)
plt.plot(Epoch, tst_acc_list, label='testing', color='red',lw =1.5, linestyle='--')#, marker='o'
plt.legend(loc='lower right');#plt.grid()
plt.ylim([0.8, 1.02])
plt.xlim([0, 30])
plt.text(2,0.82,r'(a)',fontsize = 14)
plt.xlabel('epoch',fontsize = 14)
plt.ylabel('accuracy',fontsize = 14)
plt.show()


########################
data = np.load('RegressionRes18_16.npz')#('regression.npz')#('regressHR (8).npz')

Tra_loss_batch=data['Tra_loss_batch']
Tst_loss_batch=data['Tst_loss_batch']

tra_loss_list=data['tra_loss_list']
tst_loss_list=data['tst_loss_list']

Tst_F_batch = data['Tst_F_batch']
Tra_F_batch = data['Tra_F_batch']
F_org_tra = data['F_org_tra']
F_org_tst = data['F_org_tst']
tra_F_list=data['tra_F_list']
tst_F_list = data['tst_F_list']


ac=50;
Epoch = np.linspace(0,ac,ac)
plt.style.use('science')


############

NN=10
plt.subplot(1,3,2)


for i in range(len(Tra_F_batch)):#epochs
  epp=[]
  for j in range(len(Tra_F_batch[i])):
    epp.append(i)
  plt.plot(epp[:NN],np.abs(Tra_F_batch[i]-F_org_tra[i])[:NN],lw=7,color='royalblue')
  #print(epp[:27],np.abs(Tra_F_batch[i]-F_org_tra[i])[:27])
for i in range(len(Tst_F_batch)):#epochs
  epp1=[]
  for j in range(len(Tst_F_batch[i])):
    epp1.append(i)

  plt.plot(epp1[:NN],np.abs(Tst_F_batch[i]-F_org_tst[i])[:NN],lw=5.5,color='grey',alpha = 0.6)
  #print(tst_F_list[i],np.sum(np.abs(np.array(Tst_F_batch[i])-np.array(F_org_tst[i])))/(tst_batch_num))
plt.plot(Epoch,tra_F_list, label='training', color='blue')
plt.plot(Epoch,tst_F_list, label='testing', color='black', linestyle='--')
plt.title('regression')
plt.ylim([10**(-6),10**(-1)])
plt.yscale('log')
plt.xlim([0,50])
plt.xticks([0,10,20,30,40,50])
plt.legend();#plt.grid()
plt.text(2,10**(-5)/5,r'(b)',fontsize = 14)
plt.xlabel('epoch',fontsize = 14)
plt.ylabel(r'$\Delta F$',fontsize = 14)
plt.show()



############################


data = np.load('tst_regress (1).npz')
F_cell = data['F_pre']
FF=data['F_org']
S_cell = np.linspace(0,100,100)

plt.subplot(1,3,3)
plt.title('verification')
plt.scatter( S_cell,F_cell[:100],color='tomato',marker='x',s=45,lw = 0.7)
plt.scatter( S_cell,FF[:100],marker='o',facecolors='none',edgecolors='black',lw=0.5,s=45)
#plt.grid()
plt.text(9,0.904,'(c)',fontsize=14,color='black',backgroundcolor='0.9')#,color='white'
plt.yticks([0.9,0.95,1])
plt.xlabel('realizations',fontsize = 14)
plt.ylabel('fidelity',fontsize = 14)
