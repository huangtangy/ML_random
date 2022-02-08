import matplotlib.pyplot as plt
import numpy as np
import pandas as pd





Epoch = np.linspace(0,30,30)
plt.style.use('science')
oc= 20

###############################################

'''
fig = plt.figure()
ax1 = fig.add_subplot(325)
plt.errorbar(Tf,Loss_Ave,yerr=error_loss, fmt='r-^', capsize=4, capthick=1,lw = 0.5,markersize=4)
ax1.set_ylabel(r'Loss',fontsize=12)
ax1.set_xlabel(r'$t_f$',fontsize=13,color='red')
#ax1.set_yscale('log')
for tl in ax1.get_xticklabels():
    tl.set_color('r')

ax2 = ax1.twiny()  # this is the important function
plt.errorbar(Wf,Loss_Ave1,yerr=error_loss1, fmt='b-o', capsize=4, capthick=1,lw = 0.5,markersize=4)
#ax2.set_xlabel(r'$t_f$',fontsize=15)
#ax2.set_xlabel(r'$\omega_f$',fontsize=15)
for tl in ax2.get_xticklabels():
    tl.set_color('b')
plt.ylim([0.02,0.205])
plt.yticks([0,0.05,0.1,0.15,0.2])
#plt.text(0.34,0.25,r'$Classification$',fontsize = 12)
plt.text(0.23,0.02,'(e)',fontsize = 10)
plt.text(0.45,0.23,r'$\omega_f$',fontsize = 12,color='blue')
#plt.grid()

ax1 = fig.add_subplot(326)
plt.errorbar(Tf,FLoss_Ave,yerr=error_Floss, fmt='r-^', capsize=4, capthick=1,lw = 0.5,markersize=4)
ax1.set_ylabel('Loss',fontsize=10)
ax1.set_xlabel(r'$t_f$',fontsize=10,color='red')
ax1.set_yscale('log')
for tl in ax1.get_xticklabels():
    tl.set_color('r')
ax2 = ax1.twiny()  # this is the important function
plt.errorbar(Wf,FLoss_Ave1,yerr=error_Floss1, fmt='b-o', capsize=4, capthick=1,lw = 0.5,markersize=4)
#ax2.set_xlabel(r'$t_f$',fontsize=15)
#ax2.set_xlabel(r'$\omega_f$',fontsize=15)

for tl in ax2.get_xticklabels():
    tl.set_color('b')
plt.ylim([0.1,100])
#plt.text(0.34,500,r'$Regression$',fontsize = 12)
plt.text(0.23,1/6,'(f)',fontsize = 10)
plt.text(0.45,250,r'$\omega_f$',fontsize = 12,color='blue')
#plt.yticks([10**(-5),10**(-4),10**(-3),10**(-2)])
#plt.grid()

'''


#################

plt.figure()
data = np.load('class1d.npz')

tra_acc_list1d = data['tra_acc_list'];tst_acc_list1d = data['tst_acc_list']

data = np.load('classification1.npz')
tra_acc_list2d = data['tra_acc_list'];tst_acc_list2d = data['tst_acc_list']

Epoch = np.linspace(0,30,30)

plt.subplot(2,2,3)
#plt.subplot(1,2,2)
#plt.title(r'$Classification$')
plt.plot(Epoch,tst_acc_list1d, label='1D', color='black', lw = 2,linestyle='--')
plt.plot(Epoch,tst_acc_list2d,  label='2D', color='grey', lw = 2)

plt.legend();#plt.grid();#ncol =2,loc="upper right"
#plt.yscale('log')
plt.ylim([0.8,1])
plt.text(2,0.81,'(c)',fontsize = 12)
plt.xlabel('epoch',fontsize=12)
plt.ylabel('accuracy',fontsize=12)
plt.xticks([0,15,30])
#plt.yticks([-100,-50,0,50,100])

data = np.load('RegressionRes18_16.npz')

tra_F_list2d=data['tra_F_list']
tst_F_list2d = data['tst_F_list']

data1 = np.load('regress1d (3).npz')
tra_F_list1d=data1['tra_F_list']
tst_F_list1d = data1['tst_F_list']


ac=50;
Epoch = np.linspace(0,ac,ac)
plt.subplot(2,2,4)
#plt.title(r'$Regression$')
plt.plot(Epoch,tst_F_list1d, label='1D', color='black', lw = 2,linestyle='--')
plt.plot(Epoch,tst_F_list2d, label='2D', color='grey', lw = 2)

plt.legend();
#plt.grid();
plt.yscale('log')
#plt.ylim([10**(-5),10**(-1)])
plt.text(2,10**(-4)/2,'(d)',fontsize = 12)
plt.xlabel('epoch',fontsize=12)
plt.ylabel(r'$\Delta F$',fontsize=12)
plt.xticks([0,10,20,30,40,50])
plt.xlim([0,50])




#################
######

ac=30;
Epoch = np.linspace(0,ac,ac)
plt.style.use('science')
data = np.load('classification1.npz')
Tra_loss_batch=data['Tra_loss_batch']
Tst_loss_batch=data['Tst_loss_batch']
Tra_acc_batch=data['Tra_acc_batch']
Tst_acc_batch=data['Tst_acc_batch']
tra_loss_list=data['tra_loss_list']
tst_loss_list=data['tst_loss_list']
tra_acc_list=data['tra_acc_list']
tst_acc_list=data['tst_acc_list']

plt.subplot(2,2,1)
for i in range(len(Epoch)):
    epp = []
    epp1 = []
    for j in range(len(Tra_loss_batch[i])):
        epp.append(i)
    for j in range(len(Tst_loss_batch[i])):
        epp1.append(i)

    # print(len(Tra_loss_batch[i]),len(epp))
    plt.plot(epp, Tra_loss_batch[i], lw=5, color='lightskyblue')#'lightpink'
    plt.plot(epp1, Tst_loss_batch[i], lw=5, color='lightpink',alpha = 0.6)#'silver'

plt.plot(Epoch, tra_loss_list, label='training', color='blue',lw=1.5)
plt.plot(Epoch, tst_loss_list, label='testing', color='red',lw =1.5, linestyle='--')
plt.legend();#plt.grid()
plt.title('classification')
plt.ylim([-0.02, 0.3])
plt.xlim([0, 30])
plt.text(2,0.,'(a)',fontsize = 12)
plt.xlabel('epoch',fontsize = 12)
plt.ylabel('loss',fontsize = 12)


#data = np.load('regressionRes18_16.npz')
data = np.load('RegressionRes18_16.npz')
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
plt.subplot(2,2,2)
for i in range(len(Epoch)):
    epp = []
    epp1 = []
    for j in range(len(Tra_loss_batch[i])):
        epp.append(i)
    for j in range(len(Tst_loss_batch[i])):
        epp1.append(i)

    # print(len(Tra_loss_batch[i]),len(epp))
    plt.plot(epp, Tra_loss_batch[i], lw=7, color='royalblue')
    plt.plot(epp1, Tst_loss_batch[i], lw=8,color='grey',alpha = 0.6)

plt.title('regression')
plt.plot(Epoch, tra_loss_list, label='training', color='blue')
plt.plot(Epoch, tst_loss_list, label='testing', color='black', linestyle='--')
plt.yscale('log')
plt.legend();#plt.grid()
plt.ylim([0.01, 150])
plt.xlim([0,ac])
plt.xticks([0,10,20,30,40,50])
plt.text(2,0.02,'(b)',fontsize = 12)
plt.xlabel('epoch',fontsize = 12)
plt.ylabel('loss',fontsize = 12)

