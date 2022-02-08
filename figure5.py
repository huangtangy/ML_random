import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


Epoch = np.linspace(0,50,50)
plt.style.use('science')
oc= 30

plt.subplot(2,2,1)
Acc_Ave1=[];Acc_error=[]
Loss_Ave1 =[];Loss_error=[]

Wf = [0.2,0.4,0.6,0.8]
for i in range(4):
    data = np.load('classification_wf0' + str(2 * (i + 1)) + ' (1).npz')  #

    #data = np.load('classwf0'+str(2*(i+1))+'.npz')#
    tra_loss_list = data['tra_loss_list'];tst_loss_list = data['tst_loss_list']
    tra_acc_list = data['tra_acc_list'];tst_acc_list = data['tst_acc_list']
    Acc_ave=np.sum(tst_acc_list[oc:]) / len(tst_acc_list[oc:])
    Acc_error.append([np.abs(np.min(tst_acc_list[oc:])-Acc_ave), np.max(tst_acc_list[oc:])-Acc_ave])
    Acc_Ave1.append(Acc_ave)

    loss_ave = np.sum(tst_loss_list[oc:]) / len(tst_loss_list[oc:])
    Loss_error.append([np.abs(np.min(tst_loss_list[oc:]) - loss_ave), np.max(tst_loss_list[oc:]) - loss_ave])
    Loss_Ave1.append(loss_ave)

    lab = [r'$\omega = 0.2$', r'$\omega = 0.4$', r'$\omega = 0.6$', r'$\omega = 0.8$']
    #plt.plot(Epoch, tst_acc_list, label=lab[i], lw=1);plt.ylabel('Accuracy', fontsize=12)
    plt.plot(Epoch, tst_loss_list, lw=1,label=lab[i]);
    plt.plot(Epoch, tra_loss_list, lw=0.6, linestyle='--');
    #plt.ylim([10 ** (-5), 10 ** (-1) * 2])
    plt.xticks([0, 10, 20, 30]);
    plt.xlabel('Epoch', fontsize=12);plt.ylabel('Loss', fontsize=12)
    #plt.legend(ncol =2,loc="upper right",frameon=True,fontsize = 8)
plt.grid()
#plt.title(r'$class ~for ~diff ~\omega_f$')
plt.legend()

error = np.array(Acc_error)#[0.001,0.002,0.003,0.004]
error_acc1 = np.array([error[0], error[1], error[2], error[3]]).T

error = np.array(Loss_error)#[0.001,0.002,0.003,0.004]
error_loss1 = np.array([error[0], error[1], error[2], error[3]]).T


plt.subplot(2,2,2)
Epoch = np.linspace(0,30,30)
oc= 20
Acc_Ave=[];Acc_error=[]
Loss_Ave=[];Loss_error=[]
Tf = [1,2,3,4]
for i in range(4):
    data = np.load('classification_tf'+str(i+1)+'.npz')
    tra_loss_list = data['tra_loss_list']
    tst_loss_list = data['tst_loss_list']
    tra_acc_list = data['tra_acc_list']
    tst_acc_list = data['tst_acc_list']

    Acc_ave = np.sum(tst_acc_list[oc:]) / len(tst_acc_list[oc:])
    Acc_error.append([np.abs(np.min(tst_acc_list[oc:])-Acc_ave), np.max(tst_acc_list[oc:])-Acc_ave])
    Acc_Ave.append(Acc_ave)

    loss_ave = np.sum(tst_loss_list[oc:]) / len(tst_loss_list[oc:])
    Loss_error.append([np.abs(np.min(tst_loss_list[oc:]) - loss_ave), np.max(tst_loss_list[oc:]) - loss_ave])
    Loss_Ave.append(loss_ave)

    lab = [r'$t_f = 1$', r'$t_f= 2$', r'$t_f = 3$', r'$t_f = 4$']
    #plt.plot(Epoch, tst_acc_list, label=lab[i], lw=1);plt.ylabel('Accuracy', fontsize=12)
    plt.plot(Epoch, tst_loss_list,label=lab[i], lw=1);
    plt.plot(Epoch, tra_loss_list, lw=0.6,linestyle='--');
    plt.ylabel('Loss', fontsize=12)
    #plt.ylim([10 ** (-5), 10 ** (-1) * 2])
    plt.xticks([0, 10, 20, 30]);plt.xlabel('Epoch', fontsize=12);
    # plt.title(r'$Testing$')
plt.grid()
#plt.title(r'$class ~for ~diff ~t_f$')
plt.legend()


error = np.array(Acc_error)#[0.001,0.002,0.003,0.004]
error_acc = np.array([error[0], error[1], error[2], error[3]]).T

errorF = np.array(Loss_error)#[0.001,0.002,0.003,0.004]
error_loss = np.array([errorF[0], errorF[1], errorF[2], errorF[3]]).T


plt.subplot(2,2,3)
Wf = [0.2,0.4,0.6,0.8]

FF_Ave1 =[];F_error1 = []# np.zeros(shape=(4,2))
FLoss_Ave1 =[];FLoss_error1 = []# np.zeros(shape=(4,2))

for i in range(4):
    data = np.load('RegressionReswf10'+str(2*(i+1))+'.npz')
    tra_loss_list = data['tra_loss_list'];tst_loss_list = data['tst_loss_list'];
    tra_F_list = data['tra_F_list'];tst_F_list = data['tst_F_list']
    lab = [r'$\omega = 0.2$',r'$\omega = 0.4$',r'$\omega = 0.6$',r'$\omega = 0.8$']
    F_ave = np.sum(tst_F_list[oc:]) / len(tst_F_list[oc:])
    if i ==2:
        F_error1.append([np.abs(np.min(tst_F_list[oc:])-F_ave), np.max(tst_F_list[oc:])-F_ave])
        print(tst_F_list[oc:],np.min(tst_F_list[oc:]))
    else:
        F_error1.append([np.abs(np.min(tst_F_list[oc:]) - F_ave), np.max(tst_F_list[oc:]) - F_ave])


    #print(oc,tst_F_list[oc:])

    loss_ave = np.sum(tst_loss_list[oc:]) / len(tst_loss_list[oc:])
    FLoss_error1.append([np.abs(np.min(tst_loss_list[oc:] )- loss_ave), np.max(tst_loss_list[oc:]) - loss_ave])
    FLoss_Ave1.append(loss_ave)

    FF_Ave1.append(F_ave)
    lab = [r'$\omega = 0.2$', r'$\omega = 0.4$', r'$\omega = 0.6$', r'$\omega = 0.8$']
    #plt.plot(Epoch, tst_F_list, label=lab[i], lw=1);plt.ylabel('Fidelity Difference', fontsize=12)
    plt.plot(Epoch, tst_loss_list, label=lab[i], lw=1);
    plt.plot(Epoch, tra_loss_list, lw=0.6, linestyle='--');
    plt.ylabel('Loss', fontsize=12)
    #plt.ylim([10 ** (-5), 10 ** (-1) * 2])# plt.title(r'$Testing$')
    plt.xticks([0, 10, 20, 30]);plt.xlabel('Epoch', fontsize=12);plt.legend();plt.yscale('log')
plt.grid()
#plt.title(r'$regress ~for ~diff ~\omega_f$')
plt.legend()

error = np.array(F_error1)#[0.001,0.002,0.003,0.004]
error1 = np.array([error[0], error[1], error[2], error[3]]).T

error = np.array(FLoss_error1)#[0.001,0.002,0.003,0.004]
error_Floss1 = np.array([error[0], error[1], error[2], error[3]]).T



plt.subplot(2,2,4)
oc= 30
Tf = [1,2,3,4]
FF_Ave =[];F_error = []# np.zeros(shape=(4,2))
FLoss_Ave =[];FLoss_error = []# np.zeros(shape=(4,2))
Epoch = np.linspace(0,50,50)
for i in range(4):
    data = np.load('RegressionRestf'+str(i+1)+'.npz')
    tra_loss_list = data['tra_loss_list'];tst_loss_list = data['tst_loss_list']
    tra_F_list = data['tra_F_list'];tst_F_list = data['tst_F_list']
    lab = [r'$t_f = 1.2$',r'$t_f = 1.4$',r'$t_f = 1.6$',r'$t_f = 1.8$']
    F_ave = np.sum(tst_F_list[oc:])/len(tst_F_list[oc:])
    
    F_error.append([np.abs(np.min(tst_F_list[oc:])-F_ave), np.max(tst_F_list[oc:])-F_ave])

    loss_ave = np.sum(tst_loss_list[oc:]) / len(tst_loss_list[oc:])
    FLoss_error.append([np.abs(np.min(tst_loss_list[oc:])- loss_ave), np.max(tst_loss_list[oc:]) - loss_ave])
    FLoss_Ave.append(loss_ave)

    FF_Ave.append(F_ave)
    lab = [r'$t_f = 1$', r'$t_f= 2$', r'$t_f = 3$', r'$t_f = 4$']
    plt.plot(Epoch, tst_loss_list, label=lab[i], lw=1);
    plt.plot(Epoch, tra_loss_list, lw=0.6, linestyle='--');
    plt.ylabel('Loss', fontsize=12)
    #plt.plot(Epoch, tst_F_list, label=lab[i], lw=1)#plt.ylabel('Fidelity Difference', fontsize=12)
    plt.xticks([0, 10, 20, 30])#plt.ylim([10 ** (-5), 10 ** (-1) * 2]);
    plt.xlabel('Epoch', fontsize=12);plt.legend();plt.yscale('log');
plt.grid()
#plt.title(r'$regress ~for ~diff ~t_f$')
plt.legend()

error = np.array(F_error)#[0.001,0.002,0.003,0.004]
error = np.array([error[0], error[1], error[2], error[3]]).T


errorF = np.array(FLoss_error)#[0.001,0.002,0.003,0.004]
error_Floss = np.array([errorF[0], errorF[1], errorF[2], errorF[3]]).T











####################
fig = plt.figure()

ax1 = fig.add_subplot(121)
plt.errorbar(Tf,Acc_Ave,yerr=error_acc, fmt='r-^', capsize=4, capthick=1,lw = 0.5,markersize=4)
ax1.set_ylabel('accuracy',fontsize=14)
ax1.set_xlabel(r'$t_f/\omega_0$',fontsize=16,color='red')
for tl in ax1.get_xticklabels():
    tl.set_color('r')

ax2 = ax1.twiny()  # this is the important function
plt.errorbar(Wf,Acc_Ave1,yerr=error_acc1, fmt='b-o', capsize=4, capthick=1,lw = 0.5,markersize=4)
#ax2.set_xlabel(r'$t_f$',fontsize=15)
#ax2.set_xlabel(r'$\omega_f$',fontsize=15)
for tl in ax2.get_xticklabels():
    tl.set_color('b')
#plt.ylim([10**(-4),10**(-2)])
plt.ylim([0.9,1.01])
plt.yticks([0.92,0.96,1])
plt.text(0.2,0.91,r'(a)',fontsize = 14)
plt.text(0.45,1.027,r'$\omega_f/\omega_0$',fontsize = 15,color='blue')
#plt.grid()


ax1 = fig.add_subplot(122)
plt.errorbar(Tf,FF_Ave,yerr=error, fmt='r^-', capsize=4, capthick=1,lw = 0.5,markersize=4)
ax1.set_ylabel(r'$\Delta F$',fontsize=14)
ax1.set_xlabel(r'$t_f/\omega_0$',fontsize=16,color='red')
ax1.set_yscale('log')
for tl in ax1.get_xticklabels():
    tl.set_color('r')

ax2 = ax1.twiny()  # this is the important function
plt.errorbar(Wf,FF_Ave1,yerr=error1, fmt='bo-', capsize=4, capthick=1,lw = 0.5,markersize=4)
#ax2.set_xlabel(r'$t_f$',fontsize=15)
#ax2.set_xlabel(r'$\omega_f$',fontsize=15)

for tl in ax2.get_xticklabels():
    tl.set_color('b')
plt.ylim([10**(-4),10**(-1)])
plt.text(0.2,10**(-4)/6,r'(b)',fontsize = 14)
plt.text(0.45,0.45,r'$\omega_f/\omega_0$',fontsize = 15,color='blue')
plt.yticks([10**(-5),10**(-3),10**(-1)])
#plt.grid()


plt.show()


####################
fig = plt.figure()

ax1 = fig.add_subplot(121)
plt.errorbar(Tf,Loss_Ave,yerr=error_loss, fmt='r-^', capsize=4, capthick=1,lw = 0.5,markersize=4)
ax1.set_ylabel('loss',fontsize=12)
ax1.set_xlabel(r'$t_f$',fontsize=15,color='red')
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
plt.text(0.34,0.25,r'$Classification$',fontsize = 12)
plt.text(0.2,0.02,'(a)',fontsize = 14)
plt.text(0.45,0.235,r'$\omega_f$',fontsize = 15,color='blue')
#plt.grid()

ax1 = fig.add_subplot(122)
plt.errorbar(Tf,FLoss_Ave,yerr=error_Floss, fmt='r-^', capsize=4, capthick=1,lw = 0.5,markersize=4)
ax1.set_ylabel('Loss',fontsize=12)
ax1.set_xlabel(r'$t_f$',fontsize=15,color='red')
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
plt.text(0.34,500,r'$Regression$',fontsize = 12)
plt.text(0.22,1/6,'(b)',fontsize = 14)
plt.text(0.45,260,r'$\omega_f$',fontsize = 15,color='blue')
#plt.yticks([10**(-5),10**(-4),10**(-3),10**(-2)])
#plt.grid()
plt.show()