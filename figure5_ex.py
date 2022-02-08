import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.io

def get_max_omega(tf,omega_f):
    dt = 0.01;Nt= np.int(tf/dt)
    T = np.linspace(0,tf,Nt)
    M = 30;M1 = 100;N_M = 400;N_M1 =400;
    A1_set = np.zeros(shape=(N_M, N_M1), dtype=float)
    A2_set = np.zeros(shape=(N_M, N_M1), dtype=float)
    W_set = np.zeros(shape=(N_M, N_M1), dtype=float)
    A1 = np.linspace(-M, M, N_M);
    A2 = np.linspace(-M1, M1, N_M1);

    for i in range(N_M):
        #print(i)
        a1 = A1[i];
        for j in range(N_M1):
            a0 = 1;
            a2 = A2[j];
            a3 = (omega_f - (a0 + a1 * tf + a2 * tf ** 2)) / tf ** 3;
            omega_t2 = (1 + a1 * T + a2 * T ** 2 + a3 * T ** 3)**2
            #omega_max = np.sum(omega_t2)*dt
            omega_max = np.max(omega_t2)
            A1_set[i][j] = a1
            A2_set[i][j] = a2
            W_set[i][j] = omega_max
    return A1_set,A2_set,W_set
'''
Epoch = np.linspace(0, 30, 30)
plt.style.use('science')
oc = 30
clo = ['black','blue','green','red']

Acc_Ave1 = [];
Acc_error = []
Loss_Ave1 = [];
Loss_error = []
plt.figure(1)
Wf = [0.2, 0.4, 0.6, 0.8]
for i in range(4):
    data = np.load('classification_wf0' + str(2 * (i + 1)) + ' (1).npz')  #

    # data = np.load('classwf0'+str(2*(i+1))+'.npz')#
    tra_loss_list = data['tra_loss_list'];
    tst_loss_list = data['tst_loss_list']
    tra_acc_list = data['tra_acc_list'];
    tst_acc_list = data['tst_acc_list']

    lab = [r'$t_f = 1$', r'$t_f= 2$', r'$t_f = 3$', r'$t_f = 4$']
    lab1 = [r'$\omega = 0.2$', r'$\omega = 0.4$', r'$\omega = 0.6$', r'$\omega = 0.8$']
    # plt.plot(Epoch, tst_acc_list, label=lab[i], lw=1);plt.ylabel('Accuracy', fontsize=12)

    plt.subplot(2, 2, 1)
    plt.plot(Epoch, tst_loss_list[:30], lw=2, label=lab1[i],color=clo[i]);
    plt.plot(Epoch, tra_loss_list[:30], lw=1, linestyle='--',color=clo[i]);
    # plt.ylim([10 ** (-5), 10 ** (-1) * 2])
    plt.xticks([0, 10, 20, 30]);
    plt.xlabel('Epoch', fontsize=12);
    plt.ylabel('Loss', fontsize=12)
    plt.legend()


    plt.subplot(2, 2, 2)
    plt.plot(Epoch, tst_acc_list[:30], lw=2, label=lab1[i],color=clo[i]);
    plt.plot(Epoch, tra_acc_list[:30], lw=1, linestyle='--',color=clo[i]);
    # plt.ylim([10 ** (-5), 10 ** (-1) * 2])
    plt.xticks([0, 10, 20, 30]);
    plt.xlabel('Epoch', fontsize=12);
    plt.ylabel('ACC', fontsize=12)
    plt.legend()

    data = np.load('classification_tf' + str(i + 1) + '.npz')
    tra_loss_list = data['tra_loss_list']
    tst_loss_list = data['tst_loss_list']
    tra_acc_list = data['tra_acc_list']
    tst_acc_list = data['tst_acc_list']

    plt.subplot(2, 2, 3)
    plt.plot(Epoch, tst_loss_list, lw=2, label=lab[i],color=clo[i]);
    plt.plot(Epoch, tra_loss_list, lw=1, linestyle='--',color=clo[i]);
    # plt.ylim([10 ** (-5), 10 ** (-1) * 2])
    plt.xticks([0, 10, 20, 30]);
    plt.xlabel('Epoch', fontsize=12);
    plt.ylabel('Loss', fontsize=12)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(Epoch, tst_acc_list, lw=2, label=lab[i],color=clo[i]);
    plt.plot(Epoch, tra_acc_list, lw=1, linestyle='--',color=clo[i]);
    # plt.ylim([10 ** (-5), 10 ** (-1) * 2])
    plt.xticks([0, 10, 20, 30]);
    plt.xlabel('Epoch', fontsize=12);
    plt.ylabel('ACC', fontsize=12)
    plt.legend()


plt.figure(2)
Epoch = np.linspace(0, 50, 50)
plt.style.use('science')


Acc_Ave1 = [];
Acc_error = []
Loss_Ave1 = [];
Loss_error = []
plt.figure(2)
Wf = [0.2, 0.4, 0.6, 0.8]
for i in range(4):
    data = np.load('RegressionReswf10' + str(2 * (i + 1)) + '.npz') #

    # data = np.load('classwf0'+str(2*(i+1))+'.npz')#
    tra_loss_list = data['tra_loss_list'];
    tst_loss_list = data['tst_loss_list']
    tra_F_list = data['tra_F_list'];
    tst_F_list = data['tst_F_list']

    lab = [r'$t_f = 1$', r'$t_f= 2$', r'$t_f = 3$', r'$t_f = 4$']
    lab1 = [r'$\omega = 0.2$', r'$\omega = 0.4$', r'$\omega = 0.6$', r'$\omega = 0.8$']
    # plt.plot(Epoch, tst_acc_list, label=lab[i], lw=1);plt.ylabel('Accuracy', fontsize=12)
    Epoch = np.linspace(0, 30, 30)
    plt.subplot(2, 2, 1)
    plt.plot(Epoch, tst_loss_list, lw=2, label=lab1[i],color=clo[i]);
    plt.plot(Epoch, tra_loss_list, lw=1, linestyle='--',color=clo[i]);
    # plt.ylim([10 ** (-5), 10 ** (-1) * 2])
    plt.xticks([0, 10, 20, 30]);
    plt.xlabel('Epoch', fontsize=12);
    plt.ylabel('Loss', fontsize=12)
    plt.yscale('log')
    plt.legend()


    plt.subplot(2, 2, 2)
    plt.plot(Epoch, tst_F_list, lw=2, label=lab1[i],color=clo[i]);
    plt.plot(Epoch, tra_F_list, lw=1, linestyle='--',color=clo[i]);
    # plt.ylim([10 ** (-5), 10 ** (-1) * 2])
    plt.xticks([0, 10, 20, 30]);
    plt.xlabel('Epoch', fontsize=12);
    plt.ylabel(r'$\Delta F$', fontsize=12)
    plt.yscale('log')
    plt.legend()


    data = np.load('RegressionRestf' + str(i + 1) + '.npz')
    tra_loss_list = data['tra_loss_list']
    tst_loss_list = data['tst_loss_list']
    tra_F_list = data['tra_F_list']
    tst_F_list = data['tst_F_list']
    Epoch = np.linspace(0, 50, 50)
    plt.subplot(2, 2, 3)
    plt.plot(Epoch, tst_loss_list, lw=2, label=lab[i],color=clo[i]);
    plt.plot(Epoch, tra_loss_list, lw=1, linestyle='--',color=clo[i]);
    # plt.ylim([10 ** (-5), 10 ** (-1) * 2])
    plt.xticks([0, 10, 20, 30]);
    plt.xlabel('Epoch', fontsize=12);
    plt.ylabel('Loss', fontsize=12)
    plt.ylim([0.01,100])
    plt.yscale('log')
    plt.legend()


    plt.subplot(2, 2, 4)
    plt.plot(Epoch, tst_F_list, lw=2, label=lab[i],color=clo[i]);
    plt.plot(Epoch, tra_F_list, lw=1, linestyle='--',color=clo[i]);
    # plt.ylim([10 ** (-5), 10 ** (-1) * 2])
    plt.xticks([0, 10, 20, 30]);
    plt.xlabel('Epoch', fontsize=12);
    plt.ylabel(r'$\Delta F$', fontsize=12)
    plt.yscale('log')
    plt.legend()

'''

'''
plt.figure(3)
plt.style.use('science')
lend = ['(a)','(b)','(c)','(d)']
ld1 = [r'$t_f = 1$',r'$t_f = 2$',r'$t_f = 3$',r'$t_f = 4$']
for i in range(4):
    plt.subplot(2,2,i+1)
    AA1=[];AA2=[]

    A1_set, A2_set, W_set = get_max_omega(i+1, 0.1)

    data = np.load('./data32000_tf'+str(i+1)+'.npz')
    S = data['S'];
    A1 = data['A1'];
    A2 = data['A2'];
    F = data['F'];
    RH = data['RH']

    for ii in range(len(RH)):
        if RH[ii] == 1 and F[ii] > 0.9:
            AA1.append(A1[ii])
            AA2.append(A2[ii])

    print('tf = 1,wf=0.1', sum(RH))
    F1 = scipy.io.loadmat('Ftf_'+str(i+1)+'.mat')
    F = F1['FF']
    a1 = np.linspace(-30, 30, len(F))
    a2 = np.linspace(-100, 100, len(F))
    A11, A22 = np.meshgrid(a1, a2)

    #print('$\omega_f = 1,~t_f = %s$'(i+1))



    plt.title(ld1[i])
    quadmesh = plt.pcolormesh(A11, A22, F, cmap='bwr')  # Oranges#Greys
    quadmesh.set_clim(vmin=0, vmax=1)
    plt.colorbar()

    C = plt.contour(A1_set, A2_set, W_set, [6], colors='green', lw=0.2, linestyles=['--'])
    # plt.clabel(C, inline=True, fontsize=10)
    # C = plt.contour(A1,A2,F,[0.6], colors='black',lw=0.2,linestyles=['--'])

    plt.scatter(A1, A2, marker='x', color='black', lw=0.5, s=20)
    # plt.scatter(A1[6],A2[6],marker='o',color='black' ,lw = 1)#,facecolors='none',edgecolors='black')
    #plt.text(18, 65, lend[i], fontsize=14, color='black', backgroundcolor='0.9')

    plt.xlabel(r'$a_1$', fontsize=16)
    plt.ylabel(r'$a_2$', fontsize=16)

    if i ==1:
        plt.ylim(-40, 40)
        plt.xlim(-15, 15)
    elif i==2:
        plt.ylim(-15, 15)
        plt.xlim(-10, 10)
    elif i==3:
        plt.ylim(-8, 8)
        plt.xlim(-8, 8)
    else:
        plt.ylim(-100, 100)
        plt.xlim(-30, 30)


    #plt.xlim(-30/(i+1),30/(i+1))
    #plt.xticks([-30, 0, 30])
    #plt.yticks([-100, -50, 0, 50, 100])


'''



Delta_Y =[]
N_cnn2=[]
Tf=[]

def get_DY(A1,A2):


    sum1 =0
    for i in range(len(A1)):
        sum_dt = 0
        for j in range(len(A1)):
            sum_dt+=np.sqrt((A1[i]-A1[j])**2+(A2[i]-A2[j])**2)
        sum1+=sum_dt/len(A1)
        if i%500==0:
            print(i)

    return sum1/len(A1)


for i in range(4):

    AA1 = [];
    AA2 = [];
    data = np.load('./data32000_tf' + str(i + 1) + '.npz')
    S = data['S'];
    A1 = data['A1'];
    A2 = data['A2'];
    Tf.append(i + 1)

    F = data['F'];
    RH = data['RH']
    for ii in range(len(RH)):
        if RH[ii]==1 and F[ii]>0.9:
            AA1.append(A1[ii])
            AA2.append(A2[ii])

    N_cnn2.append(sum(RH))

    Delta_Y.append(get_DY(AA1,AA2))


plt.figure(3)
plt.style.use('science')
plt.subplot(1,2,1)
plt.plot(Tf,N_cnn2,'-o')
plt.xlabel(r'$t_f$',fontsize=14)
plt.ylabel(r'$N_{FH}$',fontsize=14)
plt.subplot(1,2,2)
plt.plot(Tf,Delta_Y,'-o')
plt.xlabel(r'$t_f$',fontsize=14)
plt.ylabel('Average distance',fontsize=14)
