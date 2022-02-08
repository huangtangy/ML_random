import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



N = 40

# for different filter number
Epoch = np.linspace(0,40,N)
lab = [r'$N_f = 4$',r'$N_f = 8$',r'$N_f = 16$',r'$N_f = 32$']
lsty = ['-','--','-.',':']
cl1 = ['blue','red','black','green']
cl2 = ['cornflowerblue','lightsalmon','grey','palegreen']
ii = ['10','18','34','50']
jj= ['4','8','16','32']
plt.figure()
plt.style.use('science')
plt.subplot(1,2,1)
for i in range(4):
    data = np.load('regression34_'+jj[i]+'.npz')
    tra_loss_list = data['tra_loss_list'][:N];
    tst_loss_list = data['tst_loss_list'][:N];

    plt.plot(Epoch,tst_loss_list[:N],lw = 0.7,color = cl1[i])
    plt.plot(Epoch, tra_loss_list[:N], lw=3,label=lab[i],color = cl2[i],linestyle='-')
    plt.text(2,10**(0)/5,'(a)',fontsize = 14)
    plt.legend()
    plt.yscale('log')
    plt.ylim([0.1,100])
    plt.xlim([-2, 42])
    plt.yticks([ 0.1,1, 10,100],fontsize = 14)
    plt.xticks([0, 10, 20, 30, 40],fontsize = 14)
    plt.xlabel('Epoch',fontsize=14)
    plt.ylabel('Loss',fontsize=14)



plt.subplot(1,2,2)

lab =[r'$ResNet10$',r'$ResNet18$',r'$ResNet34$',r'$ResNet50$']
lsty = ['-','--']
for i in range(4):
    data = np.load('regression'+ii[i]+'_8.npz')
    tra_loss_list = data['tra_loss_list'][:N];
    tst_loss_list = data['tst_loss_list'][:N];
    tra_F_list = data['tra_F_list'];
    tst_F_list = data['tst_F_list']
    plt.plot(Epoch,tst_loss_list[:N],lw = 0.7,color = cl1[i])
    plt.plot(Epoch, tra_loss_list[:N], label=lab[i], lw=3,color = cl2[i],linestyle='-')#, linestyle=lsty[i]
    plt.legend()
    plt.text(2,10**(0)/5,'(b)',fontsize = 14)
    plt.legend()
    plt.yscale('log')
    plt.xlim([-2,42])
    plt.ylim([0.1,100])
    plt.yticks([0.1,1, 10,100],fontsize = 14)
    plt.xticks([0, 10, 20, 30, 40],fontsize = 14)
    plt.xlabel('Epoch',fontsize=14)
    plt.ylabel('Loss',fontsize=14)






#################
Epoch = np.linspace(0,50,50)
Lav=np.zeros(shape=[4,4])
Lav1=np.zeros(shape=[4,4])
ii = [10,18,34,50]
jj= [4,8,16,32]

for i in range(4):
    for j in range(4):
        data = np.load('regression'+str(ii[j])+'_'+str(jj[i])+'.npz')
        tra_loss_list = data['tra_loss_list'];
        tst_loss_list = data['tst_loss_list'];
        '''
        plt.plot(Epoch,tra_loss_list)
        plt.yscale('log')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        '''
        Lav[i,j]=(np.sum(tst_loss_list[40:])/10)
        Lav1[i,j] = (np.sum(tra_loss_list[40:])/10)
        print(Lav[i,j],'Res',ii[j],'filter',jj[i])

ii = [1,2,3,4]
jj= [1,2,3,4]
cl = ['tomato','grey','lightblue','palegreen']
x,y = np.meshgrid(ii,jj)
fig = plt.figure()
ax = fig.add_subplot(122, projection='3d')
ax.set_title('(b)',fontsize=14)
for i in range(4):
    ax.bar3d(x[i],y[i],0,dx=0.4,dy=0.4,dz=Lav[i],color=cl[i])
ax.set_ylabel(r'$N_f$',fontsize=14)
ax.set_xticks([0.2,1.2,2.2,3.2])
ax.set_xticklabels(['ResNet10','ResNet18','ResNet34','ResNet50',],fontsize=13)
ax.set_yticks([4.2,3.2,2.2,1.2])
ax.set_yticklabels(['32','16','8','4'],fontsize=14)
ax.set_zticks([0,0.2,0.4,0.6,0.8])
ax.set_zticklabels(['0','0.2','0.4','0.6','0.8'],fontsize=14)
ax.set_zlabel('Loss',fontsize=14,rotation = 90)


ii = [1,2,3,4]
jj= [1,2,3,4]
cl = ['tomato','grey','lightblue','palegreen']
x,y = np.meshgrid(ii,jj)
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_title('(a)',fontsize=14)
for i in range(4):
    ax1.bar3d(x[i],y[i],0,dx=0.4,dy=0.4,dz=Lav1[i],color=cl[i])
ax1.set_xticks([0.2,1.2,2.2,3.2])
ax1.set_xticklabels(['ResNet10','ResNet18','ResNet34','ResNet50'],fontsize=13)
ax1.set_ylabel(r'$N_f$',fontsize=14)
ax1.set_yticks([4.2,3.2,2.2,1.2])
ax1.set_yticklabels(['32','16','8','4'],fontsize=14)
ax1.set_zticks([0,0.4,0.8])
ax1.set_zticklabels(['0','0.1','0.2'],fontsize=14)
ax1.set_zlabel('Loss',fontsize=14,rotation = 90)


plt.figure()

ii = [1,2,3,4,5,6]
jj= [1,2,3,4,5,6]
Lav11 = np.array([[0.18977084, 0.18384933,0.17,0.16,0.15, 0.13720124],
                 [0.17977084, 0.15384933,0.17,0.16,0.15, 0.13720124],
                 [0.137084, 0.14384933,0.15,0.16,0.15, 0.13720124],
                 [0.137084, 0.14384933,0.15,0.16,0.15, 0.13720124],
                 [0.08050653, 0.0851232 ,0.09,0.1,0.11, 0.12746158],
                 [0.06256927, 0.07511215,0.08,0.12,0.13, 0.14874139]])
ii = [1,2,3,4]
jj= [1,2,3,4]
Lav0=np.array([[0.7764325 , 0.51054909, 0.46981179,],
               [0.45303052, 0.51841733, 0.64481904],
               [0.4424761 , 0.4775482 , 0.59543887]])
x,y = np.meshgrid(ii,jj)
quadmesh=plt.pcolormesh(x, y, Lav1, cmap='bwr')  # Oranges#Greys
quadmesh.set_clim(vmin=0, vmax=0.8)
plt.colorbar(quadmesh)
'''
quadmesh.set_xticklabels(['ResNet10','ResNet18','ResNet34','ResNet50'],fontsize=13)
quadmesh.set_xticks([0.2,1.2,2.2,3.2])
quadmesh.set_yticks([4.2,3.2,2.2,1.2])
quadmesh.set_yticklabels(['32','16','8','4'],fontsize=14)

'''

'''
plt.figure()

data = np.load('regression18_8.npz')
tra_loss_list = data['tra_loss_list'];
tst_loss_list = data['tst_loss_list'];
plt.plot(Epoch,tra_loss_list)
plt.plot(Epoch, tra_loss_list[:N], label=lab[i], lw=3,color = 'grey')
plt.plot(Epoch,tst_loss_list[:N],lw = 1,color = 'black')
plt.yscale('log')
plt.xlabel('Epoch',fontsize=14)
plt.ylabel('Loss',fontsize=14)
plt.text(80, 30, '(b)', fontsize=14)
plt.ylim([0.01, 100])
plt.yticks([0.01, 0.1,1, 10,100],fontsize = 14)
plt.xticks([0, 10, 20, 30, 40,50],fontsize = 14)
plt.text(-70,30, '(a)', fontsize=14)
#plt.figure()
#quadmesh =plt.pcolormesh(x,y,Lav)
#quadmesh.set_clim(vmin=0.01, vmax=0.3)
#plt.colorbar()
'''
