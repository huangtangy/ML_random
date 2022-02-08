
import numpy as np
import matplotlib.pyplot as plt
from initialization import *
from function_sets import *
import scipy.io


def get_max_omega(tf,omega_f):
    dt = 0.01;Nt= np.int(tf/dt)
    T = np.linspace(0,tf,Nt)
    M = 30;M1 = 100;N_M = 200;N_M1 =200;
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




A1_set,A2_set,W_set = get_max_omega(1,0.1)

data = np.load('./data40000_tf1_wf01.npz')
S = data['S'];A1 = data['A1'];A2=data['A2'];
F = data['F'];RH=data['RH']
print('tf = 1,wf=0.1',sum(RH[:32000]))
F1 = scipy.io.loadmat('Ftf1.mat')
F=F1['FF']
a1 = np.linspace(-30,30,200)
a2 = np.linspace(-100,100,200)
A11,A22 = np.meshgrid(a1,a2)


print('$\omega_f = 0.1,~t_f = 1$')


plt.figure()
FS = 18
plt.style.use('science')

plt.subplot(1,3,1)
plt.title(r'$\omega_f = 0.1,~t_f = 1$',fontsize = FS-2)
quadmesh=plt.pcolormesh(A11,A22,F,cmap= 'bwr')#Oranges#Greys
quadmesh.set_clim(vmin=0, vmax=1)
plt.colorbar()

C = plt.contour(A1_set,A2_set, W_set,[6], colors='black',lw=0.2,linestyles=['--'])
#plt.clabel(C, inline=True, fontsize=10)
#C = plt.contour(A1,A2,F,[0.6], colors='black',lw=0.2,linestyles=['--'])

plt.scatter(A1[:40000],A2[:40000],marker='x',color='black' ,lw =0.5,s=20)
#plt.scatter(A1[6],A2[6],marker='o',color='black' ,lw = 1)#,facecolors='none',edgecolors='black')
plt.text(18,65,'(a)',fontsize=14,color='black',backgroundcolor='0.9')

plt.xlabel(r'$a_1$',fontsize = FS)
#plt.ylabel(r'$a_2$',fontsize = FS)
plt.xticks([-30,0,30])
plt.yticks([-100,-50,0,50,100])


#####################


A1_set,A2_set,W_set = get_max_omega(1,0.8)

data = np.load('./data_16000_wf04_09.npz')
S = data['S'];A1 = data['A1'];A2=data['A2'];
F = data['F'];F=F**2;RH= data['CC']
print('wf = 2',sum(RH))

F1 = scipy.io.loadmat('Fwf04.mat')
F=F1['FF']
a1 = np.linspace(-30,30,200)
a2 = np.linspace(-100,100,200)
A11,A22 = np.meshgrid(a1,a2)

plt.subplot(1,3,2)
plt.title('$\omega_f = 0.4,~t_f = 1$',fontsize = FS-2)
quadmesh=plt.pcolormesh(A11,A22,F,cmap= 'bwr')#Oranges#Greys
quadmesh.set_clim(vmin=0, vmax=1)
plt.colorbar()

plt.scatter(A1,A2,marker='x',color='black' ,lw =0.5,s=20)
C = plt.contour(A1_set,A2_set, W_set,[6], colors='black',lw=0.2,linestyles=['--'])
#plt.clabel(C, inline=True, fontsize=10)
#C = plt.contour(A1,A2,F,[0.6], colors='black',lw=0.2,linestyles=['--'])


#plt.scatter(A1[6],A2[6],marker='o',color='black' ,lw = 1)#,facecolors='none',edgecolors='black')
plt.text(18,65,'(b)',fontsize=14,color='black',backgroundcolor='0.9')

plt.xlabel(r'$a_1$',fontsize = FS)
#plt.ylabel(r'$a_2$',fontsize = 16)
plt.xticks([-30,0,30])
plt.yticks([-100,-50,0,50,100])





#####################


A1_set,A2_set,W_set = get_max_omega(2,0.1)

data = np.load('./data32000_tf2.npz')
S = data['S']
A1 = data['A1'];
A2=data['A2'];
F = data['F']
RH = data['RH']
AA1 =[];AA2 =[];

for i in range(len(RH)):
    if RH[i]==1:
        AA1.append(A1[i])
        AA2.append(A2[i])



print('tf = 2',sum(RH))
F1 = scipy.io.loadmat('Ftf_2.mat')
F=F1['FF']
a1 = np.linspace(-30,30,400)
a2 = np.linspace(-100,100,400)
A11,A22 = np.meshgrid(a1,a2)


plt.subplot(1,3,3)
plt.title('$\omega_f = 0.1,~t_f = 2$',fontsize = FS-2)
quadmesh=plt.pcolormesh(A11,A22,F,cmap= 'bwr')#Oranges#Greys
quadmesh.set_clim(vmin=0, vmax=1)
plt.colorbar()
plt.scatter(A1,A2,marker='x',color='black' ,lw =0.5,s=20)
#plt.scatter(AA1,AA2,marker='x',color='black' ,lw =0.5,s=20)

C = plt.contour(A1_set,A2_set, W_set,[6], colors='black',lw=0.2,linestyles=['--'])

plt.text(9,20,'(c)',fontsize=14,color='black',backgroundcolor='0.9')

plt.xlabel(r'$a_1$',fontsize = FS)
#plt.ylabel(r'$a_2$',fontsize = FS)
plt.xlim([-15,15])
plt.ylim([-30,30])
plt.xticks([-15,0,15])
plt.yticks([-30,0,30])


'''

plt.figure()
lend = ['(a)','(b)','(c)','(d)']
ld1 = [r'$t_f = 1$',r'$t_f = 2$',r'$t_f = 3$',r'$t_f = 4$']

for i in range(4):
    plt.subplot(2,2,i+1)

    A1_set, A2_set, W_set = get_max_omega(i+1, 0.1)

    data = np.load('./data32000_tf'+str(i+1)+'.npz')
    S = data['S'];
    A1 = data['A1'];
    A2 = data['A2'];
    F = data['F'];
    RH = data['RH']
    print('tf = 1,wf=0.1', sum(RH[:32000]))
    F1 = scipy.io.loadmat('Ftf'+str(i+1)+'.mat')
    F = F1['FF']
    a1 = np.linspace(-30, 30, 200)
    a2 = np.linspace(-100, 100, 200)
    A11, A22 = np.meshgrid(a1, a2)

    #print('$\omega_f = 1,~t_f = %s$'(i+1))

    plt.style.use('science')

    plt.title(r'$t_f = %s$')
    quadmesh = plt.pcolormesh(A11, A22, F, cmap='bwr')  # Oranges#Greys
    quadmesh.set_clim(vmin=0, vmax=1)
    plt.colorbar()

    C = plt.contour(A1_set, A2_set, W_set, [6], colors='black', lw=0.2, linestyles=['--'])
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