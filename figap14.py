import numpy as np
import matplotlib.pyplot as plt
from initialization import *
from function_sets import *
import scipy.io

def get_max_omega():
    from initialization import T,tf,dt,omega_f
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









###
plt.figure()
plt.style.use('science')

FS = 18
F = scipy.io.loadmat('Ftf1.mat')
dbtf = scipy.io.loadmat('dBtf.mat')
btf = scipy.io.loadmat('Btf.mat')
FF=F['FF']
dB=dbtf['dB']
B=btf['B']
a1 = np.linspace(-30,30,200)
a2 = np.linspace(-100,100,200)
A11,A22 = np.meshgrid(a1,a2)


plt.subplot(1,3,1)
plt.title(r'$b(t_f)$',fontsize = FS-2)
quadmesh=plt.pcolormesh(A11,A22,B,cmap= 'bwr')#Oranges#Greys
quadmesh.set_clim(vmin=0, vmax=3)
plt.colorbar()
plt.text(16,70,'(a)',fontsize=16,color='black',backgroundcolor='0.9')
plt.xlabel(r'$a_1$',fontsize = FS)
plt.ylabel(r'$a_2$',fontsize = FS)
plt.xticks([-30,0,30])
plt.yticks([-100,-50,0,50,100])


plt.subplot(1,3,2)

plt.title(r'$\dot{b}(t_f)$',fontsize = FS-2)
quadmesh1=plt.pcolormesh(A11,A22,dB,cmap= 'bwr')#Oranges#Greys
quadmesh1.set_clim(vmin=0, vmax=10)
plt.colorbar()
plt.text(16,70,'(b)',fontsize=16,color='black',backgroundcolor='0.9')

plt.xlabel(r'$a_1$',fontsize = FS)
#plt.ylabel(r'$a_2$',fontsize = FS)
plt.xticks([-30,0,30])
plt.yticks([-100,-50,0,50,100])

plt.subplot(1,3,3)
plt.title('fidelity',fontsize = FS-2)
quadmesh2=plt.pcolormesh(A11,A22,FF,cmap= 'bwr')#Oranges#Greys
quadmesh2.set_clim(vmin=0, vmax=1)
plt.colorbar()
plt.text(16,70,'(c)',fontsize=16,color='black',backgroundcolor='0.9')

plt.xlabel(r'$a_1$',fontsize = FS)
#plt.ylabel(r'$a_2$',fontsize = FS)
plt.xticks([-30,0,30])
plt.yticks([-100,-50,0,50,100])






#####################################


data = np.load('./data40000_tf1_wf01.npz')
S = data['S']
A1 = data['A1'];A2=data['A2'];F=data['F'];
i = 400
Si = np.linspace(0,i,i)
a1 = A1[:i];a2 = A2[:i];FF = F[:i]


i=0
z = np.zeros(shape=(160,160))
Se = S[i]
for i in range(len(Se)):
    for j in range(len(Se)):
        z[i][j] = Se[i]+Se[j]


plt.figure()
plt.style.use('science')

xx = np.linspace(0,160,160)
yy = np.linspace(0,160,160)
x,y = np.meshgrid(xx,yy)


plt.subplot(2,2,4)
plt.title(r'$2D~ grid$',fontsize=10)
plt.pcolormesh(x,y,z,cmap= 'Oranges')#Oranges#Greys
plt.xlabel(r'$S$',fontsize = 12)
plt.ylabel(r'$S$',fontsize = 12)
plt.text(10,12,'(d)',fontsize=12,color='black',backgroundcolor='0.9')#,color='white'
plt.xticks([0,80,160])
plt.yticks([0,80,160])



plt.subplot(2,2,1)
A1_set,A2_set,W_set = get_max_omega()

plt.title(r'$\omega_{\max}^2(t)$')
#plt.contourf(A1_set,A2_set, W_set, 10, alpha=0.75, cmap=plt.cm.hot)
plt.pcolormesh(A1_set,A2_set, W_set,cmap= 'hot')#Oranges#Greys
plt.colorbar()
#plt.scatter(a1,a2,ww,marker='o')
C = plt.contour(A1_set,A2_set, W_set,[6], colors='silver',lw=0.2,linestyles=['--'])
plt.clabel(C, inline=True, fontsize=10)
#plt.xlim([-30,10]);plt.ylim([-50,75])
plt.text(-28,-88,'(a)',fontsize=14,color='black')#,backgroundcolor='0.9'
plt.xlabel(r'$a_1$', fontsize=14);
plt.ylabel(r'$a_2$', fontsize=14)
plt.xticks([-30,0,30])
plt.yticks([-100,-50,0,50,100])









##############################################

FB=np.linspace(0.1,1,9)
#num_hight=[0.70775, 0.6081, 0.537325, 0.479525, 0.4287, 0.383325, 0.336725, 0.283625, 0.20965]
#HR= [0.580425, 0.52455, 0.48125, 0.442725, 0.406775, 0.370875, 0.330325, 0.28125, 0.209075]

num_hight =[];HR = [];

for i in range(9):
    data = np.load('data_40000_all_0'+str(i+1)+'.npz')
    F = data['F'];R = data['RC']
    jj =0;jjj=0;
    for j in range(len(F)):
        if F[j] >0.1*(i+1):
            jj+=1
            if R[j] ==1:
                jjj+=1
    num_hight.append(jj/len(F))
    HR.append(jjj/len(R))

plt.subplot(2,2,2)
#plt.style.use('science')
plt.plot(FB,num_hight,label='F',color='black',Marker='s')
plt.plot(FB,1-np.array(num_hight),label='uF',linestyle='--',color='grey',lw = 1,Marker='o')
#plt.plot(FB,RC_cell,label='R',color='blue')
#plt.plot(FB,1-np.array(RC_cell),label='uR',color='royalblue',lw = 2,linestyle='--')
plt.plot(FB,HR,label='FR',color='blue',Marker='o')
plt.plot(FB,1-np.array(HR),label='anti-FR',color='royalblue',lw = 1,linestyle='--',Marker='s')
plt.legend(ncol =2,loc="upper center",frameon=True,fontsize = 8);
plt.xlabel(r'$F_b$',fontsize = 12)
plt.ylabel(r'ratio',fontsize = 12)
plt.text(0.05,0.08,'(b)',fontsize=14,color='black')#,color='white',backgroundcolor='0.9'
#plt.grid()
plt.xlim([0,1])
plt.ylim([0,1])


plt.subplot(2,2,3)
#plt.style.use('science')
plt.scatter(Si,FF**2,c= FF**2,marker='o',facecolors='none',edgecolors='black',cmap='bwr',lw=0.5)#,cmap='Greys'coolwarm
#plt.scatter(Si,a1,marker='o',facecolors='none',edgecolors='royalblue')
#plt.scatter(Si,a2,marker='o',facecolors='none',edgecolors='tomato')
plt.ylim([0,1])
plt.xlim([0,400])
plt.xticks([0,200,400])
#plt.grid()
plt.ylabel(r'$F_i^{\max}$',fontsize = 12)
plt.xlabel('realizations',fontsize = 12)
plt.text(20,0.1,'(c)',fontsize=14,color='black',backgroundcolor='0.9')#,color='white'




#####





