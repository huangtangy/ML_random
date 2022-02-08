import numpy as np
import matplotlib.pyplot as plt
from initialization import *
from function_sets import *
import scipy.io
def three_order_poly(A,T):
    return A[0]+A[1]*T+A[2]*T**2+A[3]*T**3
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


def get_final_states(N_t,omega_set,S,psi_i):
    ''' time-evolution for a control sequence'''
    from initialization import x,x1,xi,U0,p,dt
    # where omega_set is \omega(t)
    # using  time-split method
    random_p = random_potential(x,x1,xi,U0,S)
    psi = psi_i
    for i in range(N_t):
        omega_t = omega_set[i]
        harmonic_p = (0.5*omega_t**2) * x ** 2
        potential_list = harmonic_p + random_p  # 0.5*omega_t**2*x**2 #
        psi = FFT(psi, p, potential_list, dt)
    return psi

def get_fidelity(psi1, psi2, x):
    dx = abs(x[1] - x[0])
    return np.abs(np.sum(np.conj(psi1) * psi2) * dx)



data = np.load('./data40000_tf1_wf01.npz')
S = data['S'];A1 = data['A1'];A2=data['A2'];
F = data['F']
FFF = F

### plot the wave function for high fidelity and reasonable
from initialization import x,x1,xi,U0,omega_i,omega_f
ii = 20
Si = S[ii] # high fidelity
a1 = A1[ii];a2 = A2[ii];omega_f = 0.1;omega_i = 1;
a0 = omega_i; tf= 1;
a3 = (omega_f-omega_i-a2*tf**2-a1*tf)/tf**3;
T= np.linspace(0,tf,100)
opt_policy = three_order_poly([a0,a1,a2,a3],T)




rp_i = random_potential(x,x1,xi,U0,Si)+harmonic_potential(x,omega_i)
rp_f = random_potential(x,x1,xi,U0,Si)+harmonic_potential(x,omega_f)
initial_rp = rp_se(xmin=-L/2, xmax=L/2, ninterval=Nx, omega=omega_i, S=Si, U_0=U0, delta=d)
psi_i = initial_rp.wave_func()
final_rp = rp_se(xmin=-L/2, xmax=L/2, ninterval=Nx, omega=omega_f, S=Si, U_0=U0, delta=d)
psi_f = final_rp.wave_func()

psi_opt = get_final_states(100,opt_policy,Si,psi_i)
print('Fidelity:',F[20],'actual fidelity',get_fidelity(psi_f,psi_opt,x))

plt.figure()
plt.style.use('science')
plt.subplot(1,2,1)
plt.plot(x,abs(psi_i)**2,color='salmon',lw=1.5)
plt.plot(x,rp_i*0.1,color='red',lw=0.7,linestyle='--')
plt.plot(x,abs(psi_f)**2,color='grey',lw=1.5)
plt.plot(x,rp_f*0.1,color='black',lw=0.7,linestyle='--')
plt.plot(x,abs(psi_opt)**2,lw =2,color='blue',linestyle=':')


plt.ylim([-0.2,0.8])
plt.xlim([-5,5])
plt.text(2,0.65,'(a)',fontsize=16)
plt.grid()
plt.xticks([-5,0,5])
plt.yticks([0,0.4,0.8])
plt.ylabel(r'$|\psi(x)|^2,U(x)$',fontsize = 14)
plt.xlabel(r'$x$',fontsize = 18)

### plot the wave function for low fidelity unreasonabel

Si = S[6] # low fidelity
a1 = A1[6];a2 = A2[6];omega_f = 0.1;omega_i = 1;
a0 = omega_i; tf= 1;
a3 = (omega_f-omega_i-a2*tf**2-a1*tf)/tf**3;
T= np.linspace(0,tf,100)
opt_policy = three_order_poly([a0,a1,a2,a3],T)

print('Fidelity:',F[6])
rp_i = random_potential(x,x1,xi,U0,Si)+harmonic_potential(x,omega_i)
rp_f = random_potential(x,x1,xi,U0,Si)+harmonic_potential(x,omega_f)

initial_rp = rp_se(xmin=-L/2, xmax=L/2, ninterval=Nx, omega=omega_i, S=Si, U_0=U0, delta=d)
psi_i = initial_rp.wave_func()

final_rp = rp_se(xmin=-L/2, xmax=L/2, ninterval=Nx, omega=omega_f, S=Si, U_0=U0, delta=d)
psi_f = final_rp.wave_func()

psi_opt = get_final_states(100,opt_policy,Si,psi_i)

plt.subplot(1,2,2)
plt.plot(x,abs(psi_i)**2,color='salmon',lw=1.5)
plt.plot(x,rp_i*0.1,color='red',lw=0.7,linestyle='--')
plt.plot(x,abs(psi_f)**2,color='grey',lw=1.5)#linestyle='--'
plt.plot(x,rp_f*0.1,color='black',lw=0.7,linestyle='--')
plt.plot(x,abs(psi_opt)**2,lw =2,color='blue',linestyle=':')

plt.ylim([-0.2,0.8])
plt.xlim([-10,5])
plt.text(1,0.65,'(b)',fontsize=16)
plt.grid()
plt.xlabel(r'$x$',fontsize = 16)
#plt.text(-16,0.35,r'$|\psi(x)|^2,U_{tot}(x)$',fontsize=14,rotation=90)
plt.ylabel(r'$|\psi(x)|^2,U(x)$',fontsize = 14)
plt.xticks([-10,-5,0,5])
plt.yticks([0,0.4,0.8])




### Fig2
plt.figure()
A1_set,A2_set,W_set = get_max_omega()
### plot the fidelity dependent on a1 and a2 for HO

plt.subplot(1,2,1)
F1 = scipy.io.loadmat('Ftf1.mat')
F=F1['FF']
a1 = np.linspace(-30,30,200)
a2 = np.linspace(-100,100,200)
A11,A22 = np.meshgrid(a1,a2)

print('maximum fidelity is',np.max(F))
plt.title('fidelity',fontsize = 15)
quadmesh=plt.pcolormesh(A11,A22,F,cmap= 'bwr')#Oranges#Greys
quadmesh.set_clim(vmin=0, vmax=1)
plt.colorbar()

C = plt.contour(A1_set,A2_set, W_set,[2], colors='black',lw=0.2,linestyles=['--'])
#plt.clabel(C, inline=True, fontsize=10)
#C = plt.contour(A1,A2,F,[0.6], colors='black',lw=0.2,linestyles=['--'])
jj0=0;jj1=0;jj2=0;jj3=0;jj4=0;jj5=0
for i in range(40000):
    a1= A1[i];a2=A2[i];

    if FFF[i]>0.9 and a2<40 and a2>30 and jj0<5:
        plt.scatter(A1[i],A2[i],marker='o',facecolors='none',edgecolors='black',lw=0.5,s=25 )
        jj0+=1
    if FFF[i]>0.9 and a2<30 and a2>27 and jj1<10:
        plt.scatter(A1[i],A2[i],marker='o',facecolors='none',edgecolors='black',lw=0.5,s=25 )
        jj1+=1
    if FFF[i]>0.9 and a2<27 and a2>-10 and jj2<10:
        plt.scatter(A1[i],A2[i],marker='o',facecolors='none',edgecolors='black',lw=0.5,s=25 )
        jj2+=1

    if FFF[i]<0.9 and a1>15 and a1<24 and jj3<2:
        plt.scatter(A1[i],A2[i],marker='s',color='black' ,facecolors='none',edgecolors='black',lw = 1,s=25)#,facecolors='none',edgecolors='black')
        jj3+=1
    if FFF[i]<0.9 and a1>24 and a1<26 and jj4<2:
        plt.scatter(A1[i],A2[i],marker='s',color='black' ,facecolors='none',edgecolors='black',lw = 1,s=25)#,facecolors='none',edgecolors='black')
        jj4+=1
    if FFF[i]<0.9 and a1>26 and jj5<2:
        plt.scatter(A1[i],A2[i],marker='s',color='black' ,facecolors='none',edgecolors='black',lw = 1,s=25)#,facecolors='none',edgecolors='black')
        jj5+=1
plt.text(18,65,'(a)',fontsize=16,backgroundcolor='0.9')
plt.xlabel(r'$a_1$',fontsize = 18)
plt.ylabel(r'$a_2$',fontsize = 18)
plt.xticks([-30,0,30])
plt.yticks([-100,-50,0,50,100])

a0 = omega_i
a1 = A1[20];a2 = A2[20];
a3 = (omega_f-(1 + a1 * tf + a2 * tf ** 2)) / tf ** 3;
T = np.linspace(0,tf,100)
w_t = three_order_poly([a0,a1,a2,a3],T)

plt.subplot(2,2,2)
plt.plot(T,w_t**2,lw = 1,color='black')#,linestyle='--'
plt.text(0.85,1.5,'(b)',fontsize=16)
#plt.xlabel(r'$t$',fontsize = 18);
#plt.ylabel(r'$\omega^2(t)$',fontsize = 14)
plt.yticks([0,1,2])
plt.xticks([0,0.5,1])


a0 = omega_i
a1 = A1[6];a2 = A2[6];
a3 = (omega_f-(1 + a1 * tf + a2 * tf ** 2)) / tf ** 3;
T = np.linspace(0,tf,100)
w_t = three_order_poly([a0,a1,a2,a3],T)
plt.grid();


plt.subplot(2,2,4)
plt.plot(T,w_t**2,lw = 1,color='black')
plt.grid();
plt.text(0.85,28,'(c)',fontsize=16)
plt.text(-0.30,40,r'$\omega^2(t)$',fontsize=14,rotation=90)
#plt.ylabel(r'$\omega^2(t)$',fontsize = 14)
plt.xlabel('$t$',fontsize = 18);
plt.yticks([0,20,40])
plt.xticks([0,0.5,1])


plt.figure()

F1 = scipy.io.loadmat('Ftf_2.mat')
F=F1['FF']
a1 = np.linspace(-30,30,len(F))
a2 = np.linspace(-100,100,len(F))
A11,A22 = np.meshgrid(a1,a2)

print('maximum fidelity is',np.max(F))
plt.title(r'$t_f= 2$')
quadmesh=plt.pcolormesh(A11,A22,F,cmap= 'bwr')#Oranges#Greys
quadmesh.set_clim(vmin=0, vmax=1)
plt.colorbar()
plt.plot(a1,-1.5*a1+13,'r--',label = r'$a2 = -1.5a_1$+13')
plt.plot(a1,-1.5*a1+18,'k--',label = r'$a2 = -1.5a_1$+18')
plt.legend()
plt.xlabel(r'$a_1$',fontsize = 16)
plt.ylabel(r'$a_2$',fontsize = 16)

plt.xticks([-30,0,30])
plt.yticks([-100,-50,0,50,100])
plt.ylim([-50,50])
plt.xlim([-15,15])

'''
########################################
plt.figure()
plt.style.use('science')
plt.subplot(2,2,1)
plt.plot(x,abs(psi_i)**2,color='tomato',lw=2)
plt.plot(x,rp_i*0.1,color='red',lw=0.7,linestyle='--')
plt.plot(x,abs(psi_f)**2,color='grey',lw=2)
plt.plot(x,rp_f*0.1,color='black',lw=0.7,linestyle='--')


plt.ylim([-0.2,0.8])
plt.xlim([-10,10])
plt.text(6.6,0.56,'(a)',fontsize=14)
plt.grid()
plt.xticks([-10,-5,0,5,10])
plt.yticks([0,0.4,0.8])

plt.xlabel(r'$x$',fontsize = 16)
plt.ylabel(r'$|\psi(x)|^2,U_{tot}(x)$',fontsize = 12)


Si = S[6]
rp_i = random_potential(x,x1,xi,U0,Si)+harmonic_potential(x,omega_i)
rp_f = random_potential(x,x1,xi,U0,Si)+harmonic_potential(x,omega_f)

initial_rp = rp_se(xmin=-L/2, xmax=L/2, ninterval=Nx, omega=omega_i, S=Si, U_0=U0, delta=d)
psi_i = initial_rp.wave_func()

final_rp = rp_se(xmin=-L/2, xmax=L/2, ninterval=Nx, omega=omega_f, S=Si, U_0=U0, delta=d)
psi_f = final_rp.wave_func()

plt.subplot(2,2,2)
plt.plot(x,abs(psi_i)**2,color='tomato',lw=2)
plt.plot(x,rp_i*0.1,color='red',lw=0.7,linestyle='--')
plt.plot(x,abs(psi_f)**2,color='grey',lw=2)#linestyle='--'
plt.plot(x,rp_f*0.1,color='black',lw=0.7,linestyle='--')

plt.ylim([-0.2,0.8])
plt.xlim([-10,10])
plt.text(6.6,0.56,'(b)',fontsize=14)
plt.grid()
plt.xlabel(r'$x$',fontsize = 16)
#plt.text(-16,0.35,r'$|\psi(x)|^2,U_{tot}(x)$',fontsize=14,rotation=90)
plt.ylabel(r'$|\psi(x)|^2,U_{tot}(x)$',fontsize = 12)
plt.xticks([-10,-5,0,5,10])
plt.yticks([0,0.4,0.8])

#########
a0 = omega_i
a1 = A1[20];a2 = A2[20];
a3 = (omega_f-(1 + a1 * tf + a2 * tf ** 2)) / tf ** 3;
T = np.linspace(0,tf,100)
w_t = three_order_poly([a0,a1,a2,a3],T)

plt.subplot(4,2,6)
plt.plot(T,w_t**2,lw = 1,color='black')#,linestyle='--'
plt.text(0.85,1.5,'(d)',fontsize=14)
#plt.xlabel(r'$t$',fontsize = 18);
#plt.ylabel(r'$\omega^2(t)$',fontsize = 14)
plt.yticks([0,1,2])
plt.xticks([0,0.5,1])


a0 = omega_i
a1 = A1[6];a2 = A2[6];
a3 = (omega_f-(1 + a1 * tf + a2 * tf ** 2)) / tf ** 3;
T = np.linspace(0,tf,100)
w_t = three_order_poly([a0,a1,a2,a3],T)
plt.grid();


plt.subplot(4,2,8)
plt.plot(T,w_t**2,lw = 1,color='black')
plt.grid();
plt.text(0.85,28,'(e)',fontsize=14)
plt.text(-0.32,40,r'$\omega^2(t)$',fontsize=14,rotation=90)
#plt.ylabel(r'$\omega^2(t)$',fontsize = 14)
plt.xlabel('t',fontsize = 18);
plt.yticks([0,20,40])
plt.xticks([0,0.5,1])




###
A1_set,A2_set,W_set = get_max_omega()
### plot the fidelity dependent on a1 and a2 for HO

plt.subplot(2,2,3)

F1 = scipy.io.loadmat('F_a1a2.mat')
F=F1['FF']
a1 = np.linspace(-30,30,200)
a2 = np.linspace(-100,100,200)
A11,A22 = np.meshgrid(a1,a2)

print('maximum fidelity is',np.max(F))
plt.title(r'$Fidelity$')
quadmesh=plt.pcolormesh(A11,A22,F,cmap= 'bwr')#Oranges#Greys
quadmesh.set_clim(vmin=0, vmax=1)
plt.colorbar()

C = plt.contour(A1_set,A2_set, W_set,[6], colors='black',lw=0.2,linestyles=['--'])
#plt.clabel(C, inline=True, fontsize=10)
#C = plt.contour(A1,A2,F,[0.6], colors='black',lw=0.2,linestyles=['--'])

plt.scatter(A1[20],A2[20],marker='x',color='black' )
plt.scatter(A1[6],A2[6],marker='o',color='black' ,lw = 1)#,facecolors='none',edgecolors='black')
plt.text(18,65,r'(c)',fontsize=16)

plt.xlabel(r'$a_1$',fontsize = 15)
plt.ylabel(r'$a_2$',fontsize = 15)
plt.xticks([-30,0,30])
plt.yticks([-100,-50,0,50,100])

'''