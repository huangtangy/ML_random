import numpy as np
from eigen_SE import rp_se
from function_sets import random_sj
'''
Offering initialing parameters
'''







Si = [-1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1,
      -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, -1, 1, 1,
      -1, 1, -1, -1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, 1, -1,
      1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1,
      -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, -1, -1, 1, -1,
      -1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1]



hbar = 1;m = 1;mass= 1;L = 20;Nx = 640;
omega_i = 1;omega_f = 0.1;Nx1 = 160; #impurity number
U0 = 1 #0.01#0.01;

# sta parameters
tf = 1;Nt = 100;dt= tf/Nt;T=np.linspace(0,tf,Nt);

d=L/Nx1;xi = d; #width of impurity
x = np.linspace(-L/2,L/2, Nx)# coordinate space
dx= np.abs(x[2]-x[1])

dk = 2*np.pi/(L);K = dk*Nx/2;
p = np.linspace(-Nx/2,Nx/2-1,Nx)*dk #kinetic space

x1 = np.linspace(-L/2,L/2,Nx1); # impurity space
n_imp = Nx1/(L);

Si =random_sj(x1)
# wave-function
initial_rp = rp_se(xmin=-L/2, xmax=L/2, ninterval=Nx, omega=omega_i, S=Si, U_0=U0, delta=d)
psi_i = initial_rp.wave_func()

final_rp = rp_se(xmin=-L/2, xmax=L/2, ninterval=Nx, omega=omega_f, S=Si, U_0=U0, delta=d)
psi_f = final_rp.wave_func()

