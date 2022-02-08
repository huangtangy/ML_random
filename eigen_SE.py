
import numpy as np
import matplotlib.pyplot as plt
from function_sets import random_potential,harmonic_potential,normalization
#from initialization import *
#from funcs_set import harmonic_potential
# give a class for finding the eigenstats/eigenvalue of HO in random potential

class rp_se:
    def __init__(self,S,U_0,delta,
                 mass=1, hbar=1,
                 xmin=-10, xmax=10, ninterval=640,omega = 1,Nx1 = 160):
        self.ninterval1 = Nx1 # impulitry number
        self.x = np.linspace(xmin, xmax, ninterval)
        self.x1 = np.linspace(xmin, xmax, self.ninterval1)
        self.harmonicpotential = harmonic_potential(self.x,omega**2) # harmonic trap
        self.U0 = U_0 #impulity strength
        self.n_imp= ninterval/(xmax-xmin)
        self.xi = delta # the width of disorder

        self.random_potential = random_potential(self.x,self.x1,self.xi,self.U0,S)
        self.potential_list = self.harmonicpotential + self.random_potential # total potential
        self.U_tot = np.diag(self.potential_list, 0)
        self.Lap = self.laplacian(ninterval) # laplacian matrix
        self.H = -(hbar**2/(2*mass))* self.Lap + self.U_tot # Hamiltonian
        self.eigE, self.eigV = self.eig_solve() #eigenstates and eigenvalues

    def laplacian(self, N):
        '''构造二阶微分算子：Laplacian'''
        dx = self.x[1] - self.x[0]
        return (-2 * np.diag(np.ones((N), np.float32), 0)
                + np.diag(np.ones((N - 1), np.float32), 1)
                + np.diag(np.ones((N - 1), np.float32), -1))/(dx**2)

    def eig_solve(self):
        '''解哈密顿矩阵的本征值，本征向量；并对本征向量排序'''
        w, v = np.linalg.eig(self.H)
        idx_sorted = np.argsort(w)
        return w[idx_sorted], normalization(v[:, idx_sorted],self.x)

    def wave_func(self, n=0):
        return self.eigV[:, n]

    def eigen_value(self, n=0):
        #print('The instantaneous energy is:', self.eigE[n])
        return self.eigE[n]

    def check_eigen(self, n=1):
        #check wheter H|psi> = E |psi>
        with plt.style.context(['science']):
            HPsi = np.dot(self.H, self.eigV[:, n])
            EPsi = self.eigE[n] * self.eigV[:, n]
            plt.plot(self.x, HPsi, label=r'$H|\psi_{%s} \rangle$' % n)
            plt.plot(self.x, EPsi, 'r-.',lw=2.0, label=r'$E |\psi_{%s} \rangle$' % n)
            plt.legend(loc='upper center')
            plt.xlabel(r'$x$');plt.ylabel(r'$\psi(x)$')
            plt.ylim(EPsi.min(), EPsi.max() * 1.6)


