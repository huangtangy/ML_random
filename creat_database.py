#from function_sets import *
from initialization import *
from matlab import fft, ifft, fftshift, ifftshift
import numpy as np
import multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
import scipy.io
from eigen_SE import rp_se
from function_sets import *

AA1 = scipy.io.loadmat('A1.mat')
AA1 = AA1['AA1']
AA2 = scipy.io.loadmat('A2.mat')
AA2 = AA2['AA2']


def find_max(set):
    text = pd.DataFrame(set);  # text[i][j]=set_F[j][i]
    row, cloum = text.stack().idxmax()  # right row and cloum set_F[row][cloum]=set_F.max()
    return row,cloum

def get_maximum_inf(M,N_M,M1,N_M1,Si,psi_i,psi_f):
    from initialization import  L, omega_f, omega_i, d, Nx, p, x, x1, xi,tf, Nt, T
    # find optimal control function for a given coefficient grid [-M,M](2-dimension)
    start = time.time()
    # M = 10;N_M =100;M1 = 10;N_M1 = 100
    F_set = np.zeros(shape=(N_M, N_M1), dtype=float)
    A1_set = np.zeros(shape=(N_M, N_M1), dtype=float)
    A2_set = np.zeros(shape=(N_M, N_M1), dtype=float)

    A1 = np.linspace(-M, M, N_M);
    A2 = np.linspace(-M1, M1, N_M1);

    for i in range(N_M):
        a1 = A1[i];
        for j in range(N_M1):
            a0 = 1;a2 = A2[j];
            a3 = (omega_f - (a0 + a1 * tf + a2 * tf ** 2)) / tf ** 3;
            omega_t = 1+a1*T+a2* T ** 2 + a3 * T ** 3
            psi_t = get_final_states(Nt, omega_t, Si, psi_i)
            F = get_fidelity(psi_f, psi_t, x);
            F_set[i][j] = round(F, 4)
            # print(F)
            A1_set[i][j] = a1
            A2_set[i][j] = a2
    end = time.time()
    row, cloum = find_max(F_set)  # position of maximum
    a1_max = A1_set[row][cloum];
    a2_max = A2_set[row][cloum];
    F_max = F_set[row][cloum];
    print("耗时%s" % (end - start))
    return omega_t,a1_max,a2_max,F_max
'''
def get_maximum_inf(AA1,AA2,Si,psi_i,psi_f):
    from initialization import  L, omega_f,  x,tf, Nt, T
    # find optimal control function for a given coefficient grid [-M,M](2-dimension)
    start = time.time()
    Fi = 0;
    for i in range(len(AA1[0])):
        a1 = AA1[0][i];a2 = AA2[0][i];
        a3 = (omega_f-(1 + a1 * tf + a2 * tf ** 2)) / tf ** 3;
        omega_t = 1 + a1 * T + a2 * T ** 2 + a3 * T ** 3
        psi_t = get_final_states(Nt, omega_t, Si, psi_i)
        F = get_fidelity(psi_f, psi_t, x);
        if F>Fi:
            a1_max = a1
            a2_max =  a2
            F_max = F
            Fi = F
    end = time.time()
    a3_max = (omega_f -(1+ a1_max*tf+a2_max*tf**2))/tf**3;
    omega_t = 1 + a1_max * T + a2_max * T ** 2 + (a3_max) * T ** 3
    print("耗时%s" % (end - start))

    return omega_t,a1_max,a2_max,F_max
'''
def main_database4(name,Ni):
    # find optimal control function for a given coefficient grid [-M,M](2-dimension)
    from initialization import L, omega_f, omega_i, d, Nx, x1,Si
    start = time.time()
    #M = 30;N_M = 200;M1 = 100;N_M1 = 200
    #a1 = np.linspace(-M,M,N_M);a2 = np.linspace(-M1,M1,N_M1);
    #AA1,AA2 = np.meshgrid(a1,a2)
    F = [];A1 = [];A2 = [];SS =[];RC=[];FC=[];RH=[];
    for i in range(Ni):
        Si = random_sj(x1)#S[i] #random_sj(x1)#S_i[i]#S[i]#
        initial_rp = rp_se(xmin=-L/2, xmax=L/2, ninterval=Nx, omega=omega_i, S=Si, U_0=U0, delta=d)
        psi_i = initial_rp.wave_func()
        final_rp = rp_se(xmin=-L/2, xmax=L/2, ninterval=Nx, omega=omega_f, S=Si, U_0=U0, delta=d)
        psi_f = final_rp.wave_func()
        M = 30;N_M = 100; M1 = 100;N_M1 = 100;
        omega_t,a1_max, a2_max, F_max = get_maximum_inf(M,N_M,M1,N_M1,Si,psi_i,psi_f)
        #get_maximum_inf(AA1,AA2, Si, psi_i, psi_f)
        F.append(F_max);A1.append(a1_max);A2.append(a2_max);

        #P=6,F_b =0.9
        if max(omega_t**2)<6 and F_max > 0.9:#P =1
            RH.append(1)
        else:
            RH.append(0)

        SS.append(Si[:])
        #print(i,'max(omega2):',round(max(omega_t**2),3),round(a1_max,3), round(a2_max,3),round(F_max,3))
        print(i, a1_max, a2_max, F_max)
    end = time.time()
    np.savez('data_10000_tf1'+str(name)+'.npz',S =SS,A1=A1,A2=A2,F=F,RH=RH)
    #np.savez('data_500_0530_09extend.npz',S =SS,A1=A1,A2=A2,F=F,FC=FC,RC=RC,CC=CC)
    #print(NN)
    print(str(name)+"耗时%s" % (end - start))
    #return NN


if __name__== '__main__':

    t_i = time.time()
    cpu_num = mp.cpu_count()
    p_conn,c_conn=mp.Pipe()
    pool = mp.Pool(cpu_num)

    name = 4
    Ni = 10000
    #S_i = data_new[:Ni]

    for i in range(name):

        pros = mp.Process(target=main_database4,args=(i,Ni))
        pros.start()




