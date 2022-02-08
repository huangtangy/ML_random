'''
a set of functions
'''
#from initialization import *
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import pi, abs
from scipy.linalg import expm
import random
from numpy.fft import fft, ifft, fftshift, ifftshift
from numpy.polynomial.hermite import Hermite


def to_batch(x, batch_size):

    num = x.shape[0]
    batch_num = num / batch_size
    batch_num = int(batch_num)

    x = x[:batch_num * batch_size]
    shape = list(x.shape)
    del shape[0]
    shape.insert(0, batch_size)
    shape.insert(0, batch_num)
    y = np.zeros(shape)

    for i in range(batch_num):
        y[i] = x[batch_size * i : batch_size * (i + 1)]
    return y

def to_matrix(x):
  # turn to 2d
    x = np.expand_dims(x, -1)
    one = np.ones([x.shape[0], 160, 160])#1,160 for 1d
    x = one * x
    x = x + np.transpose(x, [0, 2, 1])
    x = np.expand_dims(x, 1)
    return x

def for_cnn(tra_x,tst_x):
    # 使用卷积请运行下方三行代码
    tra_x = to_matrix(tra_x)  # 转化为160*160
    tst_x = to_matrix(tst_x)
    return tra_x,tst_x

def random_potential(x, x1, xi, U0, s_i):
    '''
    x:spatial space;  x1:position of impurity
    U0: random strength;s_i: random sequence;xi: width of impurity
    return the sum of speckle potential in the whole spatial space
    '''
    rp_sum = np.array(np.zeros(len(x)))
    for i in range(len(s_i)):
        U_local = s_i[i] * U0 * np.exp(-((x - x1[i]) ** 2) / (xi ** 2))
        rp = np.array(U_local)
        rp_sum = rp_sum + rp
    return rp_sum


def random_sj(x):
    '''
    offering a  randome sequence of s = 1/-1
    and satisfy <s_j> = 0 according to the impurity number
    '''
    Si = []
    for i in range(int(len(x) / 2)):
        Si.append(1)
        Si.append(-1)
    random.shuffle(Si)  # randomly shuffle
    return Si


def FFT(psi,p,U_tot, dt):
    '''
    psi:wavefunction;p:moment space;U_tot: potential;
    dt:time length;
    the fast fourier transformation method
    '''
    from initialization import  Nt, x, p,dt
    Nx = len(psi)
    ''' time-operator for a single timestep'''
    P = np.exp(-0.5 * 1j * p * p * dt)
    V = np.exp(-0.5 * 1j * U_tot * dt)
    psi = np.multiply(V, psi)  # V*psi
    psik = fftshift(fft(psi) / Nx)
    psik = np.multiply(P, psik)  # P*psik
    psi = ifft(ifftshift(psik) * Nx)
    psi = np.multiply(V, psi)  # V*psi
    return psi


def get_fidelity(psi1, psi2, x):
    '''return the fidelity between psi1 and psi2'''
    dx = abs(x[1] - x[0])
    return np.abs(np.sum(np.conj(psi1) * psi2) * dx)**2


def get_final_states(N_t, omega_set, S, psi_i):
    '''
    N_t:time sequence;omega_set: sequence of trap frquency;
    psi_i: initial wavefunction;S:random sequence;
    Return final wavefunction for the control sequence
    '''
    from initialization import omega_f, T, tf, L, Nx, omega_i, U0, d, Nt, x,x1,xi,dt,p
    # where omega_set is \omega(t)
    # using  time-split method
    random_p = random_potential(x, x1, xi, U0, S)  # random sequence
    psi = psi_i
    for i in range(N_t):
        omega_t = omega_set[i]
        harmonic_p = (0.5 * omega_t ** 2) * x ** 2  # harmonic trap
        potential_list = harmonic_p + random_p
        psi = FFT(psi,potential_list, dt)  # time evolution of dt
    return psi

# convert a1 a2 to the fidelity performances
def produce_fidelity(a1, a2, Si):
    from eigen_SE import rp_se
    from initialization import omega_f, T, tf, L, Nx, omega_i, U0, d, Nt, x

    a3 = (omega_f - (1 + a1 * tf + a2 * tf ** 2)) / tf ** 3;
    omega_t = 1 + a1 * T + a2 * T ** 2 + a3 * T ** 3

    # S_i[i]#random_sj(x1)#
    initial_rp = rp_se(xmin=-L / 2, xmax=L / 2, ninterval=Nx, omega=omega_i, S=Si, U_0=U0, delta=d)
    psi_i = initial_rp.wave_func()
    final_rp = rp_se(xmin=-L / 2, xmax=L / 2, ninterval=Nx, omega=omega_f, S=Si, U_0=U0, delta=d)
    psi_f = final_rp.wave_func()

    psi_t = get_final_states(Nt, omega_t, Si, psi_i)
    F = get_fidelity(psi_f, psi_t, x);
    # print('F:',F)
    return F

def normalization(psi, x):
    '''return the normalized wave-function'''
    return psi / (np.trapz(np.conj(psi) * psi, x)) ** 0.5

def harmonic_potential(x, omega2):
    '''creating  harmonic trap input omega^2'''
    return (0.5 * omega2) * x ** 2

def laplacian(x, N):
    '''构造二阶微分算子：Laplacian'''
    dx = x[1] - x[0]
    return (-2 * np.diag(np.ones((N), np.float32), 0)
            + np.diag(np.ones((N - 1), np.float32), 1)
            + np.diag(np.ones((N - 1), np.float32), -1)) / (dx ** 2)





