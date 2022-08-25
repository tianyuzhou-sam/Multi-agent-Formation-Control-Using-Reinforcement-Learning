#!/usr/bin/env python3
import math
from random import random
from cvxpy import vstack
import numpy as np
from numpy.linalg import inv, matrix_rank
from numpy import linalg as LA, linspace, transpose
from scipy import integrate
from sympy import MatrixSymbol, Matrix, false, true
import scipy
import scipy.io
import time
import matplotlib.pyplot as plt



def mysys(t, X):
    x = X[0]
    u = np.zeros((un, 1))

    for i in range(xn-1):
        x = np.vstack((x, X[i+1]))

    for idx in range(len(i1)):
        u[0] = u[0] + math.sin(i1[idx]*t)/len(i1)
        u[1] = u[1] + math.sin(i2[idx]*t)/len(i2)

    u = 100*u

    dx = np.matmul(A, x) + np.matmul(B, u)

    dxx = np.kron(x, x)
    dux = np.kron(x, u)
    dX = np.array(dx)
    dX = np.vstack((dX, dxx))
    dX = np.vstack((dX, dux))

    return np.transpose(dX)[0]


def optimalsys(t, X):
    x = X[0]

    for i in range(xn-1):
        x = np.vstack((x, X[i+1]))

    u = -np.matmul(K, x)
    dx = np.matmul(A, x) + np.matmul(B, u)

    return np.transpose(dx)[0]


epsilon = 10e-6
A = np.array([[-0.4125,-0.0248,0.0741,0.0089,0,0],
              [101.5873,-7.2651,2.7608,2.8068,0,0],
              [0.0704,0.0085,-0.0741,-0.0089,0,0.0200],
              [0.0878,0.2672,0,-0.3674,0.0044,0.3962],
              [-1.8414,0.0990,0,0,-0.0343,-0.0330],
              [0,0,0,-359,187.5364,-87.0316]])

# A = np.array([[0,0,1,0], 
#                 [0,0,0,1],
#                 [0,0,-1,0],
#                 [0,0,0,-1]])

# A = np.array([[-0.4125,-0.0248,0.0741,0.0089],
#               [101.5873,-7.2651,2.7608,2.8068],
#               [0.0704,0.0085,-0.0741,-0.0089],
#               [0.0878,0.2672,0,-0.3674]])


B = np.array([[-0.0042,0.0064],
              [-1.0360,1.5849],
              [0.0042,0],
              [0.1261,0],
              [0,-0.0168],
              [0,0]])

# B = np.array([[0,0],
#                 [0,0],
#                 [-1,0],
#                 [0,-1]])

# B = np.array([[-0.0042,0.0064],
#               [-1.0360,1.5849],
#               [0.0042,0],
#               [0.1261,0]])


eig, v = np.linalg.eig(A)
print(eig)

continuous = true
n = len(A)
xn = len(B)                     # number of rows
un = len(B[0])                  # number of columns
Q = np.eye(n)
Q[2][2] = 0.1
Q[3][3] = 0.1
Q[4][4] = 0.1
Q[5][5] = 0.1

R = np.eye(un)

K = np.zeros((un, xn))          # initial setup of K

l = 1000                        # length of window, should be greater than xn^2
iter_max = 10                   # max iteration
dT = 0.01                        # 
step = 1

x0 = np.array([[10],[2],[10],[2],[-1],[-2]])
# x0 = np.array([[10],[2],[10],[2]])
i1 = (np.random.rand(100,1) - 0.5)*1000
i2 = (np.random.rand(100,1) - 0.5)*1000

X = x0
X = np.vstack((X, np.transpose(np.kron(np.transpose(x0), np.transpose(x0)))))
X = np.vstack((X, np.kron(x0, np.zeros((un, 1)))))


t_save = np.zeros((1,1))
X_save = np.transpose(x0)

x = x0

X = np.transpose(X)[0]

if continuous:
    for idx in range(l):
        # t_eval = np.linspace(idx*dT, (idx+1)*dT, 101)
        # new_X = integrate.solve_ivp(mysys, [idx*dT, (idx+1)*dT], X, t_eval=t_eval)
        new_X = integrate.solve_ivp(mysys, [idx*dT, (idx+1)*dT], X)

        if idx == 0:
            Dxx = np.array(np.kron(new_X.y[0:xn,-1], new_X.y[0:xn,-1]) - np.kron(new_X.y[0:xn,0], new_X.y[0:xn,0]))
            Ixx = np.array(new_X.y[xn:xn+xn**2,-1] - new_X.y[xn:xn+xn**2,0])
            Ixu = np.array(new_X.y[xn+xn**2:len(new_X.y),-1] - new_X.y[xn+xn**2:len(new_X.y),0])
        else:
            Dxx = np.vstack((Dxx, np.kron(new_X.y[0:xn,-1], new_X.y[0:xn,-1]) - np.kron(new_X.y[0:xn,0], new_X.y[0:xn,0])))
            Ixx = np.vstack((Ixx, new_X.y[xn:xn+xn**2,-1] - new_X.y[xn:xn+xn**2,0]))
            Ixu = np.vstack((Ixu, new_X.y[xn+xn**2:len(new_X.y),-1] - new_X.y[xn+xn**2:len(new_X.y),0]))


        t_save = np.vstack((t_save, (idx+1)*dT))
        X_save = np.vstack((X_save, new_X.y[0:xn,-1]))
        X = new_X.y[:,-1]
# else:
#     for idx in range(l):
#         old_x = x
#         u = np.zeros((un, 1))
#         t = idx*dT

#         for k in range(len(i1)):
#             u[0] = u[0] + math.sin(i1[k]*t)/len(i1)
#             u[1] = u[1] + math.sin(i2[k]*t)/len(i2)

#         u = 100*u

#         dx = np.matmul(A, x) + np.matmul(B, u)
#         x = x + dx*dT

#         X_save = np.vstack((X_save, np.transpose(x)))
        
#         if idx == 0:
#             Dxx = np.transpose(np.array(np.kron(x, x) - np.kron(old_x, old_x)))
#             Ixx = np.transpose(np.array((np.kron(x, x)+np.kron(old_x, old_x))*dT/2))
#             Ixu = np.transpose(np.array((np.kron(x, u)+np.kron(old_x, u))*dT/2))
#         else:
#             Dxx = np.vstack((Dxx, np.transpose(np.kron(x, x) - np.kron(old_x, old_x))))
#             Ixx = np.vstack((Ixx, np.transpose((np.kron(x, x)+np.kron(old_x, old_x))*dT/2)))
#             Ixu = np.vstack((Ixu, np.transpose((np.kron(x, u)+np.kron(old_x, u))*dT/2)))


 
# print(Ixu)
# print(new_X.y[xn+xn**2:len(new_X.y),-1])

for idx in range(int(l/step)):
    if idx == 0:
        r = np.array(Ixx[idx])
        r = np.append(r, Ixu[idx])
    else:
        new_r = np.array(Ixx[idx])
        new_r = np.append(new_r, Ixu[idx])
        r = np.vstack((r, new_r))


print(np.linalg.matrix_rank(r))
if np.linalg.matrix_rank(r) != (xn+1)*xn/2+xn*un:
    print("Not full rank!")


P_old = np.zeros(xn)
P = np.eye(xn)*10


# Dxx = scipy.io.loadmat('src/Dxx.mat')
# Dxx = Dxx.get('Dxx')
# Ixx = scipy.io.loadmat('src/XX.mat')
# Ixx = Ixx.get('XX')
# Ixu = scipy.io.loadmat('src/XU.mat')
# Ixu = Ixu.get('XU')

# data = scipy.io.loadmat('src/Y.mat')
# print(data)

X0 = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))
K0 = np.matrix(scipy.linalg.inv(R)*(B.T*X0))

iter = 0

while LA.norm(P-P_old) > epsilon and iter<50:
    iter = iter + 1
    P_old = P

    Qk = Q + np.matmul(np.matmul(np.transpose(K), R), K)

    X2 = np.matmul(Ixx, np.kron(np.eye(xn), np.transpose(K)))

    for idx in range(int(l/step)):
        if idx == 0:
            Theta = np.array(Dxx[idx])
            Theta = np.append(Theta, -X2[idx]-Ixu[idx])
        else:
            new_term = np.array(Dxx[idx])
            new_term = np.append(new_term, -X2[idx]-Ixu[idx])
            Theta = np.vstack((Theta, new_term))


    vecQ = np.zeros((len(Q)**2, 1))
    for i in range(len(vecQ)):
        vecQ[i] = Qk.flatten()[i]

    Xi = -np.matmul(Ixx, vecQ)
    pp = np.matmul(np.linalg.pinv(Theta), Xi)
    P = np.reshape(pp[0:xn*xn], (xn, xn))
    P = (P + np.transpose(P))/2

    BPv = pp[xn*xn:len(pp)]
    K = np.matmul(inv(R), np.transpose(np.reshape(BPv, (xn, un))))/2

    # print(K)
    # print(len(pp))
    # print(P)
    # print(BPv)
    print(LA.norm(P-P_old))
    # print(LA.norm(K-K0))
    time.sleep(0.5)
    

for idx in range(10000):
    x = X_save[-1]
    t_eval = np.linspace(l*dT+idx*dT, l*dT+(idx+1)*dT, 101)
    new_X = integrate.solve_ivp(optimalsys, [l*dT+idx*dT, l*dT+(idx+1)*dT], x, t_eval=t_eval)

    t_save = np.vstack((t_save, l*dT+(idx+1)*dT))
    X_save = np.vstack((X_save, new_X.y[0:xn,-1]))
    x = new_X.y[:,-1]


# print(P)
print(K)
print(K0)
print(LA.norm(K-K0))

# print(X_save)

