#!/usr/bin/env python3
from dis import dis
import math
from random import random
from cvxpy import vstack
from matplotlib import animation
import numpy as np
from numpy.linalg import inv, matrix_rank
from numpy import linalg as LA, linspace, matmul, transpose, zeros
from scipy import integrate
from sympy import MatrixSymbol, Matrix
import scipy
import scipy.io
import time
import matplotlib.pyplot as plt



def mysys(t, X):
    x = X[0]
    u = np.zeros((un, 1))

    for i in range(xn-1):
        x = np.vstack((x, X[i+1]))

    for idx in range(N_agent):
        for k in range(N_noise):
            u[idx] = u[idx] + math.sin(noise[idx][k]*t)/N_noise
        
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

def fill_diagonal(N_agent: int, D:np.array):
    row = len(D)
    col = len(D[0])
    An = np.zeros((row*N_agent, col*N_agent))
    for idx in range(N_agent):
        for j in range(row):
            for k in range(col):
                An[idx*row+j][idx*col+k] = D[j][k]
    return An


epsilon = 10e-6
# A = np.array([[-0.4125,-0.0248,0.0741,0.0089],
#               [101.5873,-7.2651,2.7608,2.8068],
#               [0.0704,0.0085,-0.0741,-0.0089],
#               [0.0878,0.2672,0,-0.3674]])

A = np.array([[0,0,1,0], 
                [0,0,0,1],
                [0,0,-1/10,0],
                [0,0,0,-1/10]])

# A = np.array([[-1,0,1,0], 
#                 [0,-2,0,1],
#                 [0,3,-1/10,0],
#                 [4,0,0,-1/10]])

# B = np.array([[-0.0042,0.0064],
#               [-1.0360,1.5849],
#               [0.0042,0],
#               [0.1261,0]])

B = np.array([[0,0],
                [0,0],
                [1/10,0],
                [0,1/10]])

N_agent = 3

plot_freq = 20

n = len(A)*N_agent
xn = len(B)*N_agent                     # number of rows
un = len(B[0])*N_agent                  # number of columns
single_xn = int(xn/N_agent)

target = np.array([[5],[5]])
distance = np.array([[-1],[-1],[1],[-1]])


for idx in range(N_agent-1):
    if idx == 0:
        desire = np.array(distance[idx*int(single_xn/2):(idx+1)*int(single_xn/2)])
        desire = np.vstack((desire, np.zeros((int(single_xn/2),1))))
    else:
        desire = np.vstack((desire, distance[idx*int(single_xn/2):(idx+1)*int(single_xn/2)]))
        desire = np.vstack((desire, np.zeros((int(single_xn/2),1))))
desire = np.vstack((desire, target))
desire = np.vstack((desire, np.zeros((int(single_xn/2),1))))

Q = np.eye(xn)
Q[8][8] = 10
Q[9][9] = 10
R = np.eye(un)*0.5


An = fill_diagonal(N_agent, A)
Bn = fill_diagonal(N_agent, B)

A = An
B = Bn

T = np.zeros((xn, xn))
for idx in range(N_agent-1):
    for k in range(single_xn):
        T[idx*single_xn+k][k] = -1
        T[idx*single_xn+k][(idx+1)*single_xn+k] = 1
for idx in range(single_xn):
    for k in range(N_agent):
        T[(N_agent-1)*single_xn+idx][k*single_xn+idx] = 1/N_agent


A0 = np.matmul(np.matmul(T, A), inv(T))
B0 = np.matmul(T, B)

eig, v = np.linalg.eig(A0)
print(eig)


K = np.zeros((un, xn))
l = 2000                        # length of window, should be greater than xn^2
iter_max = 10                   # max iteration
dT = 0.01                        # 
step = 1
N_noise = 100

x0 = np.array([[0],[0],[0],[0],
               [1],[0],[0],[0],
               [-1],[0],[0],[0]])

for idx in range(un):
    if idx == 0:
        noise = np.array((np.random.rand(1,N_noise) - 0.5)*1000)
    else:
        noise = np.vstack((noise, (np.random.rand(1,N_noise) - 0.5)*1000))


X = x0
X = np.vstack((X, np.transpose(np.kron(np.transpose(x0), np.transpose(x0)))))
X = np.vstack((X, np.kron(x0, np.zeros((un, 1)))))


t_save = np.zeros((1,1))
X_save = np.transpose(x0)
x = x0

z = np.zeros((xn, 1))

for idx in range(N_agent-1):
    for k in range(single_xn):
        z[idx*single_xn+k] = x[(idx+1)*single_xn+k]-x[k]

for k in range(single_xn):
    for idx in range(N_agent):
        z[(N_agent-1)*single_xn+k] = z[(N_agent-1)*single_xn+k]+x[idx*single_xn+k]
    z[(N_agent-1)*single_xn+k] = z[(N_agent-1)*single_xn+k]/N_agent

old_X = z - desire
# old_X = z

for idx in range(l):
    old_x = x
    u = np.zeros((un, 1))
    t = idx*dT

    for j in range(un):
        for k in range(N_noise):
            u[j] = u[j] + math.sin(noise[j][k]*t)/N_noise

    u = 100*u

    dx = np.matmul(A, x) + np.matmul(B, u)
    x = x + dx*dT
    
    t_save = np.vstack((t_save, dT*idx))
    X_save = np.vstack((X_save, np.transpose(x)))

    z = np.matmul(T, x)
    X = z - desire
    # X = z

    if idx == 0:
        Dxx = np.transpose(np.array(np.kron(X, X) - np.kron(old_X, old_X)))
        Ixx = np.transpose(np.array((np.kron(X, X)+np.kron(old_X, old_X))*dT/2))
        Ixu = np.transpose(np.array((np.kron(X, u)+np.kron(old_X, u))*dT/2))
    else:
        Dxx = np.vstack((Dxx, np.transpose(np.kron(X, X) - np.kron(old_X, old_X))))
        Ixx = np.vstack((Ixx, np.transpose((np.kron(X, X)+np.kron(old_X, old_X))*dT/2)))
        Ixu = np.vstack((Ixu, np.transpose((np.kron(X, u)+np.kron(old_X, u))*dT/2)))
    old_X = X


# print(X_save[-1])


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


X0 = np.matrix(scipy.linalg.solve_continuous_are(A0, B0, Q, R))
K0 = np.matrix(scipy.linalg.inv(R)*(B0.T*X0))

iter = 0

print(len(X))
P_old = np.zeros(xn)
P = np.eye(xn)

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
    # time.sleep(0.5)

print("Compare to optimal")
print(np.linalg.norm(K-K0))
 
print(K)
print(K0)

for idx in range(int(15/dT)):

    z = np.matmul(T, x)
    X = z - desire

    u = -np.matmul(K, X)
    # u = -np.matmul(K0, X)
    dx = np.matmul(A, x) + np.matmul(B, u)
    x = x + dx*dT
    t_save = np.vstack((t_save, l*dT+(idx+1)*dT))
    X_save = np.vstack((X_save, np.transpose(x)))


# print(P)
# print(K)
# print(K0)
# print(LA.norm(K-K0))
print(x)

fig, ax = plt.subplots()
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('Formation')
fig_legend = ['Target']

ax.plot(target[0], target[1], "*")
ax.legend(['Target'])
for idx in range(N_agent):
    ax.plot(X_save[:,idx*single_xn], X_save[:,idx*single_xn+1])
    fig_legend.append('Agent')
    
ax.legend(fig_legend)



plt.show()

# for idx in range(N_agent):
#     data_X = X_save[:,idx*single_xn]
#     data_Y = X_save[:,idx*single_xn+1]
#     data_x = data_X[::plot_freq]
#     data_y = data_Y[::plot_freq]
#     if idx == 0:
#         ani_data = np.transpose(data_x)
#         ani_data = np.vstack((ani_data, np.transpose(data_y)))
#     else:
#         ani_data = np.vstack((ani_data, np.transpose(data_x)))
#         ani_data = np.vstack((ani_data, np.transpose(data_y)))

# ani_time = t_save[::plot_freq]

# fig, ax = plt.subplots()
# ax.set_xlim([-2, 7])
# ax.set_ylim([-2, 7])
# time_template = 'time = %.1fs'
# time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


# lines = []
# xlist = np.zeros((N_agent, 1))
# ylist = np.zeros((N_agent, 1))
# for idx in range(N_agent):
#     line, = ax.plot([])
#     line.set_data([],[])
#     lines.append(line)
# time_text.set_text('')

# def animate(idx):
#     for k, line in enumerate(lines):
#         line.set_data(np.transpose(ani_data[k*2])[:idx], np.transpose(ani_data[k*2+1])[:idx])
#     time_text.set_text(time_template % ani_time[idx])
#     return lines

# ani = animation.FuncAnimation(fig, animate)

# plt.show()



