#!/usr/bin/env python3
from random import random
import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA
from scipy import integrate
import scipy
import matplotlib.pyplot as plt
import csv

def mysys(t, X):
    x = X[0]
    
    for i in range(xn-1):
        x = np.vstack((x, X[i+1]))

    for idx in range(N_agent):        
        u[idx*2] = u[idx*2] + random()/1000000
        u[idx*2+1] = u[idx*2+1] + random()/1000000
        
    dx = np.matmul(A0, x) + np.matmul(B0, u)

    dxx = np.kron(x, x)
    dux = np.kron(x, u)
    dX = np.array(dx)
    dX = np.vstack((dX, dxx))
    dX = np.vstack((dX, dux))

    return np.transpose(dX)[0]

def fill_diagonal(N_agent: int, D:np.array):
    row = len(D)
    col = len(D[0])
    An = np.zeros((row*N_agent, col*N_agent))
    for idx in range(N_agent):
        for j in range(row):
            for k in range(col):
                An[idx*row+j][idx*col+k] = D[j][k]
    return An

##############################################################################################
#### User Define ####
# N_agent = 2

# hover = 0
# l = 1200                        # length of explore window, should be greater than xn^2, timestep dT
# fly_time = 1800                 # length of fly window, timestep dT
# iter_max = 50                   # max iteration to compute
# dT = 0.01                       # timestep

# epsilon = 10e-3                 # stop threshold

# x0 = np.array([[-1.5],[-1.5],[0],[0],
#                [-1.5],[-0.5],[0],[0]])

# target = np.array([[1.5],[0]])  # target
# distance = np.array([[0],[1]])  # formation

# A = np.array([[0,0,1,0], 
#                 [0,0,0,1],
#                 [0,0,-1/10,0],
#                 [0,0,0,-1/10]])

# B = np.array([[0,0],
#                 [0,0],
#                 [1,0],
#                 [0,1]])

save_file = 0
N_agent = 3

hover = 0
l = 1500                        # length of explore window, should be greater than xn^2, timestep dT
fly_time = 1800                 # length of fly window, timestep dT
iter_max = 50                   # max iteration to compute
dT = 0.01                       # timestep

epsilon = 10e-3                 # stop threshold

x0 = np.array([[-1.5],[-0.5],[0],[0],
               [-1.5],[-1.5],[0],[0],
               [-1.5],[0.5],[0],[0]])

target = np.array([[1.5],[0]])  # target
distance = np.array([[-0.707],[-0.5],[-0.707],[0.5]])  # formation

A = np.array([[0,0,1,0], 
                [0,0,0,1],
                [0,0,-1/10,0],
                [0,0,0,-1/10]])

B = np.array([[0,0],
                [0,0],
                [1,0],
                [0,1]])
##############################################################################################


n = len(A)*N_agent
xn = len(B)*N_agent                     # number of rows
un = len(B[0])*N_agent                  # number of columns
single_xn = int(xn/N_agent)

##### desire state #######
for idx in range(N_agent-1):
    if idx == 0:
        desire = np.array(distance[idx*int(single_xn/2):(idx+1)*int(single_xn/2)])
        desire = np.vstack((desire, np.zeros((int(single_xn/2),1))))
    else:
        desire = np.vstack((desire, distance[idx*int(single_xn/2):(idx+1)*int(single_xn/2)]))
        desire = np.vstack((desire, np.zeros((int(single_xn/2),1))))
desire = np.vstack((desire, target))
desire = np.vstack((desire, np.zeros((int(single_xn/2),1))))

# desire = np.array([1],[2],[0],[0])
# you may just define your desire state directly instead of using this loop
#################

############ Weight matrix #################
Q = np.eye(xn)*1
Q[(N_agent-1)*4][(N_agent-1)*4] = 0.1
Q[(N_agent-1)*4+1][(N_agent-1)*4+1] = 0.1
R = np.eye(un)*1
##### just give as Q = something instead of using this loop
############################################


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

################ Initial K ###############################################
K = np.array([[-1,0,-1,0,1,0,1,0],[0,-1,0,-1,0,1,0,1],[1,0,1,0,1,0,1,0],[0,1,0,1,0,1,0,1]])
K = np.array([[-0.8,0,-1,0,-0.8,0,-1,0,0.25,0,1,0],
              [0,-0.8,0,-1,0,-0.8,0,-1,0,0.25,0,1],
              [1,0,1.5,0,-0.3,0,-0.5,0,0.25,0,1,0],
              [0,1,0,1.5,0,-0.3,0,-0.5,0,0.25,0,1],
              [-0.3,0,-0.5,0,1,0,1.5,0,0.25,0,1,0],
              [0,-0.3,0,-0.5,0,1,0,1.5,0,0.25,0,1]])
###########################################################################

# K = np.zeros((un,xn))
eig2, v2 = np.linalg.eig(A0-np.matmul(B0,K))

u = np.zeros((un, 1))

# explore_dir = np.ones((2*N_agent,1))
# for idx in range(N_agent):
#     if target[0] < x0[4*idx]:
#         explore_dir[2*idx] = -1
#     if target[1] < x0[4*idx+1]:
#         explore_dir[2*idx+1] = -1   

X_save = np.transpose(x0)
x0 = np.matmul(T,x0)
X = x0
X = np.vstack((X, np.transpose(np.kron(np.transpose(x0), np.transpose(x0)))))
X = np.vstack((X, np.kron(x0, np.zeros((un, 1)))))

t_save = np.zeros(1)
v_save = np.zeros((2*N_agent,1))

for idx in range(hover):
    X_save = np.vstack((X_save,X_save[-1]))
    t_save = np.hstack((t_save, t_save[-1]+dT))

X = np.transpose(X)[-1]


##### data collection #############
for idx in range(l):
    if (idx%100 == 0):
        print('Data collected: ', idx)

    new_X = integrate.solve_ivp(mysys, [idx*dT, (idx+1)*dT], X)

    if idx == 0:
        Dxx = np.array(np.kron(new_X.y[0:xn,-1], new_X.y[0:xn,-1]) - np.kron(new_X.y[0:xn,0], new_X.y[0:xn,0]))
        Ixx = np.array(new_X.y[xn:xn+xn**2,-1] - new_X.y[xn:xn+xn**2,0])
        Ixu = np.array(new_X.y[xn+xn**2:len(new_X.y),-1] - new_X.y[xn+xn**2:len(new_X.y),0])
    else:
        Dxx = np.vstack((Dxx, np.kron(new_X.y[0:xn,-1], new_X.y[0:xn,-1]) - np.kron(new_X.y[0:xn,0], new_X.y[0:xn,0])))
        Ixx = np.vstack((Ixx, new_X.y[xn:xn+xn**2,-1] - new_X.y[xn:xn+xn**2,0]))
        Ixu = np.vstack((Ixu, new_X.y[xn+xn**2:len(new_X.y),-1] - new_X.y[xn+xn**2:len(new_X.y),0]))


    t_save = np.hstack((t_save, t_save[-1]+dT))

    new_z = new_X.y[:,-1]

    new_state = np.matmul(np.linalg.inv(T),new_z[:N_agent*4].reshape((N_agent*4,1)))

    X_save = np.vstack((X_save, new_state.T))
    new_v = np.zeros((2*N_agent,1))
    for j in range(N_agent):
        new_v[j*2] = (new_X.y[j*4+2][-1]-new_X.y[j*4+2][0])/dT
        new_v[j*2+1] = (new_X.y[j*4+3][-1]-new_X.y[j*4+3][0])/dT
    # print(new_v)
    v_save = np.hstack((v_save, new_v))
    X = new_X.y[:,-1]
#############################

for idx in range(l):
    if idx == 0:
        r = np.array(Ixx[idx])
        r = np.append(r, Ixu[idx])
    else:
        new_r = np.array(Ixx[idx])
        new_r = np.append(new_r, Ixu[idx])
        r = np.vstack((r, new_r))

if np.linalg.matrix_rank(r) != (xn+1)*xn/2+xn*un:
    print("Not full rank!")
else:
    print("Rank condition satisfied!")


X0 = np.matrix(scipy.linalg.solve_continuous_are(A0, B0, Q, R))
K0 = np.matrix(scipy.linalg.inv(R)*(B0.T*X0))

iter = 0

P_old = np.zeros(xn)
P = np.eye(xn)


########## Learning #################
print("Start learning:")
while LA.norm(P-P_old) > epsilon and iter<iter_max:
    iter = iter + 1
    P_old = P

    Qk = Q + np.matmul(np.matmul(np.transpose(K), R), K)

    X2 = np.matmul(Ixx, np.kron(np.eye(xn), np.transpose(K)))

    for idx in range(l):
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

    if iter == 1:
        iter_save = np.array([iter])
        K_save = np.array([np.linalg.norm(K-K0)])
        P_save = np.array([LA.norm(P-P_old)])
    else:
        iter_save = np.hstack((iter_save, iter))
        K_save = np.hstack((K_save, np.linalg.norm(K-K0)))
        P_save = np.hstack((P_save, LA.norm(P-P_old)))

    print(LA.norm(P-P_old))
######################################

print("Compare to optimal")
print(np.linalg.norm(K-K0))
print('K:')
print(K)
print("Optimal K")
print(K0)


#### Formation ##########
x = np.reshape(X_save[-1], (xn,1))
for idx in range(fly_time):
    z = np.matmul(T, x)
    X = z - desire
    u = -np.matmul(K, X)
    # u = -np.matmul(K0, X)
    dx = np.matmul(A, x) + np.matmul(B, u)
    new_v = np.zeros((2*N_agent,1))
    for idx in range(N_agent):
        new_v[idx*2] = dx[idx*4]
        new_v[idx*2+1] = dx[idx*4+1]

    v_save = np.hstack((v_save, new_v))
    x = x + dx*dT
    t_save = np.hstack((t_save, t_save[-1]+dT))
    X_save = np.vstack((X_save, np.transpose(x)))

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
# ax.set(xlim=(-6, 6), ylim=(-6, 6))


plt.show()



# fig, ax = plt.subplots()
# ax.set_xlabel('iter')
# ax.set_title('|K-K*|')
# ax.plot(iter_save, K_save, marker='.')
# plt.show()

# fig, ax = plt.subplots()
# ax.set_xlabel('iter')
# ax.set_title('|P-P_old|')
# ax.plot(iter_save, P_save, marker='.')
# plt.show()

# with open('trajectory.csv', 'w') as file:
#     writer = csv.writer(file)
#     writer.writerow(t_save)
#     for idx in range(N_agent):
#         writer.writerow(X_save[:,idx*4])
#         writer.writerow(X_save[:,idx*4+1])

if save_file == 1:
    t_fly = t_save[::10]
    x_fly = X_save[::10]
    height = np.ones((len(x_fly)))*0.8
    v_z = np.zeros((len(x_fly)))
    v_save = v_save.T
    v_save = v_save[::10]

    v_save = np.zeros((2*N_agent,1))
    for idx in range(len(x_fly)-1):
        new_v = np.zeros((2*N_agent,1))
        for j in range(N_agent):
            new_v[j*2] = (x_fly[idx+1][j*4+2]-x_fly[idx][j*4+2])/(dT*10)
            new_v[j*2+1] = (x_fly[idx+1][j*4+3]-x_fly[idx][j*4+3])/(dT*10) 
        v_save = np.hstack((v_save,new_v))

    with open('trajectory1.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(t_fly)
        writer.writerow(np.asarray(x_fly[:,0]).flatten())
        writer.writerow(np.asarray(x_fly[:,1]).flatten())
        writer.writerow(height)
        writer.writerow(v_save[0])
        writer.writerow(v_save[1])
        writer.writerow(v_z)

    with open('trajectory2.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(t_fly)
        writer.writerow(np.asarray(x_fly[:,4]).flatten())
        writer.writerow(np.asarray(x_fly[:,5]).flatten())
        writer.writerow(height)
        writer.writerow(v_save[2])
        writer.writerow(v_save[3])
        writer.writerow(v_z)

    if N_agent == 3:
        with open('trajectory3.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(t_fly)
            writer.writerow(np.asarray(x_fly[:,8]).flatten())
            writer.writerow(np.asarray(x_fly[:,9]).flatten())
            writer.writerow(height)
            writer.writerow(v_save[4])
            writer.writerow(v_save[5])
            writer.writerow(v_z)

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



# [[-1.00064496e+00 -6.95153785e-04 -1.36274574e+00 -1.97979125e-03
#    3.15353547e+00  8.40208836e-04  2.60677809e+00  4.83826518e-03]
#  [-2.52267934e-02 -9.81742410e-01 -1.62007461e-03 -1.34505523e+00
#   -2.85254685e-02  3.11008703e+00  7.50889905e-03  2.58651499e+00]
#  [ 9.91205110e-01 -3.94736204e-03  1.34349652e+00 -2.79009924e-03
#    3.16428709e+00 -3.03703736e-03  2.56867161e+00  2.49324870e-03]
#  [ 7.12399943e-03  9.93253921e-01 -6.49187064e-05  1.35755165e+00
#    1.28360533e-02  3.17918460e+00  2.56199305e-04  2.60971976e+00]]
# [[-1.          0.         -1.36509717  0.          3.16227766  0.
#    2.60823842  0.        ]
#  [ 0.         -1.          0.         -1.36509717  0.          3.16227766
#    0.          2.60823842]
#  [ 1.          0.          1.36509717  0.          3.16227766  0.
#    2.60823842  0.        ]
#  [ 0.          1.          0.          1.36509717  0.          3.16227766
#    0.          2.60823842]]


# Compare to optimal
# 0.04479089758091712
# [[-9.73137785e-01  1.01683375e-03 -1.35739587e+00 -1.15373849e-03
#    3.13894668e+00 -1.47386278e-02  2.59816809e+00 -4.96677025e-03]
#  [-6.97748873e-03 -1.00007876e+00  1.88573282e-04 -1.36465133e+00
#    9.02342543e-03  3.16290367e+00 -1.90972588e-03  2.60774808e+00]
#  [ 9.92256391e-01  3.56511841e-04  1.36210107e+00 -8.50309327e-04
#    3.16945587e+00  6.80845274e-03  2.60993103e+00  7.72475956e-04]
#  [ 1.07991211e-03  1.00063866e+00  9.76747355e-04  1.36606156e+00
#   -2.36243292e-03  3.16168047e+00 -4.46833606e-03  2.60778494e+00]]
# [[-1.          0.         -1.36509717  0.          3.16227766  0.
#    2.60823842  0.        ]
#  [ 0.         -1.          0.         -1.36509717  0.          3.16227766
#    0.          2.60823842]
#  [ 1.          0.          1.36509717  0.          3.16227766  0.
#    2.60823842  0.        ]
#  [ 0.          1.          0.          1.36509717  0.          3.16227766
#    0.          2.60823842]]
