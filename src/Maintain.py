#!/usr/bin/env python3
from random import random
from timeit import repeat
from turtle import color
import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA
from regex import W
from scipy import integrate
import scipy
import matplotlib.pyplot as plt
import csv
import time

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

takeoff_hover = 0
l = 1500                        # length of explore window, should be greater than xn^2, timestep dT
fly_time = 1800                 # length of fly window, timestep dT
hover = 500
iter_max = 10                   # max iteration to compute
dT = 0.01                       # timestep

epsilon = 10e-3                 # stop threshold

x0 = np.array([[-1.5],[-0.5],[0],[0],
               [-1.5],[-1.5],[0],[0],
               [-1.5],[0.5],[0],[0]])

waypoints = np.array([[1.5,0],[-1.5,0]])
target = np.array([[waypoints[0][0]],[waypoints[0][1]]])  # target

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

for idx in range(N_agent-1):
    if idx == 0:
        desire = np.array(distance[idx*int(single_xn/2):(idx+1)*int(single_xn/2)])
        desire = np.vstack((desire, np.zeros((int(single_xn/2),1))))
    else:
        desire = np.vstack((desire, distance[idx*int(single_xn/2):(idx+1)*int(single_xn/2)]))
        desire = np.vstack((desire, np.zeros((int(single_xn/2),1))))
desire = np.vstack((desire, target))
desire = np.vstack((desire, np.zeros((int(single_xn/2),1))))

Q = np.eye(xn)*1
Q[(N_agent-1)*4][(N_agent-1)*4] = 0.1
Q[(N_agent-1)*4+1][(N_agent-1)*4+1] = 0.1
R = np.eye(un)*1


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

K = np.array([[-1,0,-1,0,1,0,1,0],[0,-1,0,-1,0,1,0,1],[1,0,1,0,1,0,1,0],[0,1,0,1,0,1,0,1]])
K = np.array([[-0.8,0,-1,0,-0.8,0,-1,0,0.25,0,1,0],
              [0,-0.8,0,-1,0,-0.8,0,-1,0,0.25,0,1],
              [1,0,1.5,0,-0.3,0,-0.5,0,0.25,0,1,0],
              [0,1,0,1.5,0,-0.3,0,-0.5,0,0.25,0,1],
              [-0.3,0,-0.5,0,1,0,1.5,0,0.25,0,1,0],
              [0,-0.3,0,-0.5,0,1,0,1.5,0,0.25,0,1]])

# K = np.zeros((un,xn))
eig2, v2 = np.linalg.eig(A0-np.matmul(B0,K))

u = np.zeros((un, 1))

# explore_dir = np.ones((2*N_agent,1))
# for idx in range(N_agent):
#     if target[0] < x0[4*idx]:
#         explore_dir[2*idx] = -1
#     if target[1] < x0[4*idx+1]:
#         explore_dir[2*idx+1] = -1   

show_target = target.T

X_save = np.transpose(x0)
x0 = np.matmul(T,x0)
X = x0
X = np.vstack((X, np.transpose(np.kron(np.transpose(x0), np.transpose(x0)))))
X = np.vstack((X, np.kron(x0, np.zeros((un, 1)))))

t_save = np.zeros(1)
v_save = np.zeros((2*N_agent,1))

for idx in range(takeoff_hover):
    X_save = np.vstack((X_save,X_save[-1]))
    t_save = np.hstack((t_save, t_save[-1]+dT))

X = np.transpose(X)[-1]

# time_begin = time.time()
# time_used = 0  # initialize the global time as 0
for idx in range(l):
    # t_start = time.time()
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
    show_target = np.vstack((show_target, target.T))
    new_v = np.zeros((2*N_agent,1))
    for j in range(N_agent):
        new_v[j*2] = (new_X.y[j*4+2][-1]-new_X.y[j*4+2][0])/dT
        new_v[j*2+1] = (new_X.y[j*4+3][-1]-new_X.y[j*4+3][0])/dT
    # print(new_v)
    v_save = np.hstack((v_save, new_v))
    X = new_X.y[:,-1]

    # time_used = time.time() - time_begin
    # print("Current Time [sec]: " + str(time_used))

print(X[0])
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


print("Compare to optimal")
print(np.linalg.norm(K-K0))
print('K:')
print(K)
print("Optimal K")
print(K0)

current_target = 0
change_target = np.array([0])
x = np.reshape(X_save[-1], (xn,1))
# K = K0
while current_target < len(waypoints):
    z = np.matmul(T, x)
    X = z - desire
    u = -np.matmul(K, X)
    dx = np.matmul(A, x) + np.matmul(B, u)
    new_v = np.zeros((2*N_agent,1))
    for idx in range(N_agent):
        new_v[idx*2] = dx[idx*4]
        new_v[idx*2+1] = dx[idx*4+1]

    v_save = np.hstack((v_save, new_v))
    x = x + dx*dT
    t_save = np.hstack((t_save, t_save[-1]+dT))
    X_save = np.vstack((X_save, np.transpose(x)))
    show_target = np.vstack((show_target, target.T))
    if LA.norm(X) < 10e-2:
        current_target += 1
        if current_target == len(waypoints):
            break
        change_target = np.append(change_target, t_save[-1])
        target = np.array([[waypoints[current_target][0]],[waypoints[current_target][1]]])
        desire[(N_agent-1)*4] = waypoints[current_target][0]
        desire[(N_agent-1)*4+1] = waypoints[current_target][1]
    


for idx in range(hover):
    z = np.matmul(T, x)
    X = z - desire
    u = -np.matmul(K, X)
    dx = np.matmul(A, x) + np.matmul(B, u)
    new_v = np.zeros((2*N_agent,1))
    for idx in range(N_agent):
        new_v[idx*2] = dx[idx*4]
        new_v[idx*2+1] = dx[idx*4+1]

    v_save = np.hstack((v_save, new_v))
    x = x + dx*dT
    t_save = np.hstack((t_save, t_save[-1]+dT))
    X_save = np.vstack((X_save, np.transpose(x)))
    show_target = np.vstack((show_target, target.T))

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


fig, ax = plt.subplots()

plot_every = 2

show_target = show_target[::10]
target_point, = ax.plot([target[0]], [target[1]], '*')

point, = ax.plot([x_fly[0][0]], [x_fly[0][1]], 'o')
point2, = ax.plot([x_fly[0][4]], [x_fly[0][5]], 'o')
point3, = ax.plot([x_fly[0][8]], [x_fly[0][8]], 'o')
ax.plot
ax.set_xlim([-2.5, 2.5])
ax.set_ylim([-2, 2])
axtext = fig.add_axes([0.0,0.95,0.1,0.05])
axtext.axis("off")
time = axtext.text(0.5,0.5, t_fly[0], ha="left", va="top")

def update_point(n, t_fly, x_fly, point, point2, point3, target_point, target, show_target):
    point.set_data(np.array([x_fly[plot_every*n][0], x_fly[plot_every*n][1]]))
    point2.set_data(np.array([x_fly[plot_every*n][4], x_fly[plot_every*n][5]]))
    point3.set_data(np.array([x_fly[plot_every*n][8], x_fly[plot_every*n][9]]))
    target_point.set_data([show_target[plot_every*n][0]], [show_target[plot_every*n][1]])
    time.set_text(t_fly[plot_every*n])
    return point, point2, point3, target_point, time

ani = FuncAnimation(fig, update_point, int(len(t_fly)/plot_every), fargs=(t_fly, x_fly, point, point2, point3, target_point, target, show_target), repeat=false)

plt.show()


