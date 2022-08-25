import re
import numpy as np
import csv
import matplotlib.pyplot as plt

with open('trajectory1.csv') as trajfile:
    traj = np.loadtxt(trajfile, delimiter=",")
    
with open('trajectory2.csv') as trajfile:
    traj2 = np.loadtxt(trajfile, delimiter=",")
    


plt.figure(1)
fig, ax = plt.subplots()
ax.set_xlabel('t (s)')
ax.set_ylabel('x (m)')
ax.set_title('Formation')
ax.plot(traj[0],traj[1])

plt.figure(2)
fig, ax = plt.subplots()
ax.set_xlabel('t (s)')
ax.set_ylabel('y (m)')
ax.set_title('Formation')
ax.plot(traj[0],traj[2])

plt.figure(3)
fig, ax = plt.subplots()
ax.set_xlabel('t (s)')
ax.set_ylabel('vx (m)')
ax.set_title('Formation')
ax.plot(traj[0],traj[4])

plt.figure(4)
fig, ax = plt.subplots()
ax.set_xlabel('t (s)')
ax.set_ylabel('vy (m)')
ax.set_title('Formation')
ax.plot(traj[0],traj[5])

plt.figure(5)
fig, ax = plt.subplots()
ax.set_xlabel('t (s)')
ax.set_ylabel('x (m)')
ax.set_title('Formation')
ax.plot(traj[0],traj[1])

plt.figure(6)
fig, ax = plt.subplots()
ax.set_xlabel('t (s)')
ax.set_ylabel('y (m)')
ax.set_title('Formation')
ax.plot(traj[0],traj[2])

plt.figure(7)
fig, ax = plt.subplots()
ax.set_xlabel('t (s)')
ax.set_ylabel('vx (m)')
ax.set_title('Formation')
ax.plot(traj2[0],traj2[4])

plt.figure(8)
fig, ax = plt.subplots()
ax.set_xlabel('t (s)')
ax.set_ylabel('vy (m)')
ax.set_title('Formation')
ax.plot(traj2[0],traj2[5])

plt.show()