"""
Modified so I don't keep track of the entire list of particle positions at all times.
Hopefully gonna speed things up.

"""

import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
from time import time
from scipy.optimize import minimize
from scipy.special import xlogy
from scipy.special import erf as serf
import math
from billiard_funcs import Triangle, propogate
from coarse_graining_classes import *

L1r = 0.9
L1l = -.3
L2 = 1.
triangle = Triangle(L1l,L1r,L2)
rho_num = Rho_numerical(triangle,500,500)
standardCG = standardCG_9(triangle)

time0 = time()

num_particles = 200

ts = np.concatenate(  ( [0] , np.exp(np.linspace(np.log(0.01),np.log(10),5)) )  )
num_ts = len(ts)

# Initial conditions
x_list = -.16 + 0.1*np.random.uniform(size=num_particles)
y_list = 0.08 + 0.1*np.random.uniform(size=num_particles)
thetas = np.random.uniform(size=num_particles)*2*np.pi
vx_list = np.cos(thetas)[:]
vy_list = np.sin(thetas)[:]
x_list0 = x_list.copy()
y_list0 = y_list.copy()
vx_list0 = vx_list.copy()
vy_list0 = vy_list.copy()



x_avgs = np.zeros(num_ts)
y_avgs = np.zeros(num_ts)
x2_avgs = np.zeros(num_ts)
y2_avgs = np.zeros(num_ts)
S_1s = np.zeros(num_ts)
S_2s = np.zeros(num_ts)
S_3s = np.zeros(num_ts)
S_4s = np.zeros(num_ts)
S_Ss = np.zeros(num_ts)
outcomes1 = np.zeros(num_ts)
outcomes2 = np.zeros(num_ts)
outcomes3 = np.zeros(num_ts)
outcomes4 = np.zeros(num_ts)

x_avgs[0] = np.sum(x_list)/num_particles
y_avgs[0] = np.sum(y_list)/num_particles
x2_avgs[0] = np.sum(x_list**2)/num_particles
y2_avgs[0] = np.sum(y_list**2)/num_particles
S_1s[0],outcomes1[0],lambdas1 = rho_num.S_G([x_avgs[0]],np.array([-.1,.1]))
S_2s[0],outcomes2[0],lambdas2 = rho_num.S_G([x_avgs[0],y_avgs[0]],np.array([-.1,.1,.1]))
S_3s[0],outcomes3[0],lambdas3 = rho_num.S_G([x_avgs[0],y_avgs[0],x2_avgs[0]],np.array([-.1,.1,.1,.1]))
S_4s[0],outcomes4[0],lambdas4 = rho_num.S_G([x_avgs[0],y_avgs[0],x2_avgs[0],y2_avgs[0]],np.array([-.1,.1,.1,.1,.1]))
temp,S_Ss[0] = standardCG.rho_and_S(x_list,y_list)


for i in range(1,num_ts):
	print(i)
	for particle in range(num_particles):
		dt = ts[i] - ts[i-1]
		x_list[particle],y_list[particle],vx_list[particle],vy_list[particle] = propogate(x_list[particle],y_list[particle],
																							vx_list[particle],vy_list[particle],dt,triangle)

	x_avgs[i] = np.sum(x_list)/num_particles
	y_avgs[i] = np.sum(y_list)/num_particles
	x2_avgs[i] = np.sum(x_list**2)/num_particles
	y2_avgs[i] = np.sum(y_list**2)/num_particles
	print(x_avgs[i],y_avgs[i],x2_avgs[i],y2_avgs[i])

	S_1s[i],outcomes1[i],lambdas1 = rho_num.S_G([x_avgs[i]],lambdas1)
	S_2s[i],outcomes2[i],lambdas2 = rho_num.S_G([x_avgs[i],y_avgs[i]],lambdas2)
	S_3s[i], outcomes3[i],lambdas3 = rho_num.S_G([x_avgs[i],y_avgs[i],x2_avgs[i]],lambdas3)
	S_4s[i], outcomes4[i],lambdas4 = rho_num.S_G([x_avgs[i],y_avgs[i],x2_avgs[i],y2_avgs[i]],lambdas4)
	temp,S_Ss[i] = standardCG.rho_and_S(x_list,y_list)
	print(lambdas4)
	print('\n')



time1 = time()
print('\n Time = ',time1-time0,'\n')

plt.plot(ts, S_1s)
plt.plot(ts, S_2s)
plt.plot(ts, S_3s)
plt.plot(ts, S_4s)
plt.plot(ts, S_Ss,':')
plt.plot(ts,outcomes1)
plt.plot(ts,outcomes2)
plt.plot(ts,outcomes3)
plt.plot(ts,outcomes4)
plt.xscale('log')
plt.legend(['1','2','3','4','S','o1','o2','o3','o4'])
plt.show()

#np.savez('sim2.npz',ts=ts,x_avgs=x_avgs,y_avgs=y_avgs,x2_avgs=x2_avgs,y2_avgs=y2_avgs,S_1s=S_1s,S_2s=S_2s,S_3s=S_3s,S_4s=S_4s,S_Ss=S_Ss,x_list0=x_list0,y_list0=y_list0,vx_list0=vx_list0,vy_list0=vy_list0)


#
