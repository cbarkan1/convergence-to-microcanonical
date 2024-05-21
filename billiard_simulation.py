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

L1r = 0.9
L1l = -.3
L2 = 1.
right_side_vect = np.array([1,-L2/L1r]) / np.sqrt(1 + L2**2/L1r**2)
left_side_vect = np.array([1,-L2/L1l]) / np.sqrt(1 + L2**2/L1l**2)
triangle = Triangle(L1l,L1r,L2)


x_range_num = np.linspace(L1l,L1r,500)
y_range_num = np.linspace(0,L2,500)
x_mesh_num,y_mesh_num = np.meshgrid(x_range_num,y_range_num)
x_mesh_num2,y_mesh_num2 = x_mesh_num**2, y_mesh_num**2
dA_num = (x_range_num[1] - x_range_num[0]) * (y_range_num[1] - y_range_num[0]) 
def I4_numerical(lambdas):
	# obs_list should be boolean list
	# observables order: [x, y, x2, y2]
	l0,l1,l2,l3,l4 = lambdas[:]
	rhos = exp(-1 - l0 - x_mesh_num*l1 - y_mesh_num*l2 - x_mesh_num2*l3 - y_mesh_num2*l4)
	rhos[y_mesh_num > L2*(1-x_mesh_num/L1l)] = 0.
	rhos[y_mesh_num > L2*(1-x_mesh_num/L1r)] = 0.
	return np.sum(rhos) * dA_num

def I3_numerical(lambdas):
	# obs_list should be boolean list
	# observables order: [x, y, x2, y2]
	l0,l1,l2,l3 = lambdas[:]
	rhos = exp(-1 - l0 - x_mesh_num*l1 - y_mesh_num*l2 - x_mesh_num2*l3)
	rhos[y_mesh_num > L2*(1-x_mesh_num/L1l)] = 0.
	rhos[y_mesh_num > L2*(1-x_mesh_num/L1r)] = 0.
	return np.sum(rhos) * dA_num

def I2_numerical(lambdas):
	# obs_list should be boolean list
	# observables order: [x, y, x2, y2]
	l0,l1,l2 = lambdas[:]
	rhos = exp(-1 - l0 - x_mesh_num*l1 - y_mesh_num*l2)
	rhos[y_mesh_num > L2*(1-x_mesh_num/L1l)] = 0.
	rhos[y_mesh_num > L2*(1-x_mesh_num/L1r)] = 0.
	return np.sum(rhos) * dA_num

def I1_numerical(lambdas):
	# obs_list should be boolean list
	# observables order: [x, y, x2, y2]
	l0,l1 = lambdas[:]
	rhos = exp(-1 - l0 - x_mesh_num*l1)
	rhos[y_mesh_num > L2*(1-x_mesh_num/L1l)] = 0.
	rhos[y_mesh_num > L2*(1-x_mesh_num/L1r)] = 0.
	return np.sum(rhos) * dA_num

def S_G(observables,lambdas0):
	obs = [1] + observables
	N = len(observables)

	if N==1:
		def g(lambdas):
			l0,l1 = lambdas[:]
			Integral = I1_numerical(lambdas)
			return Integral/(0.5*(L1r-L1l)*L2) + np.sum(obs*lambdas)

		result = minimize(g,lambdas0)
		lambdas = result.x

	if N==2:
		def g(lambdas):
			l0,l1,l2 = lambdas[:]
			Integral = I2_numerical(lambdas)
			return Integral/(0.5*(L1r-L1l)*L2) + np.sum(obs*lambdas)

		result = minimize(g,lambdas0)
		lambdas = result.x

	if N==3:
		def g(lambdas):
			l0,l1,l2,l3 = lambdas[:]
			Integral = I3_numerical(lambdas)
			return Integral/(0.5*(L1r-L1l)*L2) + np.sum(obs*lambdas)

		result = minimize(g,lambdas0)
		lambdas = result.x

	if N==4:
		def g(lambdas):
			l0,l1,l2,l3,l4 = lambdas[:]
			Integral = I4_numerical(lambdas)
			return Integral/(0.5*(L1r-L1l)*L2) + np.sum(obs*lambdas)

		result = minimize(g,lambdas0,bounds=((-100,100),(-400,400),(-400,400),(-10,1000),(-10,1000)))
		lambdas = result.x

	return 1 + np.sum(obs*lambdas) ,  result.success , lambdas

def rho3_norm(lambdas):
	Integral = I3_numerical(lambdas)
	return Integral/(0.5*(L1r-L1l)*L2)

def rho4_norm(lambdas):
	Integral = I4_numerical(lambdas)
	return Integral/(0.5*(L1r-L1l)*L2)

def plot_positions(x_list,y_list):
	plt.figure()
	plt.plot(x_list,y_list,'.')
	plt.xlim(L1l-.2,L1r+.2)
	plt.ylim(-.2,L2+.2)
	plt.plot([L1l,L1r],[0,0],'k',linewidth=3)
	plt.plot([L1l,0],[0,L2],'k',linewidth=3)
	plt.plot([L1r,0],[0,L2],'k',linewidth=3)
	plt.gca().set_aspect('equal')
	#plt.show()

def plot_rho4(lambdas):
	l0,l1,l2,l3,l4 = lambdas[:]
	rhos = exp(-1 - l0 - x_mesh_num*l1 - y_mesh_num*l2 - x_mesh_num2*l3 - y_mesh_num2*l4)
	rhos[y_mesh_num > L2*(1-x_mesh_num/L1l)] = np.nan
	rhos[y_mesh_num > L2*(1-x_mesh_num/L1r)] = np.nan
	plt.figure()
	plt.imshow(rhos,origin='lower',extent=(L1l,L1r,0,L2),vmin=0,vmax=2)
	plt.colorbar()
	#plt.show()

def S_S4(x_list,y_list):
	"""
	Shape of partiton
	 /|_\
	/ | |\

	left: x<0
	middle_top (mt): x>=0, y>L2/2
	middle_bottom (mb): 0<=x<L1r/2, y<=L2/2
	right: x>=L1r/2
	"""
	A_left = -0.5*L1l*L2
	A_mt = 0.5*(L1r/2)*(L2/2)
	A_mb = (L1r/2)*(L2/2)
	A_right = 0.5*(L1r/2)*(L2/2)
	A = (0.5*(L1r-L1l)*L2)

	N_left = np.sum(x_list<0)
	N_mt = np.sum(np.logical_and(x_list>=0,y_list>L2/2))
	#N_mb = np.sum(  np.logicaland( np.logicaland(x_list>=0,x_list<L1r/2) , y_list<=L2/2 )   )
	N_right = np.sum( x_list>=L1r/2 )
	N_mb = num_particles - N_left - N_mt - N_right

	rho_left = A*N_left/(A_left*num_particles)
	rho_mt = A*N_mt/(A_mt*num_particles)
	rho_mb = A*N_mb/(A_mb*num_particles)
	rho_right = A*N_right/(A_right*num_particles)

	return -1*(A_left*xlogy(rho_left,rho_left) + A_mt*xlogy(rho_mt,rho_mt) + A_mb*xlogy(rho_mb,rho_mb) + A_right*xlogy(rho_right,rho_right) )


Afc = L1r/3 * L2/3  # Area of full cell
Areas = Afc*np.ones((4,3))
Areas[0,0] = Afc - (L2/3)*(-L1l/3)*0.5
Areas[0,1] = (-L1l*2/3) * (L2/3) - (L2/3)*(-L1l/3)*0.5
Areas[0,2] = (L2/3)*(-L1l/3)*0.5
Areas[1,2] = Afc/2
Areas[2,1] = Afc/2
Areas[3,0] = Afc/2
Total_Area = (0.5*(L1r-L1l)*L2)
def rho_S9(x_list,y_list):
	H,xedges,yedges = np.histogram2d(x_list,y_list,bins=[4,3],range=[[L1l,L1r],[0,L2]])
	rho_S = Total_Area*H/(Areas*num_particles)
	S_S = -1*np.sum(Areas*xlogy(rho_S,rho_S))
	return rho_S,S_S


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
S_1s[0],outcomes1[0],lambdas1 = S_G([x_avgs[0]],[-.1,.1])
S_2s[0],outcomes2[0],lambdas2 = S_G([x_avgs[0],y_avgs[0]],[-.1,.1,.1])
S_3s[0],outcomes3[0],lambdas3 = S_G([x_avgs[0],y_avgs[0],x2_avgs[0]],[-.1,.1,.1,.1])
S_4s[0],outcomes4[0],lambdas4 = S_G([x_avgs[0],y_avgs[0],x2_avgs[0],y2_avgs[0]],[-.1,.1,.1,.1,.1])
temp,S_Ss[0] = rho_S9(x_list,y_list)

#plot_rho4(lambdas4)
#plt.show()
#quit()

for i in range(1,num_ts):
	print(i)
	for particle in range(num_particles):
		dt = ts[i] - ts[i-1]
		x_list[particle],y_list[particle],vx_list[particle],vy_list[particle] = propogate(x_list[particle],y_list[particle],
																							vx_list[particle],vy_list[particle],dt,triangle)
	#plot_positions(xs[:,i],ys[:,i])

	x_avgs[i] = np.sum(x_list)/num_particles
	y_avgs[i] = np.sum(y_list)/num_particles
	x2_avgs[i] = np.sum(x_list**2)/num_particles
	y2_avgs[i] = np.sum(y_list**2)/num_particles
	print(x_avgs[i],y_avgs[i],x2_avgs[i],y2_avgs[i])

	S_1s[i],outcomes1[i],lambdas1 = S_G([x_avgs[i]],lambdas1)
	S_2s[i],outcomes2[i],lambdas2 = S_G([x_avgs[i],y_avgs[i]],lambdas2)
	S_3s[i], outcomes3[i],lambdas3 = S_G([x_avgs[i],y_avgs[i],x2_avgs[i]],lambdas0=lambdas3)
	S_4s[i], outcomes4[i],lambdas4 = S_G([x_avgs[i],y_avgs[i],x2_avgs[i],y2_avgs[i]],lambdas0=lambdas4)
	temp,S_Ss[i] = rho_S9(x_list,y_list)
	print(lambdas4)
	#plot_positions(xs[:,i],ys[:,i])
	#plot_rho4(lambdas4)
	#plt.show()
	print('\n')
	#if outcomes4[i]==False:
	#	print(x_avgs[i],y_avgs[i],x2_avgs[i],y2_avgs[i])
	#	print(lambdas4)
	#	quit()

	
if 0: # Plot trajectories
	particles = [0]
	for particle in particles:
		plt.plot(xs[particle,:],ys[particle,:])
		plt.plot(xs[particle,0],ys[particle,0],'o',color='k')
		plt.plot(xs[particle,-1],ys[particle,-1],'o',color='red')
	plt.xlim(L1l-.2,L1r+.2)
	plt.ylim(-.2,L2+.2)
	plt.plot([L1l,L1r],[0,0],'k',linewidth=3)
	plt.plot([L1l,0],[0,L2],'k',linewidth=3)
	plt.plot([L1r,0],[0,L2],'k',linewidth=3)
	plt.gca().set_aspect('equal')
	plt.show()

print('S_Ss: ',S_Ss)

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
