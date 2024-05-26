"""
Note to self: do this after updating code for billiard simulation.


"""

import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import xlogy
import matplotlib.patches as patches
from billiard_utils import *
from coarse_graining_classes import *

# Load saved data.
# simulation_data.npz can be generated using billiard_simulation.py
file = np.load('simulation_data.npz')
x_list = file['x_list0']
y_list = file['y_list0']
vx_list = file['vx_list0']
vy_list = file['vy_list0']

triangle = Triangle(*file['triangle_data'][:])
standardCG = StandardCG_9(triangle)
rho_num = Rho_numerical(triangle,500,500)
num_particles = len(x_list)


L1r = 0.9
L1l = -.3
L2 = 1.


ts = [0,.3,5]


dts = np.diff(ts,prepend=0)


fig, ax = plt.subplots(3,3,figsize=(13,9))

for i in range(3):

	x_list, y_list,temp1,temp2 = propogate_list(x_list,y_list,vx_list,vy_list,dts[i],triangle)


	ax[0,i].plot(x_list,y_list,'.',markersize=5)
	linewidth=3
	ax[0,i].plot([L1l,L1r],[0,0],'k',linewidth=linewidth,solid_capstyle='round')
	ax[0,i].plot([L1l,0],[0,L2],'k',linewidth=linewidth,solid_capstyle='round')
	ax[0,i].plot([L1r,0],[0,L2],'k',linewidth=linewidth,solid_capstyle='round')
	ax[0,i].axis('off')
	ax[0,i].set_aspect('equal')


	rho_S,temp = standardCG.rho_and_S(x_list,y_list)
	rho_S = rho_S.T / 3 # Modifations made for plotting
	#print(rho_S)
	#quit()
	"""
	20  21  --  --
	10  11  12  --
	00  01  02  03
	"""
	cmap = plt.cm.viridis

	cell00 = patches.Polygon([(L1l,0),(0,0),(0,L2/3),(2*L1l/3,L2/3)],closed=True,color=cmap(rho_S[0,0]))
	cell10 = patches.Polygon([(2*L1l/3,L2/3),(0,L2/3),(0,2*L2/3),(L1l/3,2*L2/3)],closed=True,color=cmap(rho_S[1,0]))  
	cell20 = patches.Polygon([(L1l/3,2*L2/3),(0,2*L2/3),(0,L2)],closed=True,color=cmap(rho_S[2,0]))  

	cell01 = patches.Polygon([(0,0),(0,L2/3),(L1r/3,L2/3),(L1r/3,0)],closed=True,color=cmap(rho_S[0,1]))
	cell02 = patches.Polygon([(L1r/3,0),(2*L1r/3,0),(2*L1r/3,L2/3),(L1r/3,L2/3)],closed=True,color=cmap(rho_S[0,2]))
	cell03 = patches.Polygon([(2*L1r/3,0),(L1r,0),(2*L1r/3,L2/3)],closed=True,color=cmap(rho_S[0,3]))
	
	cell11 = patches.Polygon([(0,L2/3),(L1r/3,L2/3),(L1r/3,2*L2/3),(0,2*L2/3)],closed=True,color=cmap(rho_S[1,1]))
	cell12 = patches.Polygon([(L1r/3,L2/3),(2*L1r/3,L2/3),(L1r/3,2*L2/3)],closed=True,color=cmap(rho_S[1,2]))

	cell21 = patches.Polygon([(0,2*L2/3),(L1r/3,2*L2/3),(0,L2)],closed=True,color=cmap(rho_S[2,1]))

	ax[1,i].add_patch(cell00)
	ax[1,i].add_patch(cell10)
	ax[1,i].add_patch(cell20)
	ax[1,i].add_patch(cell01)
	ax[1,i].add_patch(cell02)
	ax[1,i].add_patch(cell03)
	ax[1,i].add_patch(cell11)
	ax[1,i].add_patch(cell12)
	ax[1,i].add_patch(cell21)

	ax[1,i].set_xlim(L1l,L1r)
	ax[1,i].set_ylim(0,L2)

	ax[1,i].set_aspect('equal')
	ax[1,i].axis('off')



	x_avgs = np.sum(x_list)/num_particles
	y_avgs = np.sum(y_list)/num_particles
	x2_avgs = np.sum(x_list**2)/num_particles
	y2_avgs = np.sum(y_list**2)/num_particles
	S_4, outcome, lambdas = rho_num.S_G([x_avgs,y_avgs,x2_avgs,y2_avgs],[-.1,.1,.1,.1,.1])
	rhos = rho_num.rho(lambdas,mask_val=np.nan)
	ax[2,i].imshow(rhos,origin='lower',extent=(L1l,L1r,0,L2),vmin=0,vmax=3)
	#plt.colorbar()
	#ax[2,i].imshow(rhos,origin='lower',extent=(0,L1,0,L2),vmin=0,vmax=2)
	ax[2,i].set_aspect('equal')
	ax[2,i].axis('off')



plt.subplots_adjust(wspace=0.2,hspace=0.05)


# Plot colorbar in separate figure
plt.figure(figsize=(3.5,3.4))
im = plt.imshow(np.nan*np.zeros((2,2)),origin='lower',extent=(0,L1r,0,L2),vmin=0,vmax=3)
plt.gca().axis('off')
plt.colorbar(aspect=10)

plt.show()

#