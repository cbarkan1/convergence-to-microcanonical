import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import xlogy
import matplotlib.patches as patches


file = np.load('../Particle in an irrational triangle/sim1.npz')
x_list0 = file['x_list0']
y_list0 = file['y_list0']
vx_list0 = file['vx_list0']
vy_list0 = file['vy_list0']
num_particles = len(x_list0)


L1r = 0.9
L1l = -.3
L2 = 1.
right_side_vect = np.array([1,-L2/L1r]) / np.sqrt(1 + L2**2/L1r**2)
left_side_vect = np.array([1,-L2/L1l]) / np.sqrt(1 + L2**2/L1l**2)

t_list = np.zeros(3)
def propogate(x,y,vx,vy,t):
	t_left = (L2 - L2*x/L1l - y) / (vy + L2*vx/L1l)
	t_bottom = -y/vy
	t_right = (L2 - L2*x/L1r - y) / (vy + L2*vx/L1r)

	t_list[0],t_list[1],t_list[2] = t_left,t_bottom,t_right
	t_list[t_list<=0] = np.inf

	if np.all(t_list>t): # No collision
		return x+vx*t, y+vy*t, vx, vy	
	else:
		collision_wall = np.argmin(t_list)
		collision_time = t_list[collision_wall]
		if collision_wall==0: #left
			x += vx*collision_time
			y = L2*(1.-x/L1l) - 0.00000001
			v_parallel = left_side_vect * (vx*left_side_vect[0] + vy*left_side_vect[1])
			v_perpendicular = [vx - v_parallel[0], vy - v_parallel[1]]
			vx = v_parallel[0] - v_perpendicular[0]
			vy = v_parallel[1] - v_perpendicular[1]
		elif collision_wall==1: #bottom
			y = 0.
			x += vx*collision_time
			vy *= -1
		else: #hypotenuse
			x += vx*collision_time
			y = L2*(1.-x/L1r) - 0.00000001
			v_parallel = right_side_vect * (vx*right_side_vect[0] + vy*right_side_vect[1])
			v_perpendicular = [vx - v_parallel[0], vy - v_parallel[1]]
			vx = v_parallel[0] - v_perpendicular[0]
			vy = v_parallel[1] - v_perpendicular[1]
		return propogate(x,y,vx,vy,t - collision_time)

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




ts = [0,.3,5]



fig, ax = plt.subplots(3,3,figsize=(13,9))

for i in range(3):
	x_list, y_list = np.zeros(num_particles),np.zeros(num_particles)
	for particle in range(num_particles):
		x_list[particle],y_list[particle],temp1,temp2 = propogate(x_list0[particle],y_list0[particle],vx_list0[particle],vy_list0[particle],ts[i])


	#rhos = get_rhoG(x_avg[indices[i]],y_avg[indices[i]],x2_avg[indices[i]])


	ax[0,i].plot(x_list,y_list,'.',markersize=5)
	linewidth=3
	axis_shift = 0.003
	ax[0,i].plot([L1l+axis_shift,L1r],[axis_shift,axis_shift],'k',linewidth=linewidth,solid_capstyle='round')
	ax[0,i].plot([L1l+axis_shift,axis_shift],[axis_shift,L2],'k',linewidth=linewidth,solid_capstyle='round')
	ax[0,i].plot([L1r,axis_shift],[axis_shift,L2],'k',linewidth=linewidth,solid_capstyle='round')
	ax[0,i].axis('off')
	ax[0,i].set_aspect('equal')
	#ax[0,i].imshow(rhos*0,origin='lower',extent=(0,L1,0,L2),vmin=-1,vmax=0,cmap='gray',alpha=0)


	rho_S,temp = rho_S9(x_list,y_list)
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
	S_4, outcome, lambdas = S_G([x_avgs,y_avgs,x2_avgs,y2_avgs],[-.1,.1,.1,.1,.1])
	l0,l1,l2,l3,l4 = lambdas[:]
	rhos = exp(-1 - l0 - x_mesh_num*l1 - y_mesh_num*l2 - x_mesh_num2*l3 - y_mesh_num2*l4)
	rhos[y_mesh_num > L2*(1-x_mesh_num/L1l)] = np.nan
	rhos[y_mesh_num > L2*(1-x_mesh_num/L1r)] = np.nan
	ax[2,i].imshow(rhos,origin='lower',extent=(L1l,L1r,0,L2),vmin=0,vmax=3)
	#plt.colorbar()
	#ax[2,i].imshow(rhos,origin='lower',extent=(0,L1,0,L2),vmin=0,vmax=2)
	ax[2,i].set_aspect('equal')
	ax[2,i].axis('off')



plt.subplots_adjust(wspace=0.2,hspace=0.05)
plt.savefig('fig1D.pdf',transparent=True)


# colorbar
plt.figure(figsize=(3.5,3.4))
im = plt.imshow(rhos,origin='lower',extent=(0,L1r,0,L2),vmin=0,vmax=3)
plt.gca().axis('off')
plt.colorbar(aspect=10)
plt.savefig('fig1D_cb.pdf',transparent=True)


plt.show()

#