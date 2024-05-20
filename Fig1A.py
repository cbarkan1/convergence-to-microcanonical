import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Sides:
L1r = 0.9
L1l = -.3
L2 = 1.
A = 0.5*(L1r-L1l)*L2
right_side_vect = np.array([1,-L2/L1r]) / np.sqrt(1 + L2**2/L1r**2)
left_side_vect = np.array([1,-L2/L1l]) / np.sqrt(1 + L2**2/L1l**2)

t_list = np.zeros(3)
def trajectory_collisions(x,y,vx,vy,t):
	# returns only the points of collision, in addition to the initial and final point

	def next_collision(x,y,vx,vy,t):
		t_left = (L2 - L2*x/L1l - y) / (vy + L2*vx/L1l)
		t_bottom = -y/vy
		t_right = (L2 - L2*x/L1r - y) / (vy + L2*vx/L1r)

		t_list[0],t_list[1],t_list[2] = t_left,t_bottom,t_right
		t_list[t_list<=0] = np.inf

		if np.all(t_list>t): # No collision
			return x+vx*t, y+vy*t, vx, vy, None
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
			return x,y,vx,vy,collision_time
	
	xs = [x]
	ys = [y]
	ts = [0]
	for i in range(10000):
		x,y,vx,vy,collision_time = next_collision(xs[-1],ys[-1],vx,vy,t-ts[-1])
		xs += [x]
		ys += [y]
		if collision_time==None:
			ts += [t]
			break
		else:
			ts += [ts[-1]+collision_time]
	else:
		print('reached end of loop without finishing.')
		quit()

	return xs,ys,ts

def plot_with_arrow(ax,xs,ys,color='k',linestyle='-',arrow_style='solid',tail_style='o',length=0.05,angle=0.4):
	ax.plot(xs,ys,linestyle=linestyle,color=color)
	ax.plot(xs[0],ys[0],marker=tail_style,color=color,markersize=7)

	v = np.array([xs[-2] - xs[-1],ys[-2]-ys[-1]])
	v *= length / np.linalg.norm(v)
	R = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
	side1 = R@v # + np.array([xs[-1],ys[-1]])
	side2 = R.T@v # + np.array([xs[-1],ys[-1]])
	if arrow_style=='V':
		ax.plot([xs[-1],xs[-1]+side1[0]],[ys[-1],ys[-1]+side1[1]],color=color)
		ax.plot([xs[-1],xs[-1]+side2[0]],[ys[-1],ys[-1]+side2[1]],color=color)
	elif arrow_style=='solid':
		vertices = [(xs[-1],ys[-1]),(xs[-1]+side1[0],ys[-1]+side1[1]),(xs[-1]+side2[0],ys[-1]+side2[1])]
		triangle = patches.Polygon(vertices, closed=True, color='black')
		ax.add_patch(triangle)


if 0: # Panel A
	t = 5.8
	xs,ys,ts = trajectory_collisions(-.1,.25,.6,-.4,t)

	plt.figure(figsize=(3.5,3.5))
	ax = plt.gca()
	


	#Axes:
	ax.axis('off')
	ax_color='grey'
	plot_with_arrow(ax,[0,L1r+0.2],[0,0],color=ax_color,arrow_style='V',tail_style=None)
	plot_with_arrow(ax,[0,L1l-.2],[0,0],color=ax_color,arrow_style='V',tail_style=None)
	plot_with_arrow(ax,[0,0],[0,L2+0.2],color=ax_color,arrow_style='V',tail_style=None)
	plot_with_arrow(ax,[0,0],[0,-0.2],color=ax_color,arrow_style='V',tail_style=None)

	plot_with_arrow(ax,xs,ys,linestyle='--',length=0.08)
	
	#tick_size = .03
	#ax.plot([0,0],[0,-tick_size],'k')
	#ax.plot([0,-tick_size],[0,0],'k')
	#ax.plot([L1r,L1r],[0,-tick_size],'k')
	#ax.plot([0,-tick_size],[L2,L2],'k')

	#Plot triangle
	linewidth=3
	axis_shift = 0.003
	plt.plot([L1l+axis_shift,L1r],[axis_shift,axis_shift],'k',linewidth=linewidth,solid_capstyle='round')
	plt.plot([L1l+axis_shift,axis_shift],[axis_shift,L2],'k',linewidth=linewidth,solid_capstyle='round')
	plt.plot([L1r,axis_shift],[axis_shift,L2],'k',linewidth=linewidth,solid_capstyle='round')
	
	ax.set_aspect('equal')
	plt.savefig('fig1A.pdf',transparent=True)
	plt.show()



if 0: # Panel B
	plt.figure(figsize=(3.5,3.5))
	ax = plt.gca()
	vertices = [(-.16,0.08),(-.06,0.08),(-.06,0.18),(-.16,.18)]
	rectangle = patches.Polygon(vertices, closed=True, color='#007FFF')
	ax.add_patch(rectangle)

	#Axes:
	ax.axis('off')
	ax_color='grey'
	plot_with_arrow(ax,[0,L1r+0.2],[0,0],color=ax_color,arrow_style='V',tail_style=None)
	plot_with_arrow(ax,[0,L1l-.2],[0,0],color=ax_color,arrow_style='V',tail_style=None)
	plot_with_arrow(ax,[0,0],[0,L2+0.2],color=ax_color,arrow_style='V',tail_style=None)
	plot_with_arrow(ax,[0,0],[0,-0.2],color=ax_color,arrow_style='V',tail_style=None)

	#Plot triangle
	linewidth=3
	axis_shift = 0.003
	plt.plot([L1l+axis_shift,L1r],[axis_shift,axis_shift],'k',linewidth=linewidth,solid_capstyle='round')
	plt.plot([L1l+axis_shift,axis_shift],[axis_shift,L2],'k',linewidth=linewidth,solid_capstyle='round')
	plt.plot([L1r,axis_shift],[axis_shift,L2],'k',linewidth=linewidth,solid_capstyle='round')
	
	ax.set_aspect('equal')
	plt.savefig('fig1B.pdf',transparent=True)
	plt.show()



if 1: # Panel C
	plt.figure(figsize=(3.5,3.5))
	ax = plt.gca()

	#Axes:
	ax.axis('off')
	ax_color='grey'
	plot_with_arrow(ax,[0,L1r+0.2],[0,0],color=ax_color,arrow_style='V',tail_style=None)
	plot_with_arrow(ax,[0,L1l-.2],[0,0],color=ax_color,arrow_style='V',tail_style=None)
	plot_with_arrow(ax,[0,0],[0,L2+0.2],color=ax_color,arrow_style='V',tail_style=None)
	plot_with_arrow(ax,[0,0],[0,-0.2],color=ax_color,arrow_style='V',tail_style=None)


	plt.plot([0,0],[0,L2],':k')
	plt.plot([.3,.3],[0,2*L2/3],':k')
	plt.plot([.6,.6],[0,L2/3],':k')
	plt.plot([2*L1l/3,.6],[L2/3,L2/3],':k')
	plt.plot([L1l/3,.3],[2*L2/3,2*L2/3],':k')

	#Plot triangle
	linewidth=3
	axis_shift = 0.003
	plt.plot([L1l+axis_shift,L1r],[axis_shift,axis_shift],'k',linewidth=linewidth,solid_capstyle='round')
	plt.plot([L1l+axis_shift,axis_shift],[axis_shift,L2],'k',linewidth=linewidth,solid_capstyle='round')
	plt.plot([L1r,axis_shift],[axis_shift,L2],'k',linewidth=linewidth,solid_capstyle='round')
	
	ax.set_aspect('equal')
	plt.show()




#



