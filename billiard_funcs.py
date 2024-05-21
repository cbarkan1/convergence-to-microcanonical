import numpy as np


def trajectory_collisions(x,y,vx,vy,t,L1l,L1r,L2):
	# returns only the points of collision, in addition to the initial and final point

	right_side_vect = np.array([1,-L2/L1r]) / np.sqrt(1 + L2**2/L1r**2)
	left_side_vect = np.array([1,-L2/L1l]) / np.sqrt(1 + L2**2/L1l**2)

	t_list = np.zeros(3)
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

