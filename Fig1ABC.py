"""
Generates Figure 1 panels A, B, and C
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from billiard_funcs import trajectory_collisions
from plot_with_arrow import plot_with_arrow

# Specifying the shape of the triangle
# Vertex coordinates are (L1r,0), (L1l,0), (0,L2)
#   /|\    |
#  / | \   L2
# /__|__\  |
# L1l L1R
L1r = 0.9
L1l = -.3
L2 = 1.

# Initializing figure, subplots a,b,c correspond to panels A, B, C
fig, (a,b,c) = plt.subplots(1,3,figsize=(10.5,3.5))


###
### Panel A (Plotting a trajectory)
###
x0 = -.1 # Initial x
y0 = 0.25 # Initial y
vx0 = 0.6 # Initial velocity vx
vy0 = -0.4 # Initial velocity vy
t = 5.8 # Time for trajectory

# Find coordinates of collisions with walls, ts=collision times
xs,ys,ts = trajectory_collisions(x0,y0,vx0,vy0,t,L1l,L1r,L2)
plot_with_arrow(a,xs,ys,linestyle='--',length=0.08)

###
### Panel B (Initial distribution)
###
vertices = [(-.16,0.08),(-.06,0.08),(-.06,0.18),(-.16,.18)]
rectangle = patches.Polygon(vertices, closed=True, color='#007FFF')
b.add_patch(rectangle)

###
### Panel C (Partition for standard coarse-graining)
###
c.plot([0,0],[0,L2],':k')
c.plot([.3,.3],[0,2*L2/3],':k')
c.plot([.6,.6],[0,L2/3],':k')
c.plot([2*L1l/3,.6],[L2/3,L2/3],':k')
c.plot([L1l/3,.3],[2*L2/3,2*L2/3],':k')

# Plot axes and triangle for each subplot.
for ax in [a,b,c]:
	
	# Turn off default axes:
	ax.axis('off')

	# Plot customized axes:
	ax_color='grey' #color of axis lines
	axis_extension = 0.2 #how far axes extend past triangle
	plot_with_arrow(ax,[0,L1r+axis_extension],[0,0],color=ax_color,
                    arrow_style='V',tail_style=None)
	plot_with_arrow(ax,[0,L1l-axis_extension],[0,0],color=ax_color,
                    arrow_style='V',tail_style=None)
	plot_with_arrow(ax,[0,0],[0,L2+axis_extension],color=ax_color,
                    arrow_style='V',tail_style=None)
	plot_with_arrow(ax,[0,0],[0,-axis_extension],color=ax_color,
                    arrow_style='V',tail_style=None)

	#Plot triangle
	linewidth=3
    """ Sides of the triangle are plotted with solid_capstyle='round' to create 
    rounded corners and avoid the sharp square corners of each line."""
	ax.plot([L1l,L1r],[0,0],'k',linewidth=linewidth,solid_capstyle='round')
	ax.plot([L1l,0],[0,L2],'k',linewidth=linewidth,solid_capstyle='round')
	ax.plot([L1r,0],[0,L2],'k',linewidth=linewidth,solid_capstyle='round')

	#Equalize aspect ratio of x and y axes
	ax.set_aspect('equal')

plt.show()







