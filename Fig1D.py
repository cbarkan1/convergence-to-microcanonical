"""
Create Figure 1 panel D

"""
import numpy as np
import matplotlib.pyplot as plt
from billiard_utils import Triangle, propogate_list
from coarse_graining_classes import Rho_numerical, StandardCG_9

FILENAME = 'simulation_data.npz'
TIMES = [0, 0.3, 5]


# Load saved data.
# simulation_data.npz can be generated using billiard_simulation.py
file = np.load(FILENAME)
x_list = file['x_list0']
y_list = file['y_list0']
vx_list = file['vx_list0']
vy_list = file['vy_list0']

triangle = Triangle(*file['triangle_data'][:])
standardCG = StandardCG_9(triangle)
rho_num = Rho_numerical(triangle, x_gridsize=500, y_gridsize=500)
num_particles = len(x_list)

dts = np.diff(TIMES, prepend=0)
fig, ax = plt.subplots(3, len(TIMES), figsize=(13, 9))

# Loop over columns in figure
for i in range(len(TIMES)):
    x_list, y_list, temp1, temp2 = propogate_list(
        x_list, y_list, vx_list, vy_list, dts[i], triangle)

    # Row 1
    ax[0, i].plot(x_list, y_list, '.', markersize=5)
    linewidth = 3
    ax[0, i].plot([triangle.L1l, triangle.L1r], [0, 0], 'k',
                  linewidth=linewidth, solid_capstyle='round')
    ax[0, i].plot([triangle.L1l, 0], [0, triangle.L2], 'k',
                  linewidth=linewidth, solid_capstyle='round')
    ax[0, i].plot([triangle.L1r, 0], [0, triangle.L2], 'k',
                  linewidth=linewidth, solid_capstyle='round')
    ax[0, i].axis('off')
    ax[0, i].set_aspect('equal')

    # Row 2
    rho_S, temp = standardCG.rho_and_S(x_list, y_list)
    standardCG.plot_rho(ax[1, i], rho_S)
    ax[1, i].set_xlim(triangle.L1l, triangle.L1r)
    ax[1, i].set_ylim(0, triangle.L2)
    ax[1, i].set_aspect('equal')
    ax[1, i].axis('off')

    # Row 3
    x_avgs = np.sum(x_list)/num_particles
    y_avgs = np.sum(y_list)/num_particles
    x2_avgs = np.sum(x_list**2)/num_particles
    y2_avgs = np.sum(y_list**2)/num_particles
    S_4, outcome, lambdas = rho_num.S_G([x_avgs, y_avgs, x2_avgs, y2_avgs],
                                        [-.1, .1, .1, .1, .1])
    rhos = rho_num.rho(lambdas, mask_val=np.nan)
    ax[2, i].imshow(rhos, origin='lower',
                    extent=(triangle.L1l, triangle.L1r, 0, triangle.L2),
                    vmin=0, vmax=3)
    ax[2, i].set_aspect('equal')
    ax[2, i].axis('off')

plt.subplots_adjust(wspace=0.2, hspace=0.05)

# Plot colorbar in separate figure
plt.figure(figsize=(3.5, 3.4))
im = plt.imshow(np.nan*np.zeros((2, 2)), origin='lower',
                extent=(0, triangle.L1r, 0, triangle.L2), vmin=0, vmax=3)
plt.gca().axis('off')
plt.colorbar(aspect=10)

plt.show()
