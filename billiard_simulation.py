"""
Simulation of ensemble dynamics and computing coarse-grained quantities.

Running this script does the following:
1) Simulates an ensemble with initial condition as in Fig. 1B
2) Shows summary plot of results
3) Saves an npz file that is read by Fig1D.py and Fig2.py.
"""
import numpy as np
import matplotlib.pyplot as plt
from billiard_utils import Triangle, propogate_list
from coarse_graining_classes import Rho_numerical, StandardCG_9


def simulation(triangle, num_particles, ts, filename=None):
    """
    Simulate an ensemble of particles in the triangle and save results.

    Parameters:
        triangle (Triangle): Triangle object
        num_particles (int): Number of particles
        ts (list): list of times
        filename (str or None): If str, simulation results will be saved
                                as an npz file to filename.

    Returns:
        tuples: 1D numpy arrays of x averages, y averages,
                x**2 averages, y**2 averages, generalized coarse-grained
                entropies (S_1, S_2,S_3,S_4), and standard coarse-grain
                entropy.
    """

    def compute_avgs(x_list, y_list):
        """Computes avgs of observables"""
        x_avg = np.sum(x_list)/num_particles
        y_avg = np.sum(y_list)/num_particles
        x2_avg = np.sum(x_list**2)/num_particles
        y2_avg = np.sum(y_list**2)/num_particles
        return x_avg, y_avg, x2_avg, y2_avg

    num_ts = len(ts)
    rho_num = Rho_numerical(triangle, 500, 500)
    standardCG = StandardCG_9(triangle)

    # Initial conditions drawn randomly as described in Fig. 1B
    x_list = -.16 + 0.1*np.random.uniform(size=num_particles)
    y_list = 0.08 + 0.1*np.random.uniform(size=num_particles)
    thetas = np.random.uniform(size=num_particles)*2*np.pi
    vx_list = np.cos(thetas)[:]
    vy_list = np.sin(thetas)[:]

    # Saving the initial condition
    x_list0 = x_list.copy()
    y_list0 = y_list.copy()
    vx_list0 = vx_list.copy()
    vy_list0 = vy_list.copy()

    # Initializing arrays to store simulation results
    x_avgs = np.zeros(num_ts)  # x avgs
    y_avgs = np.zeros(num_ts)  # y avgs
    x2_avgs = np.zeros(num_ts)  # x**2 avgs
    y2_avgs = np.zeros(num_ts)  # y**2 avgs
    S_1s = np.zeros(num_ts)  # generalized entropy with N=1
    S_2s = np.zeros(num_ts)  # generalized entropy with N=2
    S_3s = np.zeros(num_ts)  # generalized entropy with N=3
    S_4s = np.zeros(num_ts)  # generalized entropy with N=4
    S_Ss = np.zeros(num_ts)  # Standard coare-grained entropies
    outcomes1 = np.zeros(num_ts)  # Outcomes of S_1s optimizations
    outcomes2 = np.zeros(num_ts)  # Outcomes of S_2s optimizations
    outcomes3 = np.zeros(num_ts)  # Outcomes of S_3s optimizations
    outcomes4 = np.zeros(num_ts)  # Outcomes of S_4s optimizations

    x_avgs[0], y_avgs[0], x2_avgs[0], y2_avgs[0] = compute_avgs(x_list, y_list)
    S_1s[0], outcomes1[0], lambdas1 = rho_num.S_G(
        [x_avgs[0]], np.array([-.1, .1]))
    S_2s[0], outcomes2[0], lambdas2 = rho_num.S_G(
        [x_avgs[0], y_avgs[0]], np.array([-.1, .1, .1]))
    S_3s[0], outcomes3[0], lambdas3 = rho_num.S_G(
        [x_avgs[0], y_avgs[0], x2_avgs[0]], np.array([-.1, .1, .1, .1]))
    S_4s[0], outcomes4[0], lambdas4 = rho_num.S_G(
        [x_avgs[0], y_avgs[0], x2_avgs[0], y2_avgs[0]],
        np.array([-.1, .1, .1, .1, .1]))
    temp, S_Ss[0] = standardCG.rho_and_S(x_list, y_list)

    for i in range(1, num_ts):
        print('iter', i, 'of', num_ts-1)
        dt = ts[i] - ts[i-1]
        x_list, y_list, vx_list, vy_list = propogate_list(
            x_list, y_list, vx_list, vy_list, dt, triangle)

        x_avgs[i], y_avgs[i], x2_avgs[i], y2_avgs[i] = compute_avgs(x_list,
                                                                    y_list)
        S_1s[i], outcomes1[i], lambdas1 = rho_num.S_G([x_avgs[i]], lambdas1)
        S_2s[i], outcomes2[i], lambdas2 = rho_num.S_G(
            [x_avgs[i], y_avgs[i]], lambdas2)
        S_3s[i], outcomes3[i], lambdas3 = rho_num.S_G(
            [x_avgs[i], y_avgs[i], x2_avgs[i]], lambdas3)
        S_4s[i], outcomes4[i], lambdas4 = rho_num.S_G(
            [x_avgs[i], y_avgs[i], x2_avgs[i], y2_avgs[i]], lambdas4)
        temp, S_Ss[i] = standardCG.rho_and_S(x_list, y_list)

    if filename is not None:
        np.savez(filename, ts=ts, x_avgs=x_avgs, y_avgs=y_avgs,
                 x2_avgs=x2_avgs, y2_avgs=y2_avgs, S_1s=S_1s, S_2s=S_2s,
                 S_3s=S_3s, S_4s=S_4s, S_Ss=S_Ss, x_list0=x_list0,
                 y_list0=y_list0, vx_list0=vx_list0, vy_list0=vy_list0,
                 triangle_data=triangle.L_array)

    return x_avgs, y_avgs, x2_avgs, y2_avgs, S_1s, S_2s, S_3s, S_4s, S_Ss


def plot_results(ts, x_avgs, y_avgs, x2_avgs, y2_avgs, S_1s,
                 S_2s, S_3s, S_4s, S_Ss):
    """
    Plot results of simulation.
    Not for publication-ready plots.
    """
    plt.subplot(1, 2, 1)
    plt.plot(ts, x_avgs)
    plt.plot(ts, y_avgs)
    plt.plot(ts, x2_avgs)
    plt.plot(ts, y2_avgs)
    plt.xscale('log')
    plt.legend(['x', 'y', 'x^2', 'y^2'])

    plt.subplot(1, 2, 2)
    plt.plot(ts, S_1s)
    plt.plot(ts, S_2s)
    plt.plot(ts, S_3s)
    plt.plot(ts, S_4s)
    plt.plot(ts, S_Ss, ':')
    plt.xscale('log')
    plt.legend(['1', '2', '3', '4', 'Stand. CG'])
    plt.show()


def main():
    """
    1) Simulates an ensemble with initial condition as in Fig. 1B
    2) Shows summary plot of results
    3) Saves an npz file that is read by Fig.1D.py and Fig2.py.
    """

    # Define triangle shape
    L1r = 0.9
    L1l = -.3
    L2 = 1.
    triangle = Triangle(L1l, L1r, L2)

    NUM_PARTICLES = 200
    TIME_STEPS = 500
    FIRST_NONZERO_TIME = 0.01
    FINAL_TIME = 100
    FILENAME = None

    # log-scale ts
    ts = np.concatenate(([0], np.exp(np.linspace(np.log(FIRST_NONZERO_TIME),
                                     np.log(FINAL_TIME), TIME_STEPS))))

    x_avgs, y_avgs, x2_avgs, y2_avgs, S_1s, S_2s, S_3s, S_4s, S_Ss = \
        simulation(triangle, NUM_PARTICLES, ts, filename=FILENAME)
    plot_results(ts, x_avgs, y_avgs, x2_avgs, y2_avgs,
                 S_1s, S_2s, S_3s, S_4s, S_Ss)


if __name__ == "__main__":
    main()
#
