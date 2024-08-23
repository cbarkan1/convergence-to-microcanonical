import numpy as np
import matplotlib.pyplot as plt

# simulation_data.npz is created with billiard_simulation.py
file = np.load('simulation_data.npz')
ts = file['ts']  # time points
x_avg = file['x_avgs']
y_avg = file['y_avgs']
x2_avg = file['x2_avgs']
y2_avg = file['y2_avgs']
S_S = file['S_Ss']  # Entropy of standard coarse-graining
S_G = file['S_4s']  # Entropy of generalized coarse-graining


# Panel A
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 5))
a.set_xscale('log')
a.set_yscale('log')
a.spines[['right', 'bottom']].set_visible(False)
a.set_xlim(0.01, 100)
a.set_ylim(1e-3, 5)
a.plot(ts, -S_G)
a.plot(ts, -S_S)
a.xaxis.tick_top()
a.invert_yaxis()

# Panel C
S_1 = file['S_1s']
S_2 = file['S_2s']
S_3 = file['S_3s']
S_4 = file['S_4s']

b.plot(ts, -S_1)
b.plot(ts, -S_2)
b.plot(ts, -S_3)
b.plot(ts, -S_4)
b.plot([0, 10], [0, 0], ':k')
b.set_xscale('log')
b.set_yscale('log')
b.set_xlim(0.01, 100)
b.set_ylim(1e-3, 5)
b.spines[['right', 'bottom']].set_visible(False)
b.xaxis.tick_top()
b.invert_yaxis()

plt.subplots_adjust(hspace=0.3)
plt.savefig('FigS2.svg', transparent=True)
plt.show()
