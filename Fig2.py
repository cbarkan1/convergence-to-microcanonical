"""
Create Figure 2 panels A, B, and C.

"""

import numpy as np
import matplotlib.pyplot as plt

# simulation_data.npz is created with billiard_simulation.py
file = np.load('simulation_data.npz')
ts = file['ts']
x_avg = file['x_avgs']
y_avg = file['y_avgs']
x2_avg = file['x2_avgs']
y2_avg = file['y2_avgs']
S_S = file['S_Ss']
S_G = file['S_4s']


# Panel A
fig, (a, b) = plt.subplots(2, 1, figsize=(5.5, 5))
for ax in [a, b]:
    ax.set_xscale('log')
    ax.spines[['right', 'top']].set_visible(False)
    ax.plot([0, 40], [0, 0], ':k')
    ax.set_xlim(0.01, 100)

a.plot(ts, S_S, 'k')
a.set_yticks([-1, -0.5, 0])
a.set_yticklabels([-1, '', 0])
a.set_ylim(-1.5, .1)

b.plot(ts, S_G, 'k')
b.set_yticks([-4, -3, -2, -1, 0])
b.set_yticklabels([-4, '', -2, '', 0])
b.set_ylim(-4, 0.1)

plt.subplots_adjust(hspace=0.3)


# Panel B
plt.figure(figsize=(6, 5), linewidth=1.5)
plt.plot(ts, x_avg, dashes=[2, 1], linewidth=1.5)
plt.plot(ts, y_avg, linewidth=1.5)
plt.plot(ts, x2_avg, '--', linewidth=1.5)
plt.plot(ts, y2_avg, linewidth=1.5)

plt.xscale('log')
plt.ylim(-.15, 0.5)
plt.xlim(0.01, 100)
ax = plt.gca()
ax.set_yticks([-.1, 0, .1, .2, .3, .4, .5])
ax.set_yticklabels(['', 0, '', 0.2, '', 0.4, ''])
ax.spines[['right', 'top']].set_visible(False)

plt.legend([r'$\langle q_1\rangle$', r'$\langle q_2\rangle$',
           r'$\langle q_1^{\;2}\rangle$', r'$\langle q_2^{\;2}\rangle$'],
           prop={'size': 13})


# Panel C
S_1 = file['S_1s']
S_2 = file['S_2s']
S_3 = file['S_3s']
S_4 = file['S_4s']

plt.figure(figsize=(6.5, 3.5))
plt.plot(ts, S_1)
plt.plot(ts, S_2)
plt.plot(ts, S_3)
plt.plot(ts, S_4)
plt.plot([0, 10], [0, 0], ':k')
plt.xscale('log')
plt.xlim(0.01, 100)
plt.ylim(-4, 0.1)
plt.gca().spines[['right', 'top']].set_visible(False)
plt.legend(['N=1', 'N=2', 'N=3', 'N=4'])
plt.yticks([-4, -3, -2, -1, 0])
plt.gca().set_yticklabels([-4, '', '', '', 0])
plt.show()
