"""

Generates Figure A1

"""


import numpy as np
import matplotlib.pyplot as plt


def f(x,t):
	"""
	f_t(x) defined in Appendix A.


	"""
	return np.sin(x*(t+1)*np.pi)**2


# x values for plots
xs = np.linspace(0,1,1000)

fig, (a,b,c,d) = plt.subplots(1,4,figsize=(9,2.))

# Panel A
t = 0
fs = f(xs,t)
a.plot(xs,fs,'k',linewidth=1)

# Panel B
t = 5
fs = f(xs,t)
b.plot(xs,fs,'k',linewidth=1)

# Panel C
t = 20
fs = f(xs,t)
c.plot(xs,fs,'k',linewidth=1)

# Panel D
# plots the horizontal line y=0.5
d.plot([xs[0],xs[-1]],[.5,.5],'k',linewidth=1.5)

# Plot adjustments
for ax in [a,b,c,d]:
	ax.set_xlim(0,1)
	ax.set_ylim(0,1.1)
	ax.set_xticks([0,0.25,0.5,.75,1])
	ax.set_xticklabels([0,'','','',1])
	ax.set_yticks([0,0.25,0.5,.75,1])
	ax.set_yticklabels([0,'','','',1])
	ax.spines[['right', 'top']].set_visible(False)

plt.subplots_adjust(wspace=.4,bottom=0.2)

plt.show()