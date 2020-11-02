import pylab as plt
from SIR import *
import numpy as np

R0 = 2.5

cmap = plt.get_cmap('viridis')
n_plots = 10
colors = cmap(np.linspace(0, 1, n_plots))

herd_immunity = []

fig, axes = plt.subplots(1, 2, figsize=(9,6))

initial_distro = 'uniform'

if initial_distro == 'lomax':
    epmax=200
    model = NumericalLomaxSIR(c=2.5, 
                               R0 = R0,
                               epbar0=1, 
                               n_bins=epmax*1000, 
                               epmax=epmax)

if initial_distro == 'pareto':
    epmax=200
    model = NumericalParetoSIR(b=1.73, 
                               R0 = R0,
                               epbar0=1, 
                               n_bins=epmax*1000, 
                               epmax=epmax)

elif initial_distro == 'uniform':
    epmax=5
    model = UniformSIR(epmax=epmax,
            epbar0=1, 
            n_bins=epmax*1000, 
            R0=R0)

tbin, xSbin, xIbin, xRbin, xImaxbin = model.get_result(mode='binnedSIR')
axes[0].plot(tbin, xRbin, label='binnedSIR({}:{} = {:.2f})'.format(*model.get_param_info()) + '-{} bins'.format(model.n_bins), ls='-', color='k', alpha=0.5)

axes[0].set_xlabel('Time [arbitrary]')
axes[0].set_ylabel('Proportion Infected')

step_size = int(len(model.x_list)/n_plots)

times = model.t_list[::step_size]
distros = model.x_list[::step_size]
bin_means = model.bin_means

for t_i, hist_i, color in zip(times, distros, colors):
    # each entry of the x_list contains all bin info for xS ([:-2]), xI ([:-2]), and xR ([:-1]).
    axes[0].plot([t_i], [hist_i[-1]], marker='o', color=color)
    axes[1].plot(bin_means, hist_i[:-2]/(model.epmax/model.n_bins), color=color)

p_cutoff = 1e-5
eps_cutoff = epmax
for eps, p in zip(bin_means, distros[0]):
    if p < p_cutoff and eps>1:
        eps_cutoff = eps
        break

axes[1].set_xlim([0, eps_cutoff])
axes[1].set_xlabel('Susceptibility [unitless]')
axes[1].set_ylabel('Probability')
distname = model.get_param_info()[0]
fig.savefig('plot-distribution_dynamics-'+distname+'.png')

plt.show()
