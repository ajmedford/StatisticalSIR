import pylab as plt
from SIR import *
from scipy.stats import rv_histogram
import numpy as np
import matplotlib.gridspec as gridspec

R0 = 2.5

GammaModel = GammaSIR(k=1, 
                 R0 = R0,
                 epmax=25,
                 n_bins=25*1000)

BetaModel = SymmetricBetaSIR(ab=1, 
                 R0 = R0,
                 epmax=2.,
                 n_bins=50*1000)

fig = plt.figure(constrained_layout=True, figsize=(8, 6))
gs = fig.add_gridspec(3, 6)

ax_gamma_results = fig.add_subplot(gs[:2, :3])

ax_gamma_1 = fig.add_subplot(gs[2, 0])
ax_gamma_2 = fig.add_subplot(gs[2, 1])
ax_gamma_3 = fig.add_subplot(gs[2, 2])

ax_beta_results = fig.add_subplot(gs[:2, 3:])
ax_beta_1 = fig.add_subplot(gs[2, 3])
ax_beta_2 = fig.add_subplot(gs[2, 4])
ax_beta_3 = fig.add_subplot(gs[2, 5])

def dynamics_plot(model, param_name, param_list, solution_axes, dynamics_axes,  n_timeslices=12, markers=['o', 's', '^', '+', '*'], linestyles = ['-', '--', ':', '-.', '-']):

    for ax, d_ax, param, mark, ls in zip(solution_axes, dynamics_axes, param_list, markers, linestyles):

        setattr(model, param_name, param)
        model.x_list = []
        model.t_list = []

        spacing = 3

        tbin, xSbin, xIbin, xRbin, xImaxbin = model.get_result(mode='binnedSIR')

        if param_name == 'ab':
            pname = 'a = b'
        else:
            pname = param_name

        ax.plot(tbin, xRbin, label="{} = {}".format(pname, param), ls=ls, color='k', alpha=0.5)

        ax.set_xlabel('Time [arbitrary]')
        ax.set_ylabel('Proportion Infected')

        ax.legend()

        step_size = int(len(model.x_list)/n_timeslices)

        cmap = plt.get_cmap('viridis')
        colors = cmap(np.linspace(0, 1, n_timeslices))

        times = model.t_list[::step_size]
        distros = model.x_list[::step_size]
        bin_means = model.bin_means

        for t_i, hist_i, color in zip(times, distros, colors):
            # each entry of the x_list contains all bin info for xS ([:-2]), xI ([:-2]), and xR ([:-1]).
            if d_ax:
                ax.plot([t_i], [hist_i[-1]], marker='|', markersize=10, color=color)

                #d_ax.annotate('{} = {}'.format(pname, param), [0.5, 0.7], xycoords='axes fraction')

                d_eps = model.epmax/model.n_bins
                hist = hist_i[:-2]/d_eps
                d_ax.plot(bin_means, hist, color=color, alpha=0.5, ls='-', lw=1.5, label="t = {:.1f}".format(t_i))

                d_ax.set_xlim([0, min(model.epmax, 4)])
                d_ax.set_ylim([0, 1.1])


dynamics_plot(GammaModel, 'k', [0.5, 1, 2], solution_axes = [ax_gamma_results]*3, dynamics_axes = [ax_gamma_1, ax_gamma_2, ax_gamma_3])

dynamics_plot(BetaModel, 'ab', [0.5, 1, 2], solution_axes = [ax_beta_results]*3, dynamics_axes = [ax_beta_1, ax_beta_2, ax_beta_3])

for ax in [ax_gamma_results, ax_beta_results]:
    ax.set_ylim([0, 0.8])

#ax_gamma_2.set_xlabel('Susceptibility [unitless]')
#ax_gamma_1.set_ylabel('Probability')

#ax_beta_2.set_xlabel('Susceptibility [unitless]')
#ax_beta_1.set_ylabel('Probability')

fig.savefig('gamma_beta_comparison.pdf')

plt.show()
