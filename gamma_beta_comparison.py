import pylab as plt
from SIR import *
from scipy.stats import rv_histogram
import numpy as np

R0 = 2.5

GammaModel = GammaSIR(k=1, 
                 R0 = R0,
                 epmax=25,
                 n_bins=25*1000)

BetaModel = SymmetricBetaSIR(ab=1, 
                 R0 = R0,
                 epmax=2.,
                 n_bins=50*1000)

def dynamics_plot(model, param_name, param_list, solution_axes, dynamics_axes, order_axes, n_timeslices=12, markers=['o', 's', '^', '+', '*'], linestyles = ['-', '--', ':', '-.', '-']):

    for ax, d_ax, K_ax, param, mark, ls in zip(solution_axes, dynamics_axes, order_axes, param_list, markers, linestyles):

        setattr(model, param_name, param)
        model.x_list = []
        model.t_list = []

        spacing = 3

        tbin, xSbin, xIbin, xRbin, xImaxbin = model.get_result(mode='binnedSIR')

        ax.plot(tbin, xRbin, label='binnedSIR({}:{} = {:.2f})'.format(*model.get_param_info()) + '-{} bins'.format(model.n_bins), ls=ls, color='k', alpha=0.5)
        ax.plot(tbin[::spacing], xRbin[::spacing], label='binnedSIR({}:{} = {:.2f})'.format(*model.get_param_info()) + '-{} bins'.format(model.n_bins), ls='none', color='k', alpha=0.5, marker=mark, markersize=4)

        dSdt = np.gradient(xSbin, tbin)
        logdSdt = np.log(-dSdt)
        logxS = np.log(xSbin)

        K = np.gradient(logdSdt, logxS)
        mask = np.abs(dSdt) > 1e-3
        print(sum(mask))
        print(logdSdt.shape)

        K_ax.plot(tbin[mask], K[mask], label='binnedSIR({}:{} = {:.2f})'.format(*model.get_param_info()) + '-{} bins'.format(model.n_bins), ls=ls, color='k', alpha=0.5)

        ax.set_xlabel('Time [arbitrary]')
        ax.set_ylabel('Proportion Infected')

        step_size = int(len(model.x_list)/n_timeslices)

        cmap = plt.get_cmap('viridis')
        colors = cmap(np.linspace(0, 1, n_timeslices))

        times = model.t_list[::step_size]
        distros = model.x_list[::step_size]
        bin_means = model.bin_means

        for t_i, hist_i, color in zip(times, distros, colors):
            # each entry of the x_list contains all bin info for xS ([:-2]), xI ([:-2]), and xR ([:-1]).
            #axes[0].plot([t_i], [hist_i[-1]], marker='|', markersize=10, color=color)
            if d_ax:

                d_eps = model.epmax/model.n_bins
                hist = hist_i[:-2]/d_eps
                d_ax.plot(bin_means, hist, color=color, alpha=0.5, ls='-', lw=0.5, label="t = {:.1f}".format(t_i))

                d_ax.set_xlim([0, min(model.epmax, 4)])
                d_ax.set_ylim([0, 1.1])
                d_ax.set_xlabel('Susceptibility [unitless]')
                d_ax.set_ylabel('Probability')

fig1, axes_1 = plt.subplots(2, 2, figsize=(8,7))

fig2, axes_2 = plt.subplots(2, 3, figsize=(8,9))

dynamics_plot(GammaModel, 'k', [0.5, 1, 2], solution_axes = [axes_1[0,0]]*3, dynamics_axes = axes_2[0,:], order_axes=[axes_1[0,1]]*3)

dynamics_plot(BetaModel, 'ab', [0.5, 1, 2], solution_axes = [axes_1[1,0]]*3, dynamics_axes = axes_2[1,:], order_axes=[axes_1[1,1]]*3)

axes_2[0,0].legend()

plt.show()
