import pylab as plt
from SIR import GammaSIR
import numpy as np

fig, ax = plt.subplots()

params = [1, 2, 3, 4, 5]
cmap = plt.get_cmap('viridis')
colors = cmap(np.linspace(0, 1, len(params)))

herd_immunity = []

for color, param in zip(colors, params):

    model = GammaSIR(k=param, epbar0=1, n_bins=int(1e4), epmax=100)

    tbin, xSbin, xIbin, xRbin, xImaxbin = model.get_result(mode='binnedSIR')
    ax.plot(tbin, xRbin, label='binnedSIR({}:{} = {:.2f})'.format(*model.get_param_info()) + '-{} bins'.format(model.n_bins), ls='-', color=color, alpha=0.5)

    tepSIR, xSepSIR, xIepSIR, xRepSIR, xImaxepSIR = model.get_result(mode='epsilonSIR')
    ax.plot(tepSIR[::2], xRepSIR[::2], #plot every other point to make plot clearer
            label='epsilonSIR({}:{} = {:.2f})'.format(*model.get_param_info()), alpha=0.5, color=color, ls='none', marker='o', mfc='none', markersize=4)

    tSIR, xSSIR, xISIR, xRSIR, xImaxSIR = model.get_result(mode='SIR')
    ax.plot(tSIR[::3], xRSIR[::3], #plot every third point to make plot clearer
            label='SIR(n={:.2f})'.format(model.get_order()), color=color, ls='none',marker='^', mfc='none', alpha=0.5, markersize=4)

ax.legend()
ax.set_xlabel('Time [arbitrary]')
ax.set_ylabel('Proportion Infected')

distname = model.get_param_info()[0]
fig.savefig('plot-compare_methods-'+distname+'.png')

plt.show()
