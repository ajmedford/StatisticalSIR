import pylab as plt
from SIR import GammaSIR, SymmetricBetaSIR
import numpy as np

fig, ax = plt.subplots()

params = [0.5, 1, 2]
cmap = plt.get_cmap('viridis')
colors = cmap(np.linspace(0, 1, len(params)))

herd_immunity = []

for color, param in zip(colors, params):

    model = SymmetricBetaSIR(ab=param, epbar0=1, n_bins=2000)

    tbin, xSbin, xIbin, xRbin, xImaxbin = model.get_result(mode='binnedSIR')
    ax.plot(tbin, xRbin, label='binnedSIR({}:{} = {:.2f})'.format(*model.get_param_info()) + '-{} bins'.format(model.n_bins), ls='-', color=color, alpha=0.5)

    tepSIR, xSepSIR, xIepSIR, xRepSIR, xImaxepSIR = model.get_result(mode='epsilonSIR')
    ax.plot(tepSIR[::2], xRepSIR[::2], #plot every other point to make plot clearer
            label='epsilonSIR({}:{} = {:.2f})'.format(*model.get_param_info()), alpha=0.5, color=color, ls='none', marker='o', mfc='none', markersize=4)


ax.legend()
ax.set_xlabel('Time [arbitrary]')
ax.set_ylabel('Proportion Infected')

distname = model.get_param_info()[0]
fig.savefig('plot-compare_methods-'+distname+'.png')

plt.show()
