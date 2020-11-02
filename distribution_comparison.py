from SIR import *

fig, ax = plt.subplots()

fig2, ax2 = plt.subplots()

models = [GammaSIR, NumericalGenGammaSIR, NumericalLomaxSIR, NumericalParetoSIR, TwoPointSIR]

variance = 4.0
R0 = 2.5

for modelClass in models:

    epmax = 200
    model = modelClass(R0=R0,
            n_bins=int(epmax*1000), #number of bins. Excessively large number ensures continuous representation of distribution
            epmax=epmax, #maximum susceptibility. Mean is 1 for all models by default, so max is 100x the average.
            t_max=10*R0, #time limit to simulate to
            offset=0.05, #fixed parameter for OffsetGamma model (the offset from 0)
            min_peak=0.1, #fixed parameter for TwoPoint model (the position of the minimum peak)
            c = 0.5, #fixed parameter for GeneralizedGamma model
            log_histogram=False)
    param = model.variance_to_parameter(variance)

    print('#'*80) #delimiter to make output easier to read

    print(model.get_param_info())

    model.get_histogram(ax2,  alpha=0.3, label=str(model.get_param_info()))

    tbin, xSbin, xIbin, xRbin, xImaxbin = model.get_result(mode='binnedSIR')

    #ax.plot(tbin, xRbin, label='binnedSIR({}:{} = {:.2f})'.format(*model.get_param_info()) + '-{} bins'.format(model.n_bins), ls='-',  alpha=0.5)
    ax.plot(tbin, xRbin, label='binnedSIR({})'.format(model.get_param_info()) + '-{} bins'.format(model.n_bins), ls='-',  alpha=0.5)

orders = [2, 3, 5]

for order in orders:
    model = ParetoSIR(R0=R0, order=order, t_max=10*R0)
    tSIR, xSSIR, xISIR, xRSIR, xImaxSIR = model.get_result(mode='SIR')
    ax.plot(tSIR, xRSIR, label='SIR(order = {:.2f})'.format(order) , ls='--', alpha=0.5)

ax.set_title('SIR results: R0 = {}'.format(R0))
ax.set_xlabel('Time [arbitrary]')
ax.set_ylabel('Proportion Infected')
ax.legend()

ax2.set_title('Histograms of Distributions')
ax2.legend()
ax2.set_xlabel('Susceptibility [unitless]')
ax2.set_ylabel('Probability')
ax2.set_ylim([0, 3])
ax2.set_xlim([0, 10])

fig.savefig('plot-compare_distribution_results.png')
fig2.savefig('plot-compare_distributions_pdfs.png')

plt.show()
