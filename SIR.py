import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import gamma
from scipy.stats import gamma as gamma_distro
from scipy.stats import genpareto, pareto, gengamma, rv_histogram, lomax, beta
import pylab as plt

class SIRComparison:
    """Base class for comparing S-I-R models that account for susceptibility variation using different representations of the susceptibility distribution. The following model types are supported:
        
        - SIR: a S-I-R model with variable order on the susceptibility term. dS/dt = -beta*I*S^order, where order is defined by the `get_order` method.
        - epsilonSIR: an S-I-R model with susceptibility weighted by current average susceptibility. dS/dt = -beta*epsilon_bar(t)*I*S, where `epsilon_bar(t)` is defined by the `get_epbar` method.
        - binnedSIR: an S-I-R model with a numerically-defined susceptibility distribution described by a histogram. dS_i/dt = -beta*I*S_i*epsilon_i, where epsilon_i is the susceptibility of bin i, defined by `get_bin_positions`.

        The total initial proportion of susceptibles is defined by the xS0 attribute (0.9999 by default). For the binnedSIR, the initial population of each bin is defined by `get_initial_populations`.
        
        The following analyses are supported:

        - get_result: Solve the set of differential equations corresponding to the model. The method returns (time, S, I, R, max(I)). 
        - get_histogram: Plot the histogram of the initial susceptibility distribution. Only supported if the `distribution` method is defined.

        The following methods must be defined in an classes that inheret from SIR comparison to modify the susceptibility distribution used:

        - get_order: Must be defined for SIR. This sets the static order of the S term based on the properties of the susceptibility distribution.
        - get_epbar: Must be defined for epsilonSIR, otherwise static initial epsilon_bar (`epbar0`) will be used. Time should be passed as a single argument, t, and t=0 can be used for static epsilon_bar.
        - distribution: Must be defined for binnedSIR. A method taking the bin positions as an argument and returning the corresponding probability density function. The `pdf` method of `scipy.stats` distributions can be used.
        - get_param_info: Must be defined for binnedSIR. A method that returns the parameters of the susceptibility distribution. 

        The following methods and attributes have defined defaults and do not need to be overridden, but can be modified as needed:

        Attributes to modify:
        - xS0: initial proportion of susceptibles
        - R0: The R0 value for the disease
        - beta: The rate constant for transimission (dS/dt = -beta*I*S)
        - t_max: Maximum time when solving ODE's
        - rtol: relative tolerance used by ODE solver
        - atol: absolute tolerance used by ODE solver
        - epbar0: Initial average susceptibility
        - epmax: Maximum possible susceptibility
        - n_bins: The number of bins used to represent a distribution in the binned model
        - log_histogram: Plot/generate histograms with probability density in log space (True/False). Useful for heavy-tailed distributions.
        - histogram_style: Plot histogram as a continuous line (`line`) or as a bar plot (`bar`)
        - verbose: Print messages (True/False). Messages will start with the name of the method that generates them.

        Any keyword arguments passed to the initialization will automatically be assigned as attributes.

        Methods to override:
        - get_bin_positions: Positions to sample the probability distribution. Evenly spaced bins from 0-epmax is the default.

        
        """

    def __init__(self, **params):
        self.xS0 = 0.9999
        self.xR = (1. - self.xS0)
        self.R0 = 2
        self.beta = 2
        self.gamma = self.R0/self.beta
        self.t_max = 10*self.R0
        self.rtol=1e-12
        self.atol=1e-12
        self.epbar0 = 1
        self.epmax = None
        self.n_bins = 300
        self.x_list = [] #list of all x vectors. Useful for debugging and post analysis
        self.t_list = [] #list of all time points. Useful for plotting distribution dynamics.
        self.log_histogram = False
        self.histogram_style='line'
        self.verbose=True
        for key, val in params.items():
            setattr(self, key, val)

    def get_order(self):
        return self.order

    def get_epbar(self, t):
        return self.epbar0

    def get_bin_positions(self):
        delta = self.epmax/(self.n_bins+1)
        bin_means = np.linspace(delta, self.epmax, self.n_bins, endpoint=False)
        return bin_means

    def get_initial_populations(self):
        positions = self.get_bin_positions()
        populations = self.distribution(positions)*self.epmax/self.n_bins
        return populations

    def get_histogram(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        x = self.get_bin_positions()
        width = self.epmax/self.n_bins
        heights = self.distribution(x)
        x_hist = np.array([0] + list(x))
        distro = rv_histogram((heights, x_hist))
        self.histogram = (heights, x_hist)
        m, v, s = distro.stats(moments='mvs')
        if self.verbose:
            print('get_histogram: Stats from numerical histogram for : ', self.get_param_info())
            print("\tMean: {} \n\tVariance: {} \n\tSkew: {} ".format(m,v,s))
        if self.log_histogram == True:
            heights = np.log10(heights)
            ylabel = '$log_{10}(p(\epsilon$))'
        else:
            ylabel = '$p(\epsilon$)'
        if self.histogram_style == 'bar':
            ax.bar(x=x, height=heights, width=width, **kwargs)
        else:
            ax.plot(x, heights, **kwargs)

        ax.set_xlabel('$\epsilon$')
        ax.set_ylabel(ylabel)

    def variance_to_parameter(self, variance, param_name=None, param_range=None):
        if param_name is None:
            param_name = self.param_name

        if param_range is None:
            param_range = self.param_range

        def get_variance(param_val):
            setattr(self, param_name, param_val)
            x = self.get_bin_positions()
            heights = self.distribution(x)
            x_hist = np.array([0] + list(x))
            distro = rv_histogram((heights, x_hist))
            return distro.stats(moments='v')
       
        verbose = self.verbose
        self.verbose = False #turn off verbosity inside of root finding

        resid = lambda p: get_variance(p) - variance

        try:
            param_val = brentq(resid, param_range[0], param_range[1])
            setattr(self, param_name, param_val)
        except ValueError:
            print("Brent's method failed. Variance when param={} is {}, and variance when param={} is {}".format(param_range[0], get_variance(param_range[0]), param_range[1], get_variance(param_range[1])))

        assert np.isclose(variance, get_variance(param_val)), 'Failed to find valid parameter for given variance (Range of variance is from {} - {})'.format(get_variance(param_range[0]), get_variance(param_range[1]))

        self.verbose=verbose

        if self.verbose:
            print("varance_to_parameter: Variance = {} when {} = {}.".format(variance,param_name, param_val))

        return param_val

    def SIR(self, t, x):
        xS, xI, xR = x
        self.t = t
        order = self.get_order()
        dxSdt = -self.beta*xI*(xS**order)
        dxIdt = self.beta*xI*(xS**order) - self.gamma*xI
        dxRdt = self.gamma*xI
        return [dxSdt, dxIdt, dxRdt]

    def epsilonSIR(self, t, x):
        xS, xI, xR  = x
        self. t = t
        self.xR = xR
        epbar = self.get_epbar(t)
        r = self.beta*epbar*xI*xS
        dxSdt = -r
        dxIdt = r - self.gamma*xI
        dxRdt = xI
        return [dxSdt, dxIdt, dxRdt]

    def binnedSIR(self, t, x):
        xS = x[:-2]
        xI = x[-2]
        xR = x[-1]

        self.t_list.append(t)
        self.x_list.append(x)

        dxSdt = -self.beta*xI*xS*self.bin_means
        dxIdt = -dxSdt.sum() - self.gamma*xI
        dxRdt = self.gamma*xI

        dxdt = np.empty_like(x)
        dxdt[:-2] = dxSdt
        dxdt[-2] = dxIdt
        dxdt[-1] = dxRdt
        return dxdt

    def get_result(self, mode):
        x0 = (self.xS0, 1.-self.xS0, 0)

        tspan = (0.5, self.t_max)
        rtol = self.rtol
        atol = self.atol
        kwargs = {'dense_output':True, 'rtol':rtol, 'atol':atol}

        if mode == 'SIR':
            sol = solve_ivp(self.SIR, t_span = tspan, y0=x0, **kwargs) 
            t = sol.t
            x = sol.y.T
            xS = x[:,0]
            xR = x[:,-1]
            xI = x[:-2]

        if mode == 'epsilonSIR':
            sol = solve_ivp(self.epsilonSIR, t_span = tspan, y0=x0, **kwargs) 
            t = sol.t
            x = sol.y.T
            xS = x[:,0]
            xR = x[:,-1]
            xI = x[:-2]

        if mode == 'powerSIR':
            sol = solve_ivp(self.powerSIR, t_span = tspan, y0=x0, **kwargs) 
            t = sol.t
            x = sol.y.T
            xS = x[:,0]
            xR = x[:,-1]
            xI = x[:-2]

        if mode == 'binnedSIR':
            self.bin_means = self.get_bin_positions()
            x0 = np.zeros(self.n_bins+2)
            x0[:-2] = self.get_initial_populations()
            x0[-2] = 1.-self.xS0
            x0[-1] = 0
            sol = solve_ivp(self.binnedSIR, t_span = tspan, y0=x0, **kwargs)
            t = sol.t
            x = sol.y.T
            xS = x[:,:-2].sum(axis=1)
            xR = x[:, -1]
            xI = x[:, -2]

        return t, xS, xI, xR, xI.max()

class UniformSIR(SIRComparison):
    """Defines S-I-R models with uniform-distributed susceptibility distribution. Required attributes:

       - epmax: defines the maximum value of susceptibility.

    """
    def __init__(self, *args, **kwargs):
        SIRComparison.__init__(self, *args, **kwargs)

    def get_epbar(self, t):
        self.xR = max(self.xR, (1. - self.xS0)*0.001)
        A = self.beta*self.xR*self.epmax
        numerator = (1 -  (1 + A)*np.exp(-A))
        denom = (1 - np.exp(-A))
        return (1./(self.xR*self.beta))*(numerator/denom)

    def get_order(self):
        return 1. + (1./3.)

    def distribution(self, epsilon):
        return (epsilon < self.epmax)/self.epmax

    def get_param_info(self):
        return 'Uniform', '$\epsilon_{max}$', self.epmax

class GammaSIR(SIRComparison):
    """Defines S-I-R models with gamma-distributed susceptibility distribution. Required attributes:

        - k: order of gamma distribution

        Special methods:
        - get_param_from_variance: Find the gamma distribution order, k, that corresponds to a given variance. This is trivial in theory, but the function serves as a sanity check and deals with slight numerical discrepancies from representing the distribution with a finite support.

        """

    def __init__(self, *args, **kwargs):
        SIRComparison.__init__(self, *args, **kwargs)
        if "epmax" not in kwargs:
            self.epmax = 15*self.epbar0 #limit of binned models

        self.param_name = 'k'
        self.param_range = [0.01, 10]

    def get_epbar(self, t):
        epbar = 1./((1./self.epbar0) + (self.beta*self.xR/self.k))
        return epbar

    def get_order(self):
        return 1 + (1./self.k)

    def distribution(self, epsilon):
        epbar = self.get_epbar(0)
        k = self.k
        m, v, s = gamma_distro.stats(k, scale = self.epbar0/k, moments='mvs')
        if self.verbose:
            print('distribution: Theoretical stats for : ', self.get_param_info())
            print("\tMean: {} \n\tVariance: {} \n\tSkew: {} ".format(m,v,s))
        return gamma_distro.pdf(epsilon, k, scale = epbar/k)

    def variance_to_parameter(self, variance, param_name=None, param_range=None):

        if param_range is None:
           valids = [0.05,10]
        else:
            valids = param_range

        resid = lambda k: gamma_distro.stats(k, scale = self.epbar0/k, moments='v') - variance

        k_var = brentq(resid, valids[0], valids[1])
        assert np.isclose(variance, gamma_distro.stats(k_var, scale=self.epbar0/k_var, moments='v')), 'Failed to find valid parameter for given variance'
        self.k = k_var

        if self.verbose:
            print("variance_to_parameter: Parameter estimated assuming infinite support for ", self.get_param_info())
            print("variance_to_parameter: The parameter for variance={} is : {}".format(variance,k_var))
        return k_var

    def get_param_info(self):
        return 'Gamma', '$k$', np.round(self.k,3)

class OffsetGammaSIR(SIRComparison):
    def __init__(self, *args, **kwargs):
        SIRComparison.__init__(self, *args, **kwargs)
        if "epmax" not in kwargs:
            self.epmax = 15*self.epbar0 #limit of binned models
        self.param_name = 'k'
        self.param_range = [0.1, 10]

    def get_rescaling(self, maxiter=50):
        tol = 1e-3
        epbar = self.get_epbar(0)
        k = self.k
        loc = self.offset
        rescale = gamma_distro.stats(k, loc=loc, moments='m')
        j = 0
        m = rescale
        while not np.isclose(m, epbar, rtol=tol, atol=tol) and j < maxiter:
            j += 1
            m = gamma_distro.stats(k, loc=loc, scale=1./rescale, moments='m')
            rescale *= m
        real_mean = gamma_distro.stats(k, loc=loc, scale=1./rescale, moments='m')
        if not np.isclose(real_mean, epbar, rtol=tol, atol=tol):
            raise ValueError('Mean did not converge. Target: {}, Actual: {}'.format(real_mean, epbar))
        self.rescale = rescale
        return rescale

    def distribution(self, epsilon):
        epbar = self.get_epbar(0)
        k = self.k
        loc = self.offset
        m = gamma_distro.stats(k, loc=loc, moments='m')
        rescale = self.get_rescaling()
        rescale = m
        mean, var, skew = gamma_distro.stats(k, scale = 1./rescale, loc=loc, moments='mvs')
        return gamma_distro.pdf(epsilon, k, scale = 1./rescale, loc=loc)

    def get_param_info(self):
        return 'OffsetGamma', '$k$', np.round(self.k,3), 'offset', self.offset

class GenGammaSIR(SIRComparison):
    def __init__(self, *args, **kwargs):
        SIRComparison.__init__(self, *args, **kwargs)
        if "epmax" not in kwargs:
            self.epmax = 15*self.epbar0 #limit of binned models
        
        self.param_name = 'a'
        self.param_range = [0.1, 10]

    def distribution(self, epsilon):
        a = self.a
        c = self.c
        m = gengamma.stats(a, c, moments='m')
        mean, var, skew = gengamma.stats(a, c, scale = self.epbar0/m, moments='mvs')

        if self.verbose:
            print('distribution: Theoretical stats for : ', self.get_param_info())
            print("\tMean: {} \n\tVariance: {} \n\tSkew: {} ".format(mean,var,skew))

        if not np.isclose(mean,  self.epbar0): 
            print("distribution: Mean has shifted from {} to {}. Possible invalid distribution parameters.".format(self.epbar0, mean))
            raise ValueError

        return gengamma.pdf(epsilon, a, c, scale = self.epbar0/m)

    def get_param_info(self):
        return 'GenGamma', '$a$', np.round(self.a, 3), '$c$', np.round(self.c, 3)

class NumericalGenGammaSIR(SIRComparison):
    def __init__(self, *args, **kwargs):
        SIRComparison.__init__(self, *args, **kwargs)
        if "epmax" not in kwargs:
            self.epmax = 15*self.epbar0 #limit of binned models
        
        self.param_name = 'a'
        self.param_range = [0.1, 10]

    def distribution(self, epsilon):
        x = self.get_bin_positions()
        c = self.c
        a = self.a
        m = gengamma.stats(a, c, moments='m')
        mean, var, skew = gengamma.stats(a, c, scale = self.epbar0/m, moments='mvs')

        if self.verbose:
            print('distribution: Theoretical stats for : ', self.get_param_info())
            print("\tMean: {} \n\tVariance: {} \n\tSkew: {} ".format(mean,var,skew))

        heights = gengamma.pdf(x, a, c, scale = self.epbar0/m)
        x_hist = np.array([0] + list(x))
        distro = rv_histogram((heights, x_hist))
        return distro.pdf(epsilon)

    def get_param_info(self):
        return 'NumericalGenGamma', '$a$', self.a, '$c$', self.c

class SymmetricBetaSIR(SIRComparison):
    def __init__(self, *args, **kwargs):
        SIRComparison.__init__(self, *args, **kwargs)
       
        self.epmax = 2.*self.epbar0 #symmetric beta is always defined on 0-2 to have a mean of 1.
        self.param_name = 'ab'
        self.param_range = [1e-9, 10]

    def distribution(self, epsilon):
        ab = self.ab
        mean, var, skew = beta.stats(ab, ab, scale = 2., moments='mvs')

        if self.verbose:
            print('distribution: Theoretical stats for : ', self.get_param_info())
            print("\tMean: {} \n\tVariance: {} \n\tSkew: {} ".format(mean,var,skew))

        if not np.isclose(mean,  self.epbar0): 
            print("distribution: Mean has shifted from {} to {}. Possible invalid distribution parameters.".format(self.epbar0, mean))
            raise ValueError

        return beta.pdf(epsilon, ab, ab, scale = 2.)

    def get_param_info(self):
        return 'SymmetricBeta', 'a & b', np.round(self.ab, 3)


class TwoPointSIR(SIRComparison):
    def __init__(self, *args, **kwargs):
        SIRComparison.__init__(self, *args, **kwargs)
        if "epmax" not in kwargs:
            self.epmax = 10*self.epbar0

        self.param_name = 'p'
        self.param_range = [0.05, 0.95]


    def distribution(self, epsilon):
        m = self.epbar0
        w = self.p
        mu1 = self.min_peak
        mu2  = (m - w*mu1)/(1-w)
        bin_means = self.get_bin_positions()
        s = self.epmax/self.n_bins
        assert self.epmax > mu2+s, "Maximum ({}) is less than or too close to highest mean ({}).".format(self.epmax, mu2)
        assert mu1 - s >= 0, "Parameters require negative susceptibility"

        pdf = (np.exp(-(epsilon-mu1)**2 / (2. * s**2)) / np.sqrt(2. * np.pi * s**2)) * w
        pdf += (np.exp(-(epsilon-mu2)**2 / (2. * s**2)) / np.sqrt(2. * np.pi * s**2)) * (1 - w)
        return pdf

    def get_param_info(self):
        return 'TwoPoint', 'p', np.round(self.p,3), 'mu_1', np.round(self.min_peak,3)

class GenParetoSIR(SIRComparison):
    def __init__(self, *args, **kwargs):
        SIRComparison.__init__(self, *args, **kwargs)
        if "epmax" not in kwargs:
            self.epmax = 15*self.epbar0 #limit of binned models
            
        self.param_name = 'c'
        self.param_range = [-0.9, 0.45]

    def distribution(self, epsilon):
        c = self.c
        m = genpareto.stats(c, moments='m')
        mean, var, skew = genpareto.stats(c, scale = self.epbar0/m, moments='mvs')
        if not np.isclose(mean,  self.epbar0): 
            print("distribution: Mean has shifted from {} to {}. Possible invalid distribution parameters.".format(self.epbar0, mean))

        if self.verbose:
            print('distribution: Theoretical stats for : ', self.get_param_info())
            print("\tMean: {} \n\tVariance: {} \n\tSkew: {} ".format(mean,var,skew))

        return genpareto.pdf(epsilon, c, scale = self.epbar0/m)

    def variance_to_parameter(self, variance, param_name=None, param_range=None):
        epbar = self.get_epbar(0)
        if not param_range:
            valids = [-0.9, 0.45]
        else:
            valids= param_range
        resid = lambda k: genpareto.stats(k, scale = self.epbar0/genpareto.stats(k, moments='m'), moments='v') - variance
        k_var = brentq(resid, valids[0], valids[1])
        self.c = k_var
        assert np.isclose(variance, genpareto.stats(k_var, scale=self.epbar0/genpareto.stats(k_var, moments='m'), moments='v')), 'Failed to find valid parameter for given variance'
        if self.verbose:
            print("variance_to_parameter: Parameter estimated assuming infinite support for ", self.get_param_info())
            print("variance_to_parameter: The parameter for variance={} is : {}".format(variance,k_var))
        return k_var

    def get_param_info(self):
        return 'GenPareto', 'c', np.round(self.c,3)

class NumericalGenParetoSIR(SIRComparison):
    def __init__(self, *args, **kwargs):
        SIRComparison.__init__(self, *args, **kwargs)
        if "epmax" not in kwargs:
            self.epmax = 15*self.epbar0 #limit of binned models

        self.param_name = 'c'
        self.param_range = [-0.9, 0.45] 

    def distribution(self, epsilon):
        x = self.get_bin_positions()
        c = self.c
        m = genpareto.stats(c, moments='m')
        mean = genpareto.stats(c, scale = self.epbar0/m, moments='m')
        if not np.isclose(mean,  self.epbar0): 
            print("Mean has shifted from {} to {}. Possible invalid distribution parameters.".format(self.epbar0, mean))
        mean, var, skew = genpareto.stats(c, scale = self.epbar0/m, moments='mvs')
        if self.verbose:
            print('distribution: Theoretical stats for : ', self.get_param_info())
            print("\tMean: {} \n\tVariance: {} \n\tSkew: {} ".format(mean,var,skew))
        heights = genpareto.pdf(x, c, scale = self.epbar0/m)
        x_hist = np.array([0] + list(x))
        distro = rv_histogram((heights, x_hist))
        return distro.pdf(epsilon)

    def get_param_info(self):
        return 'NumericalGenPareto', 'c', np.round(self.c, 3)

class ParetoSIR(SIRComparison):
    def __init__(self, *args, **kwargs):
        SIRComparison.__init__(self, *args, **kwargs)
        if "epmax" not in kwargs:
            self.epmax = 15*self.epbar0 #limit of binned models

        self.param_name = 'b'
        self.param_range = [1.1, 3.5]

    def distribution(self, epsilon):
        b = self.b
        m = pareto.stats(b, moments='m')
        mean = pareto.stats(b, scale = self.epbar0/m, moments='m')
        if not np.isclose(mean,  self.epbar0): 
            print("Mean has shifted from {} to {}. Possible invalid distribution parameters.".format(self.epbar0, mean))

        mean, var, skew = pareto.stats(b, scale = self.epbar0/m, moments='mvs')
        if self.verbose:
            print('distribution: Theoretical stats for : ', self.get_param_info())
            print("\tMean: {} \n\tVariance: {} \n\tSkew: {} ".format(mean,var,skew))

        return pareto.pdf(epsilon, b, scale = self.epbar0/m)

    def variance_to_parameter(self, variance, param_name=None, param_range=None):
        epbar = self.get_epbar(0)
        if param_range is None:
            valids = [1.1,3.5]
        else:
            valids = param_range
        resid = lambda k: pareto.stats(k, scale = self.epbar0/pareto.stats(k, moments='m'), moments='v') - variance
        k_var = brentq(resid, valids[0], valids[1])
        self.b = k_var
        assert np.isclose(variance, pareto.stats(k_var, scale=self.epbar0/pareto.stats(k_var, moments='m'), moments='v')), 'Failed to find valid parameter for given variance'
        if self.verbose:
            print("variance_to_parameter: Parameter estimated assuming infinite support for ", self.get_param_info())
            print("variance_to_parameter: The parameter for variance={} is : {}".format(variance,k_var))
        return k_var

    def get_param_info(self):
        return 'Pareto', 'b', np.round(self.b,3)

class NumericalParetoSIR(SIRComparison):
    def __init__(self, *args, **kwargs):
        SIRComparison.__init__(self, *args, **kwargs)
        if "epmax" not in kwargs:
            self.epmax = 15*self.epbar0 #limit of binned models

        self.param_name = 'b'
        self.param_range = [1.25, 3.5] 

    def distribution(self, epsilon):
        x = self.get_bin_positions()
        b = self.b
        m = pareto.stats(b, moments='m')
        mean = pareto.stats(b, scale = self.epbar0/m, moments='m')
        if not np.isclose(mean,  self.epbar0): 
            print("distribution: Mean has shifted from {} to {}. Possible invalid distribution parameters.".format(self.epbar0, mean))

        mean, var, skew = pareto.stats(b, scale = self.epbar0/m, moments='mvs')
         
        if self.verbose:
            print('distribution: Theoretical stats for : ', self.get_param_info())
            print("\tMean: {} \n\tVariance: {} \n\tSkew: {} ".format(mean,var,skew))

        heights = pareto.pdf(x, b, scale = self.epbar0/m)
        x_hist = np.array([0] + list(x))
        distro = rv_histogram((heights, x_hist))
        return distro.pdf(epsilon)

    def get_param_info(self):
        return 'NumericalPareto', 'b', np.round(self.b, 3)

class NumericalLomaxSIR(SIRComparison):
    def __init__(self, *args, **kwargs):
        SIRComparison.__init__(self, *args, **kwargs)
        if "epmax" not in kwargs:
            self.epmax = 15*self.epbar0 #limit of binned models

        self.param_name = 'c'
        self.param_range = [1.1, 100] 

    def distribution(self, epsilon):
        x = self.get_bin_positions()
        c = self.c
        m = lomax.stats(c, moments='m')
        mean = lomax.stats(c, scale = self.epbar0/m, moments='m')
        if not np.isclose(mean,  self.epbar0): 
            print("distribution: Mean has shifted from {} to {}. Possible invalid distribution parameters.".format(self.epbar0, mean))

        mean, var, skew = lomax.stats(c, scale = self.epbar0/m, moments='mvs')
         
        if self.verbose:
            print('distribution: Theoretical stats for : ', self.get_param_info())
            print("\tMean: {} \n\tVariance: {} \n\tSkew: {} ".format(mean,var,skew))

        heights = lomax.pdf(x, c, scale = self.epbar0/m)
        x_hist = np.array([0] + list(x))
        distro = rv_histogram((heights, x_hist))
        return distro.pdf(epsilon)

    def get_param_info(self):
        return 'NumericalLomax', 'c', np.round(self.c, 3)
