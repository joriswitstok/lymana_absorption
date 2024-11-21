import numpy as np
from scipy.stats import gaussian_kde

def get_mode_hdi(sample, prob=0.05):
    """
    Calculate highest density interval (HDI) of array for given probability.
    The HDI is the minimum width Bayesian credible interval (BCI).

    Sources: https://python.arviz.org/en/stable/api/generated/arviz.hdi.html,
    https://github.com/aloctavodia/BAP/blob/master/first_edition/code/Chp1/hpd.py

    Parameters
    ----------
    
    sample : Numpy array or python list
        An array containing MCMC samples
    prob : float
        Desired probability

    Returns
    ----------
    hpd: 
          
    """
    
    sample = np.asarray(sample)
    sample = sample[~np.isnan(sample)]
    
    # Get lower and upper bounds
    l = np.min(sample)
    u = np.max(sample)
    x = np.linspace(l, u, 2000)
    dx = (x[-1]-x[0])/x.size
    
    # Kernel density estimate of the distribution
    density = gaussian_kde(sample)(x)
    density /= np.sum(density)
    # Find values that lie within the HDI
    indices = np.argsort(-density)
    sorted_interval_values = np.sort(x[indices][density[indices].cumsum() <= prob])

    modes = []
    hdi_intervals = []
    for interval in np.split(sorted_interval_values, np.where(np.diff(sorted_interval_values) >= dx * 1.1)[0]+1):
        int_select = (x > interval[0]) * (x < interval[-1])
        modes.append(x[int_select][np.argmax(density[int_select])])
        hdi_intervals.append([interval[0], interval[-1]])
    
    return (modes, hdi_intervals)