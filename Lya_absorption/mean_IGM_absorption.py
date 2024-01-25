#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script for modelling mean IGM transmission curves (based on Inoue et al. 2014). Adapted from https://github.com/brantr/igm-absorption

Joris Witstok, 2023
"""

import numpy as np

# Lyman series and Lyman limit (all in Angstroms)
l_LyS  = [1215.6701, 1025.72, 972.537, 949.743, 937.803, 930.748, 926.226, 923.150, 920.963, 919.352, 918.129, 917.181, 916.429, 915.824, 915.329, 914.919, 914.576, 914.286, 914.039, 913.826, 913.641, 913.480, 913.339, 913.215, 913.104, 913.006, 912.918, 912.839, 912.768, 912.703, 912.645, 912.592, 912.543, 912.499, 912.458, 912.420, 912.385, 912.353, 912.324]
l_LL = 911.8

def tau_LAF_LS(lambda_obs, z):
    # Table 2 of Inoue et al. (2014)
    A_j1s = [1.690e-2, 4.692e-3, 2.239e-3, 1.319e-3, 8.707e-4, 6.178e-4, 4.609e-4, 3.569e-4, 2.843e-4, 2.318e-4, 1.923e-4, 1.622e-4, 1.385e-4, 1.196e-4, 1.043e-4, 9.174e-5, 8.128e-5, 7.251e-5, 6.505e-5, 5.868e-5, 5.319e-5, 4.843e-5, 4.427e-5, 4.063e-5, 3.738e-5, 3.454e-5, 3.199e-5, 2.971e-5, 2.766e-5, 2.582e-5, 2.415e-5, 2.263e-5, 2.126e-5, 2.000e-5, 1.885e-5, 1.779e-5, 1.682e-5, 1.593e-5, 1.510e-5]
    A_j2s = [2.354e-3, 6.536e-4, 3.119e-4, 1.837e-4, 1.213e-4, 8.606e-5, 6.421e-5, 4.971e-5, 3.960e-5, 3.229e-5, 2.679e-5, 2.259e-5, 1.929e-5, 1.666e-5, 1.453e-5, 1.278e-5, 1.132e-5, 1.010e-5, 9.062e-6, 8.174e-6, 7.409e-6, 6.746e-6, 6.167e-6, 5.660e-6, 5.207e-6, 4.811e-6, 4.456e-6, 4.139e-6, 3.853e-6, 3.596e-6, 3.364e-6, 3.153e-6, 2.961e-6, 2.785e-6, 2.625e-6, 2.479e-6, 2.343e-6, 2.219e-6, 2.103e-6]
    A_j3s = [1.026e-4, 2.849e-5, 1.360e-5, 8.010e-6, 5.287e-6, 3.752e-6, 2.799e-6, 2.167e-6, 1.726e-6, 1.407e-6, 1.168e-6, 9.847e-7, 8.410e-7, 7.263e-7, 6.334e-7, 5.571e-7, 4.936e-7, 4.403e-7, 3.950e-7, 3.563e-7, 3.230e-7, 2.941e-7, 2.689e-7, 2.467e-7, 2.270e-7, 2.097e-7, 1.943e-7, 1.804e-7, 1.680e-7, 1.568e-7, 1.466e-7, 1.375e-7, 1.291e-7, 1.214e-7, 1.145e-7, 1.080e-7, 1.022e-7, 9.673e-8, 9.169e-8]

    tau = np.zeros_like(lambda_obs)
    
    assert len(l_LyS) == len(A_j1s) and len(l_LyS) == len(A_j2s) and len(l_LyS) == len(A_j3s)
    for l_j, A_j1, A_j2, A_j3 in zip(l_LyS, A_j1s, A_j2s, A_j3s):
        # Equation (21) of Inoue et al. (2014)
        attenuated = (lambda_obs > l_j) * (lambda_obs < l_j * (1.0 + z))
        tau += np.where(attenuated * (lambda_obs < 2.2*l_j), A_j1 * (lambda_obs/l_j)**1.2, 0)
        tau += np.where(attenuated * (lambda_obs >= 2.2*l_j) * (lambda_obs < 5.7*l_j), A_j2 * (lambda_obs/l_j)**3.7, 0)
        tau += np.where(attenuated * (lambda_obs >= 5.7*l_j), A_j3 * (lambda_obs/l_j)**5.5, 0)

    # Return combined tau
    return tau

def tau_DLA_LS(lambda_obs, z):
    # Table 2 of Inoue et al. (2014)
    A_j1s = [1.617e-4, 1.545e-4, 1.498e-4, 1.460e-4, 1.429e-4, 1.402e-4, 1.377e-4, 1.355e-4, 1.335e-4, 1.316e-4, 1.298e-4, 1.281e-4, 1.265e-4, 1.250e-4, 1.236e-4, 1.222e-4, 1.209e-4, 1.197e-4, 1.185e-4, 1.173e-4, 1.162e-4, 1.151e-4, 1.140e-4, 1.130e-4, 1.120e-4, 1.110e-4, 1.101e-4, 1.091e-4, 1.082e-4, 1.073e-4, 1.065e-4, 1.056e-4, 1.048e-4, 1.040e-4, 1.032e-4, 1.024e-4, 1.017e-4, 1.009e-4, 1.002e-4]
    A_j2s = [5.390e-5, 5.151e-5, 4.992e-5, 4.868e-5, 4.763e-5, 4.672e-5, 4.590e-5, 4.516e-5, 4.448e-5, 4.385e-5, 4.326e-5, 4.271e-5, 4.218e-5, 4.168e-5, 4.120e-5, 4.075e-5, 4.031e-5, 3.989e-5, 3.949e-5, 3.910e-5, 3.872e-5, 3.836e-5, 3.800e-5, 3.766e-5, 3.732e-5, 3.700e-5, 3.668e-5, 3.637e-5, 3.607e-5, 3.578e-5, 3.549e-5, 3.521e-5, 3.493e-5, 3.466e-5, 3.440e-5, 3.414e-5, 3.389e-5, 3.364e-5, 3.339e-5]

    tau = np.zeros_like(lambda_obs)
    
    assert len(l_LyS) == len(A_j1s) and len(l_LyS) == len(A_j2s)
    for l_j, A_j1, A_j2 in zip(l_LyS, A_j1s, A_j2s):
        # Equation (22) of Inoue et al. (2014)
        attenuated = (lambda_obs > l_j) * (lambda_obs < l_j * (1.0 + z))
        tau += np.where(attenuated * (lambda_obs < 3.0*l_j), A_j1 * (lambda_obs/l_j)**2.0, 0)
        tau += np.where(attenuated * (lambda_obs >= 3.0*l_j), A_j2 * (lambda_obs/l_j)**3.0, 0)

    # Return combined tau
    return tau

def tau_DLA_LC(lambda_obs, z):
    tau = np.zeros_like(lambda_obs)

    # Equations (28) and (29) of Inoue et al. (2014)
    attenuated = lambda_obs < l_LL * (1.0+z)
    if z < 2.0:
        tau += np.where(attenuated, 0.211 * (1.0+z)**2 - 7.66e-2 * (1.0+z)**2.3 * (lambda_obs/l_LL)**-0.3 - 0.135*(lambda_obs/l_LL)**2, 0)
    else:
        tau += np.where(attenuated, 4.70e-2 * (1.0+z)**3 - 1.78e-2 * (1.0+z)**3.3 * (lambda_obs/l_LL)**-0.3, 0)
        tau += np.where(attenuated * (lambda_obs < 3*l_LL), 0.634 - 0.135 * (lambda_obs/l_LL)**2 - 0.291 * (lambda_obs/l_LL)**-0.3, 0)
        tau -= np.where(attenuated * (lambda_obs >= 3*l_LL), 2.92e-2 * (lambda_obs/l_LL)**3, 0)
    
    return tau

def tau_LAF_LC(lambda_obs, z):
    tau = np.zeros_like(lambda_obs)

    # Equations (25), (26) and (29) of Inoue et al. (2014)
    attenuated = lambda_obs < l_LL * (1.0+z)
    if z < 1.2:
        tau += np.where(attenuated, 0.325 * ((lambda_obs/l_LL)**1.2 - (1.0+z)**-0.9 * (lambda_obs/l_LL)**2.1), 0)
    elif z < 4.7:
        tau += np.where(attenuated, 2.55e-2 * (1.0+z)**1.6 * (lambda_obs/l_LL)**2.1, 0)
        tau += np.where(attenuated * (lambda_obs < 2.2*l_LL), 0.325 * (lambda_obs/l_LL)**1.2 - 0.250 * (lambda_obs/l_LL)**2.1, 0)
        tau -= np.where(attenuated * (lambda_obs >= 2.2*l_LL), 2.55e-2 * (lambda_obs/l_LL)**3.7, 0)
    else:
        tau += np.where(attenuated, 5.22e-4 * (1.0+z)**3.4 * (lambda_obs/l_LL)**2.1, 0)
        tau += np.where(attenuated * (lambda_obs < 2.2*l_LL), 0.325 * (lambda_obs/l_LL)**1.2 - 3.14e-2*(lambda_obs/l_LL)**2.1, 0)
        tau += np.where(attenuated * (lambda_obs >= 2.2*l_LL) * (lambda_obs < 5.7*l_LL), 0.218 * (lambda_obs/l_LL)**2.1 - 2.55e-2 * (lambda_obs/l_LL)**3.7, 0)
        tau -= np.where(attenuated * (lambda_obs >= 5.7*l_LL), 5.22e-4 * (lambda_obs/l_LL)**5.5, 0)
    
    return tau

def tau_IGM(lambda_obs, z):
    # Equation (15) of Inoue et al. (2014); only valid above the observed Lyman limit wavelength
    above_limit = lambda_obs > l_LL
    l_above_limit = lambda_obs[above_limit]

    tau = np.zeros_like(lambda_obs)
    tau[above_limit] += tau_LAF_LS(l_above_limit, z) + tau_DLA_LS(l_above_limit, z) + tau_LAF_LC(l_above_limit, z) + tau_DLA_LC(l_above_limit, z)
    
    return tau

def igm_absorption(lambda_obs, z):
    """
    Function adapted from https://github.com/brantr/igm-absorption
    Provide an array of observed wavelengths in Angstrom
    and return an array of the attenuation at a redshift z owing
    to neutral H in the IGM along the line of sight.
    See Inoue et al. (2014)
    """
    attenuation = np.exp(-tau_IGM(lambda_obs, z))

    return attenuation



if __name__ == "__main__":
    import os, sys, inspect
    print("Python", sys.version)

    # Find current path
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

    from General.Tools.run_location import determine_rl
    run_loc, sys_root, obs_root = determine_rl()

    import numpy as np
    # import math

    import matplotlib
    if run_loc == "MacBook":
        # matplotlib.use("TkAgg") # use interactive Agg backend
        pass
    elif run_loc in ["serj", "serk", "prospero"]:
        matplotlib.use("Agg") # use non-interactive Agg backend
    if __name__ == "__main__":
        print("Matplotlib", matplotlib.__version__, "(backend: " + matplotlib.get_backend() + ')')
    from matplotlib import pyplot as plt
    import seaborn as sns
    sns.set_style("ticks")

    # Load style file
    plt.style.use(sys_root + "Code/General/mpl/Styles/PaperDoubleFig.mplstyle")
    fontsize = plt.rcParams["font.size"]
    def_linewidth = plt.rcParams["lines.linewidth"]
    def_markersize = plt.rcParams["lines.markersize"]
    from General.mpl.Styles.format import pformat
    from General.mpl.Axes.transform import axtr
    
    from General.Libraries.Emission_lines import line

    pformat = ".pdf"

    # Find current path for plot folders to follow the folder structure inside code
    pltfol = sys_root + "Plots/" + currentdir.split("Code/", 1)[1] + '/'
    if not os.path.exists(pltfol):
        os.makedirs(pltfol)

    fig, ax = plt.subplots()
    ax.set_facecolor('k')
    ax.patch.set_alpha(0.1)

    ax_z = ax.secondary_xaxis("top", functions=(lambda l: l*1e4/l_LyS[0]-1.0, lambda z: (1.0 + z)*l_LyS[0]/1e4))
    ax.set_axisbelow(False)
    ax.tick_params(axis='x', which="both", top=False, labeltop=False)

    l_min, l_max = l_LL/1e4, 1.4
    lambda_obs = np.linspace(l_min, l_max, int(1e4))

    redshifts = np.arange(0, 9)
    z_colors = sns.color_palette("RdYlBu_r", len(redshifts))

    for z, z_col in zip(redshifts, z_colors):
        ax.axvline(x=l_LL/1e4 * (1.0 + z), linestyle='--', color=z_col, alpha=0.8, zorder=-10-z)
        if z == redshifts[-1]:
            ax.annotate(text="Lyman limit", xy=(l_LL/1e4 * (1.0 + z), 0.5), xytext=(-4, 0),
                        xycoords=axtr(ax, "data", "axes"), textcoords="offset points",
                        va="center", ha="right", rotation="vertical", size="xx-small", color=z_col, alpha=0.8)
        ax.plot(lambda_obs, igm_absorption(lambda_obs*1e4, z), linewidth=2, color=z_col, alpha=0.8, label=r"$z = {:.0f}$".format(z), zorder=-z)
    
    ax.set_xlim(0.15, l_max)
    # ax.set_ylim(-0.05, 1.05)

    ax.set_xlabel(r"$\lambda_\mathrm{obs} \, (\mathrm{\mu m})$")
    ax_z.set_xlabel(r"$z_\mathrm{{{}}}$".format(line("HI", 1216).smlabel))
    ax.set_ylabel(r"$T_\mathrm{IGM}$")
    
    ax.legend(loc="center right", framealpha=0)

    fig.savefig(pltfol + "IGM_absorption" + pformat, dpi=150, bbox_inches="tight")

    # plt.show()
    plt.close(fig)