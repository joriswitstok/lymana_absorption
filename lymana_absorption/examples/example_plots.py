# Import Lyman-alpha absorption profile fitting class
from lymana_optical_depth import *





if __name__ == "__main__":
    import sys
    print("Python", sys.version)

    import numpy as np

    import matplotlib
    if __name__ == "__main__":
        print("Matplotlib", matplotlib.__version__, "(backend: " + matplotlib.get_backend() + ')')
    from matplotlib import pyplot as plt
    import seaborn as sns
    sns.set_style("ticks")

    pformat = ".pdf"
    
    cosmo = FlatLambdaCDM(H0=67.4, Om0=0.315, Ob0=0.02237/0.674**2, Tcmb0=2.726)

    # Example calculation for DLA absorption
    fig, ax = plt.subplots()
    
    ax_v = ax.secondary_xaxis("top", functions=(lambda wl: (1.0 - wl_Lya/wl)*299792.458, lambda v: wl_Lya / (1.0 - v/299792.458)))
    ax.tick_params(axis="both", which="both", top=False)
    
    wl_emit_array = np.linspace(1100, 1450, 2000)
    
    z_source = 9.0
    wl_obs_array = wl_emit_array * (1.0 + z_source)
    for x_HI_global in [0.1, 0.5, 1.0]:
        ax.plot(wl_emit_array, np.exp(-tau_IGM(wl_obs_array, z_source, R_ion=0.0, x_HI_global=x_HI_global, cosmo=cosmo)), alpha=0.8,
                label=r"$z = {:.3g}$, $\bar{{x}}_\mathrm{{HI}} = {:.1f}$".format(z_source, x_HI_global))

    T = 100.0
    ls = '--'
    for N_HI in [1e21, 1e22]:
        ax.plot(wl_emit_array, np.exp(-tau_DLA(wl_emit_array, N_HI, T)), linestyle=ls, alpha=0.8,
                label=r"$N_\mathrm{{HI}} = 10^{{{:.0f}}} \, \mathrm{{cm^{{-2}}}}$".format(np.log10(N_HI)))

    ax.set_xlabel(r"$\lambda_\mathrm{emit} \, (\mathrm{\AA})$")
    ax_v.set_xlabel(r"$v \, (\mathrm{km \, s^{{-1}}})$")
    ax.set_ylabel(r"$T_\mathrm{Ly\alpha}$")
    
    ax.set_xlim(wl_Lya / (1.0 + 8000/299792.458), wl_Lya / (1.0 - 40000/299792.458))
    ax.set_ylim(-0.075, 1.075)
    
    ax.legend()

    fig.savefig("IGM_DLA_absorption" + pformat, dpi=150, bbox_inches="tight")

    # plt.show()
    plt.close(fig)

    # Example calculation for IGM transmission curves, inspired by Mason & Gronke (2020)
    z_source = 7.0
    wl_emit_array = np.linspace(1200, 1230, 500)
    wl_obs_array = wl_emit_array * (1.0 + z_source)
    
    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(8.27, 8.27/2))

    ax = axes[0]
    ax_v = ax.secondary_xaxis("top", functions=(lambda wl: (1.0 - wl_Lya/wl)*299792.458, lambda v: wl_Lya / (1.0 - v/299792.458)))
    ax.tick_params(axis="both", which="both", top=False)
    
    x_HI = 1e-8
    for R_ion in np.arange(0.0, 5.25, 0.5):
        ax.plot(wl_emit_array, np.exp(-tau_IGM(wl_obs_array, z_source, R_ion=R_ion, x_HI=x_HI, cosmo=cosmo)), linestyle='--', color='k')
        ax.plot(wl_emit_array, np.exp(-tau_IGM(wl_obs_array, z_source, R_ion=R_ion, x_HI=x_HI, x_HI_profile="quadratic", cosmo=cosmo)),
                label=r"$R_\mathrm{{ion}} = {:.1f} \, \mathrm{{pMpc}}$".format(R_ion), zorder=5)
    
    ax.set_xlim(wl_Lya / (1.0 + 1100/299792.458), wl_Lya / (1.0 - 1100/299792.458))
    ax.set_ylim(-0.05, 1.175)

    ax.set_xlabel(r"$\lambda_\mathrm{emit} \, (\mathrm{\AA})$")
    ax_v.set_xlabel(r"$v \, (\mathrm{km \, s^{{-1}}})$")
    ax.set_ylabel(r"$T_\mathrm{Ly\alpha}$")

    leg = ax.legend(ncol=3, loc="upper center", fontsize="xx-small")
    leg.set_title(r"$z = {:.1f}$, $x_\mathrm{{HI}} = 10^{{-8}}$".format(z_source), prop={"size": "xx-small"})
    
    ax = axes[1]
    ax_v = ax.secondary_xaxis("top", functions=(lambda wl: (1.0 - wl_Lya/wl)*299792.458, lambda v: wl_Lya / (1.0 - v/299792.458)))
    ax.tick_params(axis="both", which="both", top=False)
    
    R_ion = 1.0 # Mpc
    for x_HI in 10**np.arange(-8.0, 0.0):
        # ax.plot(wl_emit_array, np.exp(-tau_IGM(wl_obs_array, z_source, R_ion=R_ion, x_HI=x_HI, cosmo=cosmo)), linestyle='--', color='k')
        ax.plot(wl_emit_array, np.exp(-tau_IGM(wl_obs_array, z_source, R_ion=R_ion, x_HI=x_HI, x_HI_profile="quadratic", cosmo=cosmo)),
                label=r"$x_\mathrm{{HI}} = 10^{{{:.0f}}}$".format(np.log10(x_HI)), zorder=5)
    
    ax.set_xlabel(r"$\lambda_\mathrm{emit} \, (\mathrm{\AA})$")
    ax_v.set_xlabel(r"$v \, (\mathrm{km \, s^{{-1}}})$")

    leg = ax.legend(ncol=2, loc="upper center", fontsize="xx-small")
    leg.set_title(r"$z = {:.1f}$, $R_\mathrm{{ion}} = 1 \, \mathrm{{pMpc}}$".format(z_source), prop={"size": "xx-small"})

    fig.savefig("Lya_bubble_transmission" + pformat, dpi=150, bbox_inches="tight")

    # plt.show()
    plt.close(fig)

    z_source = 11
    l_min, l_max = 1000, 2000

    fig, ax = plt.subplots()
    
    ax.set_title("IGM damping wing for $z_\mathrm{{source}} = {:.5g}$".format(z_source))
    ax.axvline(x=wl_Lya, linestyle='--', color='k', alpha=0.8)

    ax_r = ax.secondary_xaxis("top", functions=(lambda wl_emit: wl_emit*(1.0+z_source)/1e4, lambda wl_obs: wl_obs/(1.0+z_source)*1e4))
    ax.set_axisbelow(False)
    ax.tick_params(axis='x', which="both", top=False, labeltop=False)

    # Observed wavelengths
    wl_emit_array = np.linspace(l_min, l_max, 1000)
    wl_obs_array = wl_emit_array * (1.0+z_source)

    x_HIs = np.linspace(0.0, 1.0, 10)
    x_colors = sns.color_palette("mako_r", len(x_HIs))
    
    for x_HI, x_col in zip(x_HIs, x_colors):
        ax.plot(wl_emit_array, np.exp(-tau_IGM(wl_obs_array, z_source, R_ion=0.0, x_HI_global=x_HI, cosmo=cosmo)),
                color=x_col, alpha=0.8, label=r"$x_\mathrm{{HI}} = {:.1f}$".format(x_HI))
    
    ax.set_xlim(l_min, l_max)
    ax.set_ylim(0.85, 1.05)

    ax.set_xlabel(r"$\lambda_\mathrm{emit} \, (\mathrm{\AA})$")
    ax_r.set_xlabel(r"$\lambda_\mathrm{obs} \, (\mathrm{\mu m})$")
    ax.set_ylabel(r"$T_\mathrm{IGM}$")
    
    ax.legend()

    fig.savefig("IGM_damping_wing" + pformat, dpi=150, bbox_inches="tight")

    # plt.show()
    plt.close(fig)