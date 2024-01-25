#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script for modelling IGM and DLA absorption curves.

Joris Witstok, 2023
"""

import numpy as np

# Import astropy cosmology
from astropy.cosmology import FLRW, FlatLambdaCDM, z_at_value
from astropy import units

# Fundamental constants
c = 299792458.0 # speed of light in m/s
k_B = 1.380649e-23 # Boltzmann constant in J/K
m_p = 1.672621e-27 # proton mass in kg
m_e = 9.109383e-31 # electon mass in kg
m_H = 1.6737236e-27 # hydrogen mass in kg
e_charge = 1.602176e-19 # electon charge in C

# Lyman-alpha (2p -> 1s transition) properties
A_Lya = 6.265e8 # Einstein A coefficient
f_Lya = 0.4162 # oscillator strength
wl_Lya = 1215.67 # wavelength Angstrom
nu_Lya = 1e10 * c / wl_Lya # frequency in Hz

K_Lya = 1e-7 * f_Lya * np.sqrt(np.pi) * e_charge**2 * c / m_e

def deltanu_D(T, b_turb):
    """
    
    Thermally broadened frequency (in Hz), given a temperature `T` in K (see Dijkstra 2014)
    and optional turbulent velocity `b` in km/s
    
    
    """
    return nu_Lya * np.sqrt(2.0 * k_B * T / m_p + (b_turb*1e3)**2) / c

def Doppler_x(wl_emit, T, b_turb=0.0):
    """
    
    Dimensionless Doppler parameter `x` given a temperature `T` in K (see Dijkstra 2014)
    and optional turbulent velocity `b` in km/s
    
    
    """

    # Use the wavelength definition (see Lee 2013 for the distinction)
    nu = 1e10 * c / wl_emit # frequency in Hz
    x = (nu - nu_Lya) / deltanu_D(T, b_turb=b_turb)
    
    return x

def Voigt(x, T, b_turb, approximation="Tasitsiomi2006"):
    """
    
    Approximation of Voigt function of `x` (from Tasitsiomi 2006 or Tepper-Garcia 2006),
    given a temperature `T` in K and optional turbulent velocity `b` in km/s
    
    
    """

    # Voigt parameter
    a_V = A_Lya / (4.0 * np.pi * deltanu_D(T, b_turb=b_turb))
    x_squared = x**2

    if approximation == "Tasitsiomi2006":
        z = (x_squared - 0.855) / (x_squared + 3.42)
        q = np.where(z > 0.0, z * (1.0 + 21.0/x_squared) * a_V / (np.pi * (x_squared + 1.0)) * (0.1117 + z * (4.421 + z * (5.674 * z - 9.207))), 0.0)
        
        phi = np.sqrt(np.pi) * (q + np.exp(-x_squared)/1.77245385)
    elif approximation == "Tepper-Garcia2006":
        H0 = np.exp(-x_squared)
        Q = 1.5 * x**-2
        
        phi = np.where(x_squared > 1e-6, H0 - a_V / np.sqrt(np.pi) / x_squared * (H0 * H0 * (4 * x_squared * x_squared + 7 * x_squared + 4.0 + Q) - Q - 1.0), H0)
    else:
        raise ValueError("Voigt approximation identifier not recognised")
    
    return phi

def correction_Lee2013(nu):
    """
    
    Correction to scattering cross section based on full quantum-mechanical treatment (from Lee 2013), given frequencies `nu` in Hz
    
    
    """

    # Correct cross-section profile (see equation (18) in Dijkstra 2014)
    return (1.0 - 1.792 * (nu - nu_Lya) / nu_Lya)

def sigma_alpha(x, T, b_turb=0.0, approximation="Tasitsiomi2006", quantum_correction=True):
    """
    
    Lyman-alpha scattering cross section (e.g. Dijkstra 2014) given a temperature `T` in K and optional turbulent velocity `b` in km/s;
    note that electron charge and speed of light work slightly differently in SI and cgs units
    
    
    """
    
    sigma_alpha = K_Lya / deltanu_D(T, b_turb=b_turb) * Voigt(x, T, b_turb=b_turb, approximation=approximation)
    
    if quantum_correction:
        # Work out the frequency given the Doppler parameter x and deltanu_D
        sigma_alpha *= correction_Lee2013(nu=nu_Lya + x * deltanu_D(T, b_turb=b_turb))
    
    return sigma_alpha

def tau_DLA(wl_emit_array, N_HI, T, b_turb=0.0, approximation="Tasitsiomi2006"):
    """
    
    Lyman-alpha optical depth given a neutral hydrogen column density `N_HI` in cm^-2,
    temperature `T` in K, and optional turbulent velocity `b` in km/s
    
    
    """
    
    # Dimensionless Doppler parameter
    x = Doppler_x(wl_emit=wl_emit_array, T=T, b_turb=b_turb)
    
    # Convert column density from cm^-2 to m^-2 as the cross section is in m^2
    tau = N_HI * 1e4 * sigma_alpha(x, T, b_turb=b_turb, approximation=approximation)
    
    return tau

def tau_integral(wl_obs, z, x_HI, n_H, T, cosmo, approximation="Tasitsiomi2006"):
    """
    
    Lyman-alpha optical depth integral at observed wavelength `wl_obs` given a redshift range `z`,
    neutral hydrogen fraction `x_HI`, hydrogen density `n_H`, temperature `T`, and cosmology `cosmo`
    
    
    """

    assert hasattr(z, "__len__")
    if hasattr(x_HI, "__len__"):
        assert len(z) == len(x_HI)
    if hasattr(n_H, "__len__"):
        assert len(z) == len(n_H)
    assert not hasattr(T, "__len__")
    
    # Rest-frame wavelength (in Angstrom) and frequency (in Hz) at the various redshifts
    wl_emit = wl_obs / (1.0 + z)
    
    # Dimensionless Doppler parameter (note it is assumed the IGM does not have a turbulent component)
    x = Doppler_x(wl_emit=wl_emit, T=T, b_turb=0.0)

    dldz = c / (cosmo.H(z).to("Hz").value * (1.0 + z))
    
    return np.trapz(dldz * x_HI * n_H * sigma_alpha(x, T, b_turb=0.0, approximation=approximation), x=z)

def tau_IGM(wl_obs_array, z_s, R_ion=1.0, Delta=1.0, x_HI=1e-8, x_HI_profile="constant", T=1e4, x_HI_global=1.0,
            cosmo=None, z_reion=5.0, f_H=0.76, H0=70.0, Om0=0.3, Ob0=0.05, approximation="Tasitsiomi2006"):
    """
    
    Calculates the optical depth of a Lyman-alpha photon travelling through an
    ionised bubble and the neutral IGM (e.g. Dijkstra 2014, Mason & Gronke 2020)
    
    Parameters
    ----------
    wl_obs_array : array_like
        Observed wavelength array in Angstrom.
    z_s : float
        Redshift of the source.
    R_ion : array_like, optional
        Size of the ionised bubble in Mpc.
    Delta : float, optional
        Overdensity inside the ionised bubble.
    x_HI : float or array_like, optional
        Residual neutral fraction (at 0.1 pMpc if the profile is not constant).
    x_HI_profile : {"constant" or "quadratic"}, optional
        Type of profile for the residual neutral hydrogen fraction.
    T : float
        Temperature (in K) in the ionised bubble.
    cosmo : instance of astropy.cosmology.FLRW, optional
        Custom cosmology (see `astropy` documentation for details).
    z_reion : float, optional
        Reionisation redshift.
    f_H : float, optional
        Baryonic hydrogen fraction.
    H0 : float, optional
        Hubble constant.
    Om0 : float, optional
        Matter density parameter at z = 0.
    Ob0 : float, optional
        Baryonic matter density parameter at z = 0.

    Returns
    ----------
    tau : array_like
        Optical depth

    """

    if z_s < z_reion:
        return np.zeros_like(wl_obs_array)
    
    if cosmo is None:
        # Initiate astropy cosmology object, given H0 and Omega_matter
        cosmo = FlatLambdaCDM(H0=H0, Om0=Om0, Ob0=Ob0)
    else:
        assert isinstance(cosmo, FLRW)
    
    if hasattr(wl_obs_array, "__len__"):
        tau = np.zeros_like(wl_obs_array)
    else:
        tau = 0.0
    
    # Determine the redshift at which the ionised bubble ends
    small_bubble = cosmo.comoving_distance(z_s).to("Mpc").value - R_ion*(1.0+z_s) > cosmo.comoving_distance(z_reion).to("Mpc").value
    if small_bubble:
        # Local ionised bubble does not extend to the redshift where the Universe is entirely ionised (redshift tolerance: Δz ~ Δv/c ~ ΔR H(z)/c)
        z_ion = z_at_value(cosmo.comoving_distance, cosmo.comoving_distance(z_s) - R_ion*(1.0+z_s) * units.Mpc,
                            zmin=z_reion, zmax=100, ztol=R_ion*cosmo.H(z_s).to("km/(s Mpc)").value/c/1e6, method="bounded").value
    else:
        # Ionised bubble exceeds the specified redshift at which (i.e. overlaps with the point where) reionisation ends
        z_ion = z_reion
    
    # Mean hydrogen number density (in m^-3) at z = 0
    n_H_bar_0 = f_H * cosmo.critical_density(0).to("kg/m^3").value * cosmo.Ob(0) / m_H

    # First part of the integral, through the ionised bubble (assumed)
    z = np.linspace(z_ion, z_s, 1000)
    l = (cosmo.comoving_distance(z_s) - cosmo.comoving_distance(z)).to("Mpc").value / (1.0+z_s)
    assert np.abs(np.abs(l[0] - l[-1]) - R_ion) < 1e-4
    if x_HI_profile == "quadratic":
        x_HI = x_HI * (l / 0.1)**2
    n_H = Delta * n_H_bar_0 * (1.0 + z)**3
    
    if hasattr(wl_obs_array, "__len__"):
        tau += np.array([tau_integral(wl_obs=wl_obs, z=z, x_HI=x_HI, n_H=n_H, T=T, cosmo=cosmo, approximation=approximation) for wl_obs in wl_obs_array])
    else:
        tau += tau_integral(wl_obs=wl_obs_array, z=z, x_HI=x_HI, n_H=n_H, T=T, cosmo=cosmo, approximation=approximation)

    if small_bubble:
        # Second part of the integral, through the neutral IGM (assume x_HI = 1, Δ = 1, T = 1 K)
        z = np.linspace(z_reion, z_ion, 10000)
        
        if hasattr(wl_obs_array, "__len__"):
            tau += np.array([tau_integral(wl_obs=wl_obs, z=z, x_HI=x_HI_global, n_H=n_H_bar_0*(1.0 + z)**3, T=1, cosmo=cosmo, approximation=approximation) for wl_obs in wl_obs_array])
        else:
            tau += tau_integral(wl_obs=wl_obs_array, z=z, x_HI=x_HI_global, n_H=n_H_bar_0*(1.0 + z)**3, T=1, cosmo=cosmo, approximation=approximation)
    
    # Return optical depth
    return tau





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
    plt.style.use(sys_root + "Code/General/mpl/Styles/Nature.mplstyle")
    fontsize = plt.rcParams["font.size"]
    def_linewidth = plt.rcParams["lines.linewidth"]
    def_markersize = plt.rcParams["lines.markersize"]
    from General.mpl.Styles.format import pformat
    
    from General.mpl.Tools import vcoord_funcs
    from General.Libraries.Emission_lines import line
    
    Lya = line("HI", 1215)

    pformat = ".pdf"

    # Find current path for plot folders to follow the folder structure inside code
    pltfol = sys_root + "Plots/" + currentdir.split("Code/", 1)[1] + '/'
    if not os.path.exists(pltfol):
        os.makedirs(pltfol)
    
    cosmo = FlatLambdaCDM(H0=67.4, Om0=0.315, Ob0=0.02237/0.674**2, Tcmb0=2.726)

    cross_section_comparison = False
    DLA_absorption_comparison = False
    IGM_DLA_comparison = False
    IGM_absorption_example = True
    
    if cross_section_comparison:
        wl_emit_array = np.linspace(1200, 1300, 2000)
        
        create_transmission_curves = False
        if create_transmission_curves:
            transmission_list = []
            z_sources = np.arange(4, 15, 0.5)
            for z in z_sources:
                wl_obs_array = wl_emit_array * (1.0 + z)
                transmission_list.append(np.exp(-tau_IGM(wl_obs_array, z, R_ion=0.0, x_HI=1e-8, cosmo=cosmo)))
            np.savetxt(sys_root + "Results/" + currentdir.split("Code/", 1)[1] + "/Lya_transmission_curves.txt",
                        np.insert(transmission_list, 0, wl_emit_array, axis=0).transpose(), header="Wavelength\tz = " + "\tz = ".join(z_sources.astype(str)))
        
        # Comparisons of cross-section calculations
        fig, axes = plt.subplots(nrows=2, sharex=True, sharey=False, gridspec_kw=dict(hspace=0, height_ratios=[2, 1]))
        ax = axes[0]

        x = np.concatenate([-np.logspace(-6, 6, 2000)[::-1], np.linspace(-1e-6, 1e-6, 100), np.logspace(-6, 6, 2000)], axis=0)
        T = 1e4
        ax.plot(x, K_Lya / deltanu_D(T, b_turb=0.0) * np.exp(-x**2), linestyle=':', color='k', alpha=0.8, label="Gaussian core")

        colors = sns.color_palette()

        sigmas_dict = {}
        for qc, ls in zip([False, True], ['--', '-']):
            ax.plot(np.nan, np.nan, linestyle=ls, color='k', alpha=0.8, label="QM correction (Lee 2013)" if qc else "Pure Voigt profile")
            sigmas = []
            for approximation, color in zip(["Tepper-Garcia2006", "Tasitsiomi2006"], colors):
                sigma = sigma_alpha(x, T, approximation=approximation, quantum_correction=qc)
                sigmas.append(sigma)
                ax.plot(x, sigma, linestyle=ls, color=color, alpha=0.8, label=approximation if qc else None)
            
            sigmas_dict[qc] = sigmas
        
        axes[1].plot(x, (sigmas_dict[False][1]-sigmas_dict[False][0])/sigmas_dict[False][0], linestyle='--', color='k', alpha=0.8, label="Analytical Voigt profile")
        axes[1].plot(x, (sigmas_dict[True][0]-sigmas_dict[False][0])/sigmas_dict[False][0], linestyle='-', color='k', alpha=0.8, label="QM correction")

        axes[1].set_xlabel(r"$x_\nu$")
        ax.set_ylabel(r"$\sigma \, (\mathrm{m^2})$")
        axes[1].set_ylabel(r"$\Delta \sigma / \sigma$")
        
        ax.set_xscale("symlog", linthresh=0.15)
        ax.set_yscale("log")
        # axes[1].set_yscale("log")
        
        ax.set_xlim(8e3, -8e3)
        ax.set_ylim(2e-29, 2e-17)
        axes[1].set_ylim(-1.05, 1.05)
        
        ax.legend()
        axes[1].legend()

        fig.savefig(pltfol + "Cross_section_comparison" + pformat, dpi=150, bbox_inches="tight")

        # plt.show()
        plt.close(fig)
    
    if DLA_absorption_comparison:
        # Comparisons of DLA absorption calculations
        fig, ax = plt.subplots()
        
        vfunc = vcoord_funcs(0, 0, wl_Lya)
        ax_v = ax.secondary_xaxis("top", functions=(vfunc.l2v, vfunc.v2l))
        ax.tick_params(axis="both", which="both", top=False)
        
        wl_emit_array = np.linspace(1050, 1450, 2000)

        N_HI = 1e23
        for T, ls in zip([1.0, 100.0, 1e4], [':', '-', '--']):
            for approximation in ["Tepper-Garcia2006", "Tasitsiomi2006"]:
                ax.plot(wl_emit_array, np.exp(-tau_DLA(wl_emit_array, N_HI, T, approximation=approximation)), linestyle=ls, alpha=0.8,
                        label=r"{}, $N_\mathrm{{HI}} = 10^{{{:.0f}}} \, \mathrm{{cm^{{-2}}}}$, $T = 10^{{{:.0f}}} \, \mathrm{{K}}$".format(approximation, np.log10(N_HI), np.log10(T)))

        ax.set_xlabel(r"$\lambda_\mathrm{emit} \, (\mathrm{\AA})$")
        ax_v.set_xlabel(r"$v \, (\mathrm{km \, s^{{-1}}})$")
        ax.set_ylabel(r"$T_\mathrm{{{}}}$".format(Lya.smlabel))
        
        ax.set_xlim(vfunc.v2l(-40000), vfunc.v2l(40000))
        ax.set_ylim(-0.075, 1.075)
        
        ax.legend()

        fig.savefig(pltfol + "DLA_absorption_comparison" + pformat, dpi=150, bbox_inches="tight")

        plt.show()
        plt.close(fig)
    
    if IGM_DLA_comparison:
        # Example calculation for DLA absorption
        fig, ax = plt.subplots()
        
        vfunc = vcoord_funcs(0, 0, wl_Lya)
        ax_v = ax.secondary_xaxis("top", functions=(vfunc.l2v, vfunc.v2l))
        ax.tick_params(axis="both", which="both", top=False)
        
        wl_emit_array = np.linspace(1100, 1450, 2000)
        
        z_source = 9.0
        wl_obs_array = wl_emit_array * (1.0 + z_source)
        for x_HI_global in [0.1, 0.5, 1.0]:
            ax.plot(wl_emit_array, np.exp(-tau_IGM(wl_obs_array, z_source, R_ion=0.0, x_HI_global=x_HI_global, cosmo=cosmo)), alpha=0.8,
                    label=r"$z = {:.3g}$, $\bar{{x}}_\mathrm{{HI}} = {:.1f}$".format(z_source, x_HI_global))

        # for T, ls in zip([1.0, 100.0, 1e4], [':', '-', '--']):
        T = 100.0
        ls = '--'
        for N_HI in [1e21, 1e22]:
            ax.plot(wl_emit_array, np.exp(-tau_DLA(wl_emit_array, N_HI, T)), linestyle=ls, alpha=0.8,
                    label=r"$N_\mathrm{{HI}} = 10^{{{:.0f}}} \, \mathrm{{cm^{{-2}}}}$".format(np.log10(N_HI)))

        ax.set_xlabel(r"$\lambda_\mathrm{emit} \, (\mathrm{\AA})$")
        ax_v.set_xlabel(r"$v \, (\mathrm{km \, s^{{-1}}})$")
        ax.set_ylabel(r"$T_\mathrm{{{}}}$".format(Lya.smlabel))
        
        ax.set_xlim(vfunc.v2l(-8000), vfunc.v2l(40000))
        ax.set_ylim(-0.075, 1.075)
        
        ax.legend()

        fig.savefig(pltfol + "IGM_DLA_absorption" + pformat, dpi=150, bbox_inches="tight")

        # plt.show()
        plt.close(fig)

    if IGM_absorption_example:
        # Example calculation for IGM damping-wing absorption
        z_source = 13.0
        wl_emit_array = np.linspace(1200, 1230, 500)
        wl_obs_array = wl_emit_array * (1.0 + z_source)
        
        fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(8.27, 8.27/2))

        ax = axes[0]
        vfunc = vcoord_funcs(0, 0, wl_Lya)
        ax_v = ax.secondary_xaxis("top", functions=(vfunc.l2v, vfunc.v2l))
        ax.tick_params(axis="both", which="both", top=False)
        
        x_HI = 1e-8
        for R_ion in np.arange(0.0, 5.25, 0.5):
            ax.plot(wl_emit_array, np.exp(-tau_IGM(wl_obs_array, z_source, R_ion=R_ion, x_HI=x_HI, cosmo=cosmo)), linestyle='--', color='k')
            ax.plot(wl_emit_array, np.exp(-tau_IGM(wl_obs_array, z_source, R_ion=R_ion, x_HI=x_HI, x_HI_profile="quadratic", cosmo=cosmo)),
                    label=r"$R_\mathrm{{ion}} = {:.1f} \, \mathrm{{pMpc}}$".format(R_ion), zorder=5)
        
        ax.set_xlim(vfunc.v2l(-1100), vfunc.v2l(1100))
        ax.set_ylim(-0.05, 1.175)

        ax.set_xlabel(r"$\lambda_\mathrm{emit} \, (\mathrm{\AA})$")
        ax_v.set_xlabel(r"$v \, (\mathrm{km \, s^{{-1}}})$")
        ax.set_ylabel(r"$T_\mathrm{{{}}}$".format(Lya.smlabel))

        ax.legend(ncol=2, loc="upper center", title=r"$z = {:.1f}$, $x_\mathrm{{HI}} = 10^{{-8}}$".format(z_source))
        
        ax = axes[1]
        vfunc = vcoord_funcs(0, 0, wl_Lya)
        ax_v = ax.secondary_xaxis("top", functions=(vfunc.l2v, vfunc.v2l))
        ax.tick_params(axis="both", which="both", top=False)
        
        R_ion = 1.0 # Mpc
        for x_HI in 10**np.arange(-8.0, 0.0):
            # ax.plot(wl_emit_array, np.exp(-tau_IGM(wl_obs_array, z_source, R_ion=R_ion, x_HI=x_HI, cosmo=cosmo)), linestyle='--', color='k')
            ax.plot(wl_emit_array, np.exp(-tau_IGM(wl_obs_array, z_source, R_ion=R_ion, x_HI=x_HI, x_HI_profile="quadratic", cosmo=cosmo)),
                    label=r"$x_\mathrm{{HI}} = 10^{{{:.0f}}}$".format(np.log10(x_HI)), zorder=5)
        
        ax.set_xlabel(r"$\lambda_\mathrm{emit} \, (\mathrm{\AA})$")
        ax_v.set_xlabel(r"$v \, (\mathrm{km \, s^{{-1}}})$")

        ax.legend(ncol=2, loc="upper center", title=r"$z = {:.1f}$, $R_\mathrm{{ion}} = 1 \, \mathrm{{pMpc}}$".format(z_source))

        fig.savefig(pltfol + "Lya_bubble_transmission" + pformat, dpi=150, bbox_inches="tight")

        # plt.show()
        plt.close(fig)