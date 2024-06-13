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
from .constants import c, k_B, m_p, m_H, A_Lya, nu_Lya, K_Lya

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

def tau_integral_vec(wl_obs, z, x_HI, n_H, T, cosmo, approximation="Tasitsiomi2006"):
    """
    
    Lyman-alpha optical depth integral at observed wavelength `wl_obs` given a redshift range `z`,
    neutral hydrogen fraction `x_HI`, hydrogen density `n_H`, temperature `T`, and cosmology `cosmo`
    This is a vectorised version of `tau_integral`, checked for a random sample of inputs with
    `np.allclose` to give the same results as `tau_integral`.
    This is faster than `tau_integral` when the number of steps is <5,000 (same speed otherwise).
    
    
    """

    z    = np.atleast_1d(z)
    x_HI = np.atleast_1d(x_HI)
    n_H  = np.atleast_1d(n_H)
    assert (len(x_HI)==1) or (len(z)==len(x_HI)), (
        f'Maldito, {x_HI=} must be scalar or same len as {z=}')
    assert (len(n_H)==1) or (len(z)==len(n_H)), (
        f'Maldito, {n_H=} must be scalar or same len as {z=}')
    assert not hasattr(T, "__len__"), f'You goaty goose, {T=} must be a scalar'
    
    # Rest-frame wavelength (in Angstrom) and frequency (in Hz) at the various redshifts
    wl_emit = wl_obs[:, None] / (1.0 + z[None, :])
    
    # Dimensionless Doppler parameter (note it is assumed the IGM does not have a turbulent component)
    x = Doppler_x(wl_emit=wl_emit, T=T, b_turb=0.0)

    dldz = c / (cosmo.H(z).to("Hz").value * (1.0 + z))
    
    return np.trapz(dldz[None, :] * x_HI[None, :] * n_H[None, :] * sigma_alpha(x, T, b_turb=0.0, approximation=approximation),
                    x=z, axis=1)

def tau_IGM(wl_obs_array, z_s, R_ion=1.0, Delta=1.0, x_HI=1e-8, x_HI_profile="constant", T=1e4, x_HI_global=1.0,
            cosmo=None, z_reion=5.3, f_H=0.76, H0=70.0, Om0=0.3, Ob0=0.05, approximation="Tasitsiomi2006",
            use_vector=False):
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
    use_vector : bool, optional
        If True, replace `tau_integral` with `tau_integral_vec`; faster
        under some circumstances.

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

    if use_vector:
        tau += tau_integral_vec(
            wl_obs=wl_obs_array, z=z, x_HI=x_HI, n_H=n_H, T=T, cosmo=cosmo,
            approximation=approximation)
        if small_bubble:
            # Second part of the integral, through the neutral IGM (assume x_HI = 1, Δ = 1, T = 1 K)
            z = np.linspace(z_reion, z_ion, 10000)
        
            tau += tau_integral_vec(
                wl_obs=wl_obs_array, z=z, x_HI=x_HI_global,
                n_H=n_H_bar_0*(1.0 + z)**3, T=1, cosmo=cosmo, approximation=approximation)

        # Return optical depth for vectorised calls.
        return tau

    if hasattr(wl_obs_array, "__len__"):
        tau += np.array([tau_integral(wl_obs=wl_obs, z=z, x_HI=x_HI, n_H=n_H, T=T, cosmo=cosmo, approximation=approximation) for wl_obs in wl_obs_array])
    else:
        tau += tau_integral(wl_obs=wl_obs_array, z=z, x_HI=x_HI, n_H=n_H, T=T, cosmo=cosmo, approximation=approximation)

    if small_bubble:
        # Second part of the integral, through the neutral IGM (assume Δ = 1, T = 1 K)
        z = np.linspace(z_reion, z_ion, 10000)
        
        if hasattr(wl_obs_array, "__len__"):
            tau += np.array([tau_integral(wl_obs=wl_obs, z=z, x_HI=x_HI_global, n_H=n_H_bar_0*(1.0 + z)**3, T=1, cosmo=cosmo, approximation=approximation) for wl_obs in wl_obs_array])
        else:
            tau += tau_integral(wl_obs=wl_obs_array, z=z, x_HI=x_HI_global, n_H=n_H_bar_0*(1.0 + z)**3, T=1, cosmo=cosmo, approximation=approximation)
    
    # Return optical depth
    return tau


