#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script for recombination rates.

Joris Witstok, 2019
"""

import numpy as np

from .constants import E_Lya

def alpha_A_HII_Draine2011(T):
    """Case A H II recombination coefficient.
    
    Draine equation 14.5

    """

    return 4.13e-13*(T/1.0e4)**(-0.7131-0.0115*np.log(T/1.0e4)) # cm^3 s^-1



def alpha_B_HII_Draine2011(T):
    """Case B H II recombination coefficient.
    
    Draine equation 14.6

    """

    return 2.54e-13*(T/1.0e4)**(-0.8163-0.0208*np.log(T/1.0e4)) # cm^3 s^-1



def alpha_A_HII_HuiGnedin1997(T):
    """Case A H II recombination coefficient.
    
    Hui and Gnedin 1997 (MNRAS 292 27; Appendix A).

    """

    t_HI = 1.57807e5 # K; H ionization threshold 
    reclambda = 2.0*t_HI/T
    
    return (1.269e-13 * (reclambda**1.503) /
            (1.0 + (reclambda/0.522)**0.470)**1.923) # cm^3 s^-1



def alpha_B_HII_HuiGnedin1997(T):
    """Case B H II recombination coefficient.
    
    Hui and Gnedin 1997 (MNRAS 292 27; Appendix A).

    """

    t_HI = 1.57807e5 # K; H ionization threshold 
    reclambda = 2.0*t_HI/T
    
    return (2.753e-14 * (reclambda**1.5) /
            (1.0 + (reclambda/2.74)**0.407)**2.242) # cm^3 s^-1



def alpha_A_HII_FukugitaKawasaki1994(T):
    """Case A H II recombination coefficient.
    
    Fukugita & Kawasaki (1994).

    """

    return 6.28e-11 * T**(-0.5) * (T/1e3)**(-0.2) / (1.0 + (T/1e5)**(0.7)) # cm^3 s^-1



def f_recA(T):
    """Probability of emission of a Ly-a photon per recombination.
    
    Dijkstra (2014)
    
    This is the conversion of number of hydrogen-ionising photons to number of Lya photons in case A, f_rec = a_eff_Lya / alpha:
    L_Lya / Q(H) = E_Lya * f_rec = (E_Lya a_eff_Lya / alpha) = 4π j_Lya / (n_e * n_HII) / alpha (Osterbrock & Ferland 2006)

    """

    T4 = T/1.0e4
    return 0.41 - 0.165 * np.log10(T4) - 0.015 * (T4)**(-0.44)

def f_recB(T):
    """Probability of emission of a Ly-a photon per recombination.
    
    Dijkstra (2014), taken from Cantalupo et al. (2008)
    
    This is the conversion of number of hydrogen-ionising photons to number of Lya photons in case B, f_rec = a_eff_Lya / alpha:
    L_Lya / Q(H) = E_Lya * f_rec = (E_Lya a_eff_Lya / alpha) = 4π j_Lya / (n_e * n_HII) / alpha (Osterbrock & Ferland 2006)

    """
    
    T4 = T/1.0e4
    return 0.686 - 0.106 * np.log10(T4) - 0.009 * (T4)**(-0.44)

def eps_recLya(n_e, n_HII, T, alpha="Draine", f_case='B'):
    """ A function calculating the luminosity density of Lya due to recombination
    n_e, n_HII should be given in cm^-3, T in K
    apu: turn on astropy units or not
    
    """
    
    if f_case == 'A':
        f_rec = f_recA(T)
    
        if alpha == "Draine":
            a = alpha_A_HII_Draine2011(T) # cm^3/s
        elif alpha == "Hui":
            a = alpha_A_HII_HuiGnedin1997(T) # cm^3/s
        elif alpha == "Fukugita":
            a = alpha_A_HII_FukugitaKawasaki1994(T) # cm^3/s
        else:
            a = alpha # cm^3/s
    if f_case == 'B':
        f_rec = f_recB(T)
    
        if alpha == "Draine":
            a = alpha_B_HII_Draine2011(T) # cm^3/s
        elif alpha == "Hui":
            a = alpha_B_HII_HuiGnedin1997(T) # cm^3/s
        else:
            a = alpha # cm^3/s
    
    # returns emissivity ε in erg/s/cm^3
    return f_rec * n_e * n_HII * a * E_Lya

