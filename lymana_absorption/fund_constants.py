#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fundamental and Lya-related constants.

Joris Witstok, 2024
"""

import numpy as np

# Fundamental constants
c = 299792458.0 # speed of light in m/s
h = 6.62607015e-34 # Planck constant in J/Hz
k_B = 1.380649e-23 # Boltzmann constant in J/K
m_p = 1.672621e-27 # proton mass in kg
m_e = 9.109383e-31 # electon mass in kg
m_H = 1.6737236e-27 # hydrogen mass in kg
e_charge = 1.602176e-19 # electon charge in C

# Lyman-alpha (2p -> 1s transition) properties
A_Lya = 6.265e8 # Einstein A coefficient
f_Lya = 0.4162 # oscillator strength
wl_Lya = 1215.6701 # wavelength Angstrom
nu_Lya = 1e10 * c / wl_Lya # frequency in Hz
E_Lya = h * nu_Lya * 1e7 # from J to erg

K_Lya = 1e-7 * f_Lya * np.sqrt(np.pi) * e_charge**2 * c / m_e

# Unit conversions
Myr_to_seconds = 1e6 * 365.0 * 24.0 * 60.0 * 60.0 # s/Myr
Mpc_to_meter = 3.085677e22 # m/Mpc