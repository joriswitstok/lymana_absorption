#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example script for fitting a DLA/IGM absorption to R100 NIRSpec spectra.

Joris Witstok, 18 March 2024
"""

import os, sys, time

# Controls whether to run on multiple cores with mpi4py
mpi_run = True
# mpi_run = False

if mpi_run:
    from mpi4py import MPI

    # Set global communication
    mpi_comm = MPI.COMM_WORLD

    # Rank is an integer ID from 0 to n, unique to every core
    mpi_rank = mpi_comm.Get_rank()

    # Number of cores available
    mpi_ncores = mpi_comm.Get_size()
else:
    mpi_comm = None
    mpi_rank = 0
    mpi_ncores = 1

mpi_serial = mpi_ncores > 1
def mpi_synchronise(comm):
    if mpi_run:
        comm.Barrier()
        time.sleep(0.01)

if __name__ == "__main__" and mpi_rank == 0:
    print("Python", sys.version)

import numpy as np
rng = np.random.default_rng(seed=9)

# Synchonize execution
mpi_synchronise(mpi_comm)
if mpi_serial:
    print("Running with MPI: {} (rank {:d} of {:d})".format(mpi_serial, mpi_rank, mpi_ncores))
mpi_synchronise(mpi_comm)

from astropy import table
from astropy.io import fits
from spectres import spectres
from lymana_absorption.fit_lymana_absorption import MN_IGM_DLA_solver

# Import astropy cosmology, given H0 and Omega_matter
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=67.4, Om0=0.315, Ob0=0.02237/0.674**2, Tcmb0=2.726)

# Conversion factor for flux density
conv = 1e20

if mpi_rank == 0:
    with fits.open("../aux/GS_z13_3215_prism_clear_v3.1_1D.fits") as hdulist:
        wl = hdulist["WAVELENGTH"].data * 1e10 # from m to Å
    with fits.open("../aux/GS_z13_1210_3215_1D.fits") as hdulist:
        flux = hdulist["DATA"].data * 1e-7 # from W/m^3 to erg/s/cm^2/Å
        flux_err = hdulist["ERR"].data * 1e-7 # from W/m^3 to erg/s/cm^2/Å
    
    # Read in a BEAGLE model's intrinsic wavelength and flux (both in rest frame)
    BEAGLE_fits = "../aux/GS-z13-1210-3215_BEAGLE_MAP.fits"
    z_BEAGLE = float(table.Table.read(BEAGLE_fits, hdu="GALAXY PROPERTIES")["redshift"])
    wl_emit_intrinsic = np.array(table.Table.read(BEAGLE_fits, hdu="FULL SED WL")["wl"][0]) # Angstrom
    with fits.open(BEAGLE_fits) as f:
        flux_intrinsic = np.array(f["FULL SED"].data) * conv # 1/conv erg/s/cm^2/Angstrom (rest frame)
    
    res_table = table.Table.read("../aux/jwst_nirspec_prism_clear_disp.fits")
    wl_obs_res = np.array(res_table["WAVELENGTH"].data) * 1e4 # micron to Angstrom
    resolution_curve = np.array(res_table['R'].data)
else:
    wl = None
    flux = None
    flux_err = None
    wl_emit_intrinsic = None
    flux_intrinsic = None
    wl_obs_res = None
    resolution_curve = None

if mpi_run:
    wl = mpi_comm.bcast(wl, root=0)
    flux = mpi_comm.bcast(flux, root=0)
    flux_err = mpi_comm.bcast(flux_err, root=0)
    wl_emit_intrinsic = mpi_comm.bcast(wl_emit_intrinsic, root=0)
    flux_intrinsic = mpi_comm.bcast(flux_intrinsic, root=0)
    wl_obs_res = mpi_comm.bcast(wl_obs_res, root=0)
    resolution_curve = mpi_comm.bcast(resolution_curve, root=0)

# Specify parameters of the fit
z = 13.1
dz = 0.3

redshift_dict = {"vary": True, "min_z": z - dz, "max_z": z + dz}
add_IGM = {"cosmo": cosmo, "vary_R_ion": False, "fixed_R_ion": 0.0, "vary_x_HI_global": False, "fixed_x_HI_global": 1.0,
           "grid_file_name": "../IGM_curves.npz"}
add_DLA = {}
# add_DLA = {"min_logN_HI": 19, "max_logN_HI": 24, "T_HI": 100.0, "vary_redshift": False, "vary_b_turb": False}
convolve = {"R_min": 1000.0, "wl_obs_res": wl_obs_res, "resolution_curve": resolution_curve}

fit_select = (wl/(1.0+z+dz) > 1100.0) * (wl/(1.0+z-dz) < 1520.0)

wl_obs_fit = wl[fit_select]
flux_fit = flux[fit_select] * conv
flux_fit_err = flux_err[fit_select] * conv

omnrfol = "./MultiNest_IGM_DLA_GS-z13/"
if not os.path.exists(omnrfol) and mpi_rank == 0:
    os.makedirs(omnrfol)
mpi_synchronise(mpi_comm)
try:
    os.chdir(omnrfol)
    MN_solv = MN_IGM_DLA_solver(wl_emit_intrinsic=wl_emit_intrinsic, flux_intrinsic=flux_intrinsic,
                                wl_obs=wl_obs_fit, flux=flux_fit, flux_err=flux_fit_err,
                                redshift=redshift_dict, add_IGM=add_IGM, add_DLA=add_DLA, convolve=convolve,
                                plot_setup=False, mpi_run=mpi_run, mpi_comm=mpi_comm, mpi_rank=mpi_rank,
                                mpi_ncores=mpi_ncores, mpi_synchronise=mpi_synchronise,
                                outputfiles_basename="MN", n_live_points=1000, evidence_tolerance=0.5,
                                sampling_efficiency=0.8, max_iter=0, resume=True)
except Exception as e:
    raise RuntimeError("error occurred while running MultiNest fit for GS-z13...\n{}".format(e))

# Analyse results: make corner plot of posterior distributions
mpi_synchronise(mpi_comm)
os.chdir("..")
hdf = {"redshift": 0.682689, "redshift_DLA": 0.682689, "logN_HI": 0.682689}
vals, labs = MN_solv.analyse_posterior(hdf=hdf, figsize=(8.27/2, 8.27/2),
                                        figname="IGM_DLA_corner_GS-z13_vary_redshift_fixed_xHI_global_1_DLA_STScI_LSF.pdf")

if mpi_rank == 0:
    print("Best-fit redshift: {}".format(labs["redshift_label"].replace(r'$', r'').replace(r'{', r'').replace(r'}', r'')))

# Analyse results: access posterior samples, compute best-fit model
flat_samples = MN_solv.samples
samples_dict = {"{}_samples".format(p): flat_samples[:, pi] for pi, p in enumerate(MN_solv.params)}
n_samples = flat_samples.shape[0]

# Divide calculation of samples over multiple cores
wl_emit_range = np.linspace(850, 1500, 500)
wl_obs_range = wl_emit_range * (1.0 + z)
MN_solv.set_wl_arrays(wl_obs_range)

# Divide calculation of samples over multiple cores
sample_indices_rank = [np.arange(corei, n_samples, mpi_ncores) for corei in range(mpi_ncores)]
flux_cont_samples = np.tile(np.nan, (n_samples, wl_emit_range.size))

mpi_synchronise(mpi_comm)
for si in sample_indices_rank[mpi_rank]:
    flux_cont_samples[si] = MN_solv.get_profile(theta=flat_samples[si])
mpi_synchronise(mpi_comm)
if mpi_run:
    # Use gather to concatenate arrays from all ranks on the master rank
    flux_cont_samples_full = np.zeros((mpi_ncores, n_samples, wl_emit_range.size)) if mpi_rank == 0 else None
    mpi_comm.Gather(flux_cont_samples, flux_cont_samples_full, root=0)
    if mpi_rank == 0:
        for corei in range(1, mpi_ncores):
            flux_cont_samples[sample_indices_rank[corei]] = flux_cont_samples_full[corei, sample_indices_rank[corei]]

if mpi_rank == 0:
    flux_cont_median = np.median(flux_cont_samples, axis=0)
    chi2 = np.nansum(((spectres(wl_obs_fit, wl_obs_range, flux_cont_median) - flux_fit) / flux_fit_err)**2)
    n_dof = np.sum(fit_select) - MN_solv.n_dims
    red_chi2 = chi2 / n_dof