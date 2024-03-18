#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script for modelling Lyman-alpha damping-wing absorption.

Joris Witstok, 2023
"""

import os

import numpy as np
import math

from pymultinest.solve import Solver
from scipy.ndimage import median_filter, gaussian_filter1d
from scipy.interpolate import RegularGridInterpolator
from spectres import spectres

from .mean_IGM_absorption import igm_absorption
from .lymana_optical_depth import tau_IGM, tau_DLA
from .aux.stats import hpd_grid

def import_matplotlib():
    from matplotlib import pyplot as plt
    import seaborn as sns
    sns.set_style("ticks")

    return (plt, sns)

def import_corner():
    import corner
    return corner

class MN_IGM_DLA_solver(Solver):
    def __init__(self, wl_emit_intrinsic, flux_intrinsic, wl_obs, flux, flux_err, redshift, add_IGM={}, add_DLA={}, convolve={},
                    flux_intrinsic_lowerr=None, flux_intrinsic_uperr=None, print_setup=True, plot_setup=False, mpl_style=None, verbose=True,
                    mpi_run=False, mpi_comm=None, mpi_rank=0, mpi_ncores=1, mpi_synchronise=None, **solv_kwargs):
        self.mpi_run = mpi_run
        self.mpi_comm = mpi_comm
        self.mpi_rank = mpi_rank
        self.mpi_ncores = mpi_ncores
        self.mpi_synchronise = lambda _: None if mpi_synchronise is None else mpi_synchronise

        self.mpl_style = mpl_style

        self.verbose = verbose
        
        if self.mpi_rank == 0 and self.verbose:
            print("Initialising MultiNest Solver object{}...".format(" with {} cores".format(self.mpi_ncores) if self.mpi_run else ''))

        # Intrinsic wavelength and flux (both in rest frame to allow variation in redshift)
        self.wl_emit_intrinsic = wl_emit_intrinsic # Angstrom
        self.flux_intrinsic = flux_intrinsic # in units of F_λ
        self.flux_intrinsic_lowerr = flux_intrinsic_lowerr # in units of F_λ
        self.flux_intrinsic_uperr = flux_intrinsic_uperr # in units of F_λ
        self.model_uncertainty = self.flux_intrinsic_lowerr is not None and self.flux_intrinsic_uperr is not None

        # Observed wavelength and flux
        self.flux = flux # same units of F_λ as the intrinsic spectrum
        self.flux_err = flux_err # same units of F_λ as the intrinsic spectrum
        
        # Information on model parameters
        self.redshift = redshift
        self.add_IGM = add_IGM
        self.add_DLA = add_DLA
        self.convolve = convolve
        assert self.add_IGM or self.add_DLA
        if self.add_IGM:
            self.IGM_damping_interp = None
        
        self.set_prior()
        self.set_wl_arrays(wl_obs)
        if self.add_IGM:
            self.compute_IGM_curves()
        
        if print_setup and self.mpi_rank == 0 and self.verbose:
            mean_params = [np.mean(r) for r in self.theta_range]
            min_params = [r[0] for r in self.theta_range]
            max_params = [r[1] for r in self.theta_range]
            print("\n{:<5}\t{:<30}\t{:<20}\t{:<20}\t{:<20}".format('', "Parameter", "Minimum prior value", "Mean prior value", "Maximum prior value"))
            print(*["{:<5}\t{:<30}\t{:<20}\t{:<20}\t{:<20}".format(*x) for x in zip(range(1, self.n_dims+1), self.params, min_params, mean_params, max_params)], sep='\n')
            print("\n{:<5}\t{:<30}\t{:<20}\t{:<20}\t{:<20}\n".format('', "Log-likelihood", self.LogLikelihood(min_params), self.LogLikelihood(mean_params), self.LogLikelihood(max_params)))
        
        if plot_setup and self.mpi_rank == 0:
            plt, sns = import_matplotlib()
            if self.mpl_style:
                plt.style.use(self.mpl_style)
            sns.set_style("ticks")

            fig, ax = plt.subplots()
            
            z_fid = 0.5*(self.redshift["min_z"]+self.redshift["max_z"]) if self.redshift["vary"] else self.redshift["fixed_redshift"]
            ax.plot(np.nan, np.nan, color="None", label=r"Fiducial $z = {:.6g}$:".format(z_fid))
            ax.errorbar(self.wl_emit_intrinsic*(1.0 + z_fid), self.flux_intrinsic/(1.0 + z_fid),
                        yerr=[self.flux_intrinsic_lowerr/(1.0 + z_fid), self.flux_intrinsic_lowerr/(1.0 + z_fid)] if self.model_uncertainty else None,
                        color='k', alpha=0.8, label="Intrinsic")
            ax.errorbar(self.wl_obs, self.flux, yerr=self.flux_err, linestyle="None", marker='o', markersize=0.5, color='k', alpha=0.8,
                        label="Measurements")
            
            if self.add_DLA:
                for logN_HI in np.linspace(self.add_DLA["min_logN_HI"], self.add_DLA["max_logN_HI"], 5):
                    taui = tau_DLA(wl_emit_array=self.wl_emit_intrinsic, N_HI=10**logN_HI, T=self.add_DLA["T_HI"], b_turb=0.0)
                    ax.plot(self.wl_emit_intrinsic, np.exp(-taui)*self.flux_intrinsic/(1.0 + z_fid),
                            alpha=0.8, label=r"$N_\mathrm{{HI}} = 10^{{{:.1f}}} \, \mathrm{{cm^{{-2}}}}$".format(logN_HI))

            wl_emit_range = np.linspace(800, 1800, 200)
            for z, ls in zip([z_fid, self.redshift["min_z"], self.redshift["max_z"]] if self.redshift["vary"] else [self.redshift["fixed_redshift"]], ['-', '--', ':']):
                wl_obs_range = wl_emit_range * (1.0 + z)
                self.set_wl_arrays(wl_obs_range)
                
                ax.plot(np.nan, np.nan, color="None", alpha=0, label=r"$z = {:.6g}$:".format(z))
                z_list = [z] if self.redshift["vary"] else []
                
                label = ", ".join(["Model"] + [r"{} = {}$".format(l[:-1], r[0]) for r, p, l in zip(self.theta_range, self.params, self.math_labels) if p != "redshift"])
                ax.plot(wl_obs_range, self.get_profile(z_list + [r[0] for r, p in zip(self.theta_range, self.params) if p != "redshift"]), linestyle=ls, alpha=0.8, label=label)
                label = ", ".join(["Model"] + [r"{} = {}$".format(l[:-1], 0.5*(r[0]+r[1])) for r, p, l in zip(self.theta_range, self.params, self.math_labels) if p != "redshift"])
                ax.plot(wl_obs_range, self.get_profile(z_list + [0.5*(r[0]+r[1]) for r, p in zip(self.theta_range, self.params) if p != "redshift"]), linestyle=ls, alpha=0.8, label=label)
                label = ", ".join(["Model"] + [r"{} = {}$".format(l[:-1], r[1]) for r, p, l in zip(self.theta_range, self.params, self.math_labels) if p != "redshift"])
                ax.plot(wl_obs_range, self.get_profile(z_list + [r[1] for r, p in zip(self.theta_range, self.params) if p != "redshift"]), linestyle=ls, alpha=0.8, label=label)

            ax.set_xlabel(r"$\lambda_\mathrm{obs} \, (\mathrm{\AA})$")
            ax.set_ylabel(r"$F_\lambda$")

            ax.set_xlim(1100*(1.0 + z_fid), 1500*(1.0 + z_fid))

            ax.legend()
            
            plt.show()
            plt.close(fig)
            exit()

        self.mpi_synchronise(self.mpi_comm)
        super().__init__(n_dims=self.n_dims, use_MPI=self.mpi_run, **solv_kwargs)
        if self.mpi_run:
            self.samples = self.mpi_comm.bcast(self.samples, root=0)
        self.mpi_synchronise(self.mpi_comm)
        self.fitting_complete = True

    def analyse_posterior(self, hdf={}, plot_corner=True, fig=None, figsize=None, figname=None, showfig=False, color=None):
        if self.mpi_rank != 0:
            params_vals = None
            params_labs = None
        else:
            assert self.fitting_complete and hasattr(self, "samples")
            n_samples = self.samples.shape[0]

            if plot_corner:
                plt, sns = import_matplotlib()
                if self.mpl_style:
                    plt.style.use(self.mpl_style)
                sns.set_style("ticks")
                if color is None:
                    color = sns.color_palette()[0]
                corner = import_corner()
                
                n_bins = max(50, n_samples//500)
                bins = [n_bins] * len(self.params)

            data = self.samples.transpose()
            
            # Deselect non-finite data for histograms
            select_data = np.product([np.isfinite(d) for d in data], axis=0).astype(bool)
            data = [d[select_data] for d in data]
            
            if plot_corner:
                if figsize is not None:
                    fig = plt.figure(figsize=(8.27/2, 8.27/2))
                fig = corner.corner(np.transpose(data), labels=self.params, bins=bins, range=self.theta_range,
                                    smooth=1, smooth1d=1, fig=fig, color=color, show_titles=False)

                # Extract the axes
                axes_c = np.array(fig.axes).reshape((self.n_dims, self.n_dims))
                
                for ri in range(self.n_dims):
                    for ci in range(ri+1):
                        axes_c[ri, ci].set_axisbelow(False)
                
                for ci in range(self.n_dims):
                    axes_c[-1, ci].set_xlabel(self.labels[ci] + self.math_labels[ci])
                    for ax in axes_c[:, ci]:
                        ax.set_xlim(self.theta_range[ci])
                for ri in range(1, self.n_dims):
                    axes_c[ri, 0].set_ylabel(self.labels[ri] + self.math_labels[ri])
                    for ax in axes_c[ri, :ri]:
                        ax.set_ylim(self.theta_range[ri])
            
            params_hpds = {}
            for pi, param in enumerate(self.params):
                if param in hdf:
                    try:
                        params_hpds[param] = hpd_grid(gaussian_filter1d(data[pi], 1.0), alpha=1.0-hdf[param])
                    except:
                        params_hpds[param] = [np.percentile(data[pi], [0.5*(100-68.2689), 0.5*(100+68.2689)])], np.nan, np.nan, [np.median(data[pi])]
            
            params_vals = {}
            params_labs = {}
            for pi, param in enumerate(self.params):
                if param in hdf:
                    hpd_mu, _, _, modes_mu = params_hpds[param]
                    if len(modes_mu) == 1:
                        params_vals[param + "_value"] = modes_mu[0], modes_mu[0]-hpd_mu[0][0], hpd_mu[0][1]-modes_mu[0]
                        prec = max(0, 1-math.floor(np.log10(min(params_vals[param + "_value"][1:])))) if min(params_vals[param + "_value"][1:]) > 0 else 0
                        params_labs[param + "_label"] = r"{} = {:.{prec}f}_{{-{:.{prec}f}}}^{{+{:.{prec}f}}}$".format(self.math_labels[pi][:-1],
                                                                                                                        *params_vals[param + "_value"], prec=prec)
                        
                        if plot_corner:
                            hpd_3sigma, _, _, modes_3sigma = hpd_grid(data[pi], alpha=1.0-0.9973)
                            if len(hpd_3sigma) == 1:
                                for ax in axes_c[:, pi]:
                                    ax.set_xlim(max(self.theta_range[pi][0], hpd_3sigma[0][0]-3*(modes_3sigma[0]-hpd_3sigma[0][0])),
                                                min(self.theta_range[pi][1], hpd_3sigma[0][1]+3*(hpd_3sigma[0][1]-modes_3sigma[0])))
                                for ax in axes_c[pi, :pi]:
                                    ax.set_ylim(max(self.theta_range[pi][0], hpd_3sigma[0][0]-3*(modes_3sigma[0]-hpd_3sigma[0][0])),
                                                min(self.theta_range[pi][1], hpd_3sigma[0][1]+3*(hpd_3sigma[0][1]-modes_3sigma[0])))
                        
                            axes_c[pi, pi].annotate(text=params_labs[param + "_label"], xy=(0.5, 1), xytext=(0, 4),
                                                    xycoords="axes fraction", textcoords="offset points", va="bottom", ha="center",
                                                    size="small", alpha=0.8, bbox=dict(boxstyle="Round, pad=0.05", facecolor='w', edgecolor="None", alpha=0.8))
                            axes_c[pi, pi].axvline(x=modes_mu[0], color="grey", alpha=0.8)
                            axes_c[pi, pi].axvline(x=hpd_mu[0][0], linestyle='--', color="grey", alpha=0.8)
                            axes_c[pi, pi].axvline(x=hpd_mu[0][1], linestyle='--', color="grey", alpha=0.8)
                            if param == "redshift_DLA" and not self.redshift["vary"]:
                                axes_c[pi, pi].axvline(x=self.redshift["fixed_redshift"], linestyle=':', color='k', alpha=0.8)
            
            # Loop over the histograms
            for ri in range(self.n_dims):
                for ci in range(ri):
                    # High-density intervals
                    rhdf = self.params[ri] in hdf
                    chdf = self.params[ci] in hdf

                    if chdf:
                        hpd_cmu, _, _, modes_cmu = params_hpds[self.params[ci]]
                        if len(modes_cmu) == 1 and plot_corner:
                            axes_c[ri, ci].axvline(x=modes_cmu[0], color="grey", alpha=0.8)
                            axes_c[ri, ci].axvline(x=hpd_cmu[0][0], linestyle='--', color="grey", alpha=0.8)
                            axes_c[ri, ci].axvline(x=hpd_cmu[0][1], linestyle='--', color="grey", alpha=0.8)
                            if self.params[ci] == "redshift_DLA" and not self.redshift["vary"]:
                                axes_c[ri, ci].axvline(x=self.redshift["fixed_redshift"], linestyle=':', color='k', alpha=0.8)
                    
                    if rhdf:
                        hpd_rmu, _, _, modes_rmu = params_hpds[self.params[ri]]
                        if len(modes_rmu) == 1 and plot_corner:
                            axes_c[ri, ci].axvline(x=modes_rmu[0], color="grey", alpha=0.8)
                            axes_c[ri, ci].axvline(x=hpd_rmu[0][0], linestyle='--', color="grey", alpha=0.8)
                            axes_c[ri, ci].axvline(x=hpd_rmu[0][1], linestyle='--', color="grey", alpha=0.8)
                            if self.params[ri] == "redshift_DLA" and not self.redshift["vary"]:
                                axes_c[ri, ci].axhline(y=self.redshift["fixed_redshift"], linestyle=':', color='k', alpha=0.8)
                    
                    if plot_corner and rhdf and chdf:
                        # Modes
                        if len(modes_cmu) == 1 and len(modes_rmu) == 1:
                            axes_c[ri, ci].plot(modes_cmu[0], modes_rmu[0], color="grey", marker='s', mfc="None", mec="grey", alpha=0.8)
            
            if plot_corner:
                if figname:
                    fig.savefig(figname, bbox_inches="tight")
                if showfig:
                    plt.show()

                plt.close(fig)
            
        if self.mpi_run:
            params_vals = self.mpi_comm.bcast(params_vals, root=0)
            params_labs = self.mpi_comm.bcast(params_labs, root=0)
        
        return (params_vals, params_labs)
    
    def set_wl_arrays(self, wl_obs):
        # Set observed wavelength array, assuring it is covered by the intrinsic spectrum and does not have any major gaps
        self.wl_obs = wl_obs
        assert np.min(self.wl_emit_intrinsic) <= np.min(self.wl_obs / (1.0 + self.z_max))
        assert np.max(self.wl_emit_intrinsic) >= np.max(self.wl_obs / (1.0 + self.z_min))
        assert np.all(np.diff(self.wl_obs) < 2 * median_filter(self.wl_obs[:-1], size=10))

        if self.convolve:
            self.wl_obs_model = self.get_highres_wl_array(wl_obs=self.wl_obs)
        else:
            self.wl_obs_model = self.wl_obs
    
    def compute_IGM_curves(self, verbose=None):
        if verbose is None:
            verbose = self.verbose

        wl_emit_array = np.arange(0.95*np.min(self.wl_obs_model / (1.0 + self.z_max)), 1.05*np.max(self.wl_obs_model / (1.0 + self.z_min)), 0.5)
        points = [wl_emit_array]
        self.IGM_params = ["rest_wavelength"]

        if self.redshift["vary"] or self.add_IGM["vary_R_ion"] or self.add_IGM["vary_x_HI_global"]:
            if self.redshift["vary"]:
                dz = min(0.05, (self.z_max-self.z_min)/20.0)
                z_array = np.arange(self.z_min-0.05, self.z_max+0.1, dz)
                points.append(z_array)
                self.IGM_params.append("redshift")
            if self.add_IGM["vary_R_ion"]:
                dR = min(0.1, (self.add_IGM["max_R_ion"]-self.add_IGM["min_R_ion"])/20.0)
                R_ion_array = np.arange(self.add_IGM["min_R_ion"]-0.1, self.add_IGM["max_R_ion"]+0.2, dR)
                points.append(R_ion_array[R_ion_array >= 0])
                self.IGM_params.append("R_ion")
            if self.add_IGM["vary_x_HI_global"]:
                dx_HI = min(0.01, (self.add_IGM["max_x_HI_global"]-self.add_IGM["min_x_HI_global"])/20.0)
                x_HI_global_array = np.arange(self.add_IGM["min_x_HI_global"]-0.01, self.add_IGM["max_x_HI_global"]+0.02, dx_HI)
                points.append(x_HI_global_array[(x_HI_global_array >= 0) * (x_HI_global_array <= 1)])
                self.IGM_params.append("x_HI_global")
            
            array_sizes = [p.size for p in points]
            calc_curves = True
            if self.add_IGM.get("grid_file_name", False):
                assert self.add_IGM["grid_file_name"].endswith(".npz")
                if self.mpi_rank == 0:
                    if not os.path.isfile(self.add_IGM["grid_file_name"]) and verbose:
                        print("IGM damping-wing transmission curve(s) will be calculated and saved to {}...".format(self.add_IGM["grid_file_name"].split('/')[-1]))
                    else:
                        grid_file_npz = np.load(self.add_IGM["grid_file_name"])
                        grid_points = [grid_file_npz[p + "_array"] for p in self.IGM_params]
                        if np.all([s == grid_points[si].size for si, s in enumerate(array_sizes)]) and np.all([np.allclose(p, gp) for p, gp in zip(points, grid_points)]):
                            calc_curves = False
                            if verbose:
                                print("IGM damping-wing transmission curve(s) have successfully been loaded from {}...".format(self.add_IGM["grid_file_name"].split('/')[-1]))
                        else:
                            if verbose:
                                print("Failed to load IGM damping-wing transmission curve(s) from {}:".format(self.add_IGM["grid_file_name"].split('/')[-1]))
                                print("there is a shape mismatch (expected {} but loaded {} from disk), need to recompute them...".format(tuple(array_sizes), tuple(gp.size for gp in grid_points)))
            else:
                if self.mpi_rank == 0 and verbose:
                    print("IGM damping-wing transmission curve(s) will be calculated (but not saved afterwards)...")
            
            if self.mpi_run:
                calc_curves = self.mpi_comm.bcast(calc_curves, root=0)
            
            if calc_curves:
                n_curves = np.product(array_sizes[1:])
                points_mg = np.meshgrid(*points, indexing="ij")
                if self.mpi_rank == 0 and verbose:
                    print("Computing {:d} IGM damping-wing transmission curve(s) of size {:d}".format(n_curves, array_sizes[0]), end='')
                    print(" with {:d} cores...".format(self.mpi_ncores) if self.mpi_run else "...")

                mg_indices_rank = [np.arange(corei, n_curves, self.mpi_ncores) for corei in range(self.mpi_ncores)]
                IGM_damping_arrays = np.tile(np.nan, (array_sizes[0], n_curves))

                self.mpi_synchronise(self.mpi_comm)
                for mgi in mg_indices_rank[self.mpi_rank]:
                    ind = (0,) + np.unravel_index(mgi, array_sizes[1:])
                    z = points_mg[self.IGM_params.index("redshift")][ind] if self.redshift["vary"] else self.redshift["fixed_redshift"]
                    IGM_damping_arrays[:, mgi] = np.exp(-tau_IGM(wl_obs_array=wl_emit_array*(1.0+z), z_s=z,
                                                                    R_ion=points_mg[self.IGM_params.index("R_ion")][ind] if self.add_IGM["vary_R_ion"] else self.add_IGM["fixed_R_ion"],
                                                                    x_HI_global=points_mg[self.IGM_params.index("x_HI_global")][ind] if self.add_IGM["vary_x_HI_global"] else self.add_IGM["fixed_x_HI_global"],
                                                                    cosmo=self.add_IGM["cosmo"]))
                
                if self.mpi_run:
                    # Use gather to concatenate arrays from all ranks on the master rank
                    self.mpi_synchronise(self.mpi_comm)
                    IGM_damping_arrays_full = np.zeros((self.mpi_ncores, array_sizes[0], n_curves)) if self.mpi_rank == 0 else None
                    self.mpi_comm.Gather(IGM_damping_arrays, IGM_damping_arrays_full, root=0)
                    if self.mpi_rank == 0:
                        for corei in range(1, self.mpi_ncores):
                            for mgi in mg_indices_rank[corei]:
                                IGM_damping_arrays[:, mgi] = IGM_damping_arrays_full[corei, :, mgi]
                        del IGM_damping_arrays_full
                    IGM_damping_arrays = self.mpi_comm.bcast(IGM_damping_arrays, root=0)
                    self.mpi_synchronise(self.mpi_comm)
                
                IGM_damping_arrays = IGM_damping_arrays.reshape(array_sizes)
                if self.add_IGM.get("grid_file_name", False) and self.mpi_rank == 0:
                    np.savez_compressed(self.add_IGM["grid_file_name"], IGM_damping_arrays=IGM_damping_arrays,
                                        **{p + "_array": arr for arr, p in zip(points, self.IGM_params)})
            else:
                IGM_damping_arrays = grid_file_npz["IGM_damping_arrays"] if self.mpi_rank == 0 else None
                if self.mpi_run:
                    IGM_damping_arrays = self.mpi_comm.bcast(IGM_damping_arrays, root=0)
                    self.mpi_synchronise(self.mpi_comm)
        else:
            IGM_damping_arrays = np.exp(-tau_IGM(wl_obs_array=wl_emit_array*(1.0+self.redshift["fixed_redshift"]), z_s=self.redshift["fixed_redshift"],
                                                    R_ion=self.add_IGM["fixed_R_ion"], x_HI_global=self.add_IGM["fixed_x_HI_global"], cosmo=self.add_IGM["cosmo"]))
        
        self.IGM_damping_interp = RegularGridInterpolator(points=points, values=IGM_damping_arrays, method="linear", bounds_error=False)

        if self.mpi_rank == 0 and verbose:
            print("IGM damping-wing transmission curve(s) ready!")
    
    def IGM_damping_transmission(self, wl_obs_array, theta):
        assert self.IGM_damping_interp is not None
        wl_emit_array = wl_obs_array / (1.0 + (theta[self.params.index("redshift")] if self.redshift["vary"] else self.redshift["fixed_redshift"]))
        points = np.moveaxis(np.meshgrid(wl_emit_array, *[[theta[self.params.index(p)]] for p in self.IGM_params if p != "rest_wavelength"], indexing="ij"), 0, -1).squeeze()
        
        return self.IGM_damping_interp(points)
    
    def get_highres_wl_array(self, wl_obs):
        # Create higher-resolution wavelength array to convolve to lower resolution
        assert self.convolve
        wl_obs_bin_edges = []
        wl = np.min(wl_obs) - 15 * np.min(wl_obs) / np.min(self.convolve["resolution_curve"])
        
        # Require a spectral resolution of at least R_min, need to increase the number of wavelength bins
        dl_obs = self.wl_obs / np.interp(self.wl_obs, self.convolve["wl_obs_res"], self.convolve["resolution_curve"])
        self.n_res = 3.0 * max(1.0, *dl_obs/(1215.6701/self.convolve.get("R_min", 1000.0)))
        
        while wl < np.max(wl_obs) + 15 * np.max(wl_obs) / np.min(self.convolve["resolution_curve"]):
            wl_obs_bin_edges.append(wl)
            wl += wl / (self.n_res * np.interp(wl, self.convolve["wl_obs_res"], self.convolve["resolution_curve"]))
        
        wl_obs_bin_edges = np.array(wl_obs_bin_edges)
        assert wl_obs_bin_edges.size > 1
        
        return 0.5 * (wl_obs_bin_edges[:-1] + wl_obs_bin_edges[1:])
    
    def set_prior(self):
        self.params = []
        self.labels = []
        self.math_labels = []
        ranges = []

        if self.redshift["vary"]:
            ranges.append([self.redshift["min_z"], self.redshift["max_z"]])
            self.params.append("redshift")
            self.labels.append("Redshift ")
            self.math_labels.append(r"$z$")
        
        self.z_min = self.redshift["min_z"] if self.redshift["vary"] else self.redshift["fixed_redshift"]
        self.z_max = self.redshift["max_z"] if self.redshift["vary"] else self.redshift["fixed_redshift"]

        if self.add_DLA:
            ranges.append([self.add_DLA["min_logN_HI"], self.add_DLA["max_logN_HI"]])
            self.params.append("logN_HI")
            self.labels.append("HI column density\n")
            self.math_labels.append(r"$\log_{{10}} \left( N_\mathrm{{HI}} \, (\mathrm{{cm^{{-2}}}}) \right)$")
            if self.add_DLA["vary_redshift"]:
                ranges.append([self.add_DLA["min_z"], self.add_DLA["max_z"]])
                self.params.append("redshift_DLA")
                self.labels.append("DLA redshift ")
                self.math_labels.append(r"$z_\mathrm{{DLA}}$")
            if self.add_DLA["vary_b_turb"]:
                ranges.append([self.add_DLA["min_b_turb"], self.add_DLA["max_b_turb"]])
                self.params.append("b_turb")
                self.labels.append("DLA turbulent velocity\n")
                self.math_labels.append(r"$b_\mathrm{{turb, \, DLA}} \, (\mathrm{{km \, s^{{-1}}}})$")
        if self.add_IGM:
            if self.add_IGM["vary_R_ion"]:
                ranges.append([self.add_IGM["min_R_ion"], self.add_IGM["max_R_ion"]])
                self.params.append("R_ion")
                self.labels.append("Ionised bubble radius ")
                self.math_labels.append(r"$R_\mathrm{{ion}} \, (\mathrm{{pMpc}})$")
            elif self.add_IGM["vary_x_HI_global"]:
                ranges.append([self.add_IGM["min_x_HI_global"], self.add_IGM["max_x_HI_global"]])
                self.params.append("x_HI_global")
                self.labels.append("IGM HI fraction ")
                self.math_labels.append(r"$\bar{{x}}_\mathrm{{HI}}$")

        self.n_dims = len(self.params)
        self.theta_range = np.array(ranges)

    def Prior(self, cube):
        assert hasattr(self, "theta_range")
        # Scale the input unit cube to apply priors across all parameters
        for di in range(len(cube)):
            # Uniform prior
            cube[di] = cube[di] * (self.theta_range[di, 1] - self.theta_range[di, 0]) + self.theta_range[di, 0]
        
        return cube

    def get_profile(self, theta, model_uncertainty=False):
        z = theta[self.params.index("redshift")] if self.redshift["vary"] else self.redshift["fixed_redshift"]
        wl_emit_model = self.wl_obs_model / (1.0 + z) # in Angstrom
        
        # Convert intrinsic flux density between the rest frame, as provided, and the observed frame, in which the
        # flux density in units of F_λ decreases by a factor (1+z), while the wavelength increases by the same factor
        model_profile = np.interp(wl_emit_model, self.wl_emit_intrinsic, self.flux_intrinsic/(1.0 + z))
        if model_uncertainty:
            assert self.model_uncertainty
            model_profile_low = np.interp(wl_emit_model, self.wl_emit_intrinsic, (self.flux_intrinsic-self.flux_intrinsic_lowerr)/(1.0 + z))
            model_profile_up = np.interp(wl_emit_model, self.wl_emit_intrinsic, (self.flux_intrinsic+self.flux_intrinsic_uperr)/(1.0 + z))
        
        if self.add_DLA:
            wl_emit_array = self.wl_obs_model / (1.0 + theta[self.params.index("redshift_DLA")]) if self.add_DLA["vary_redshift"] else wl_emit_model
            tau_DLA_theta = tau_DLA(wl_emit_array=wl_emit_array, N_HI=10**theta[self.params.index("logN_HI")], T=self.add_DLA["T_HI"],
                                    b_turb=theta[self.params.index("b_turb")] if self.add_DLA["vary_b_turb"] else self.add_DLA.get("fixed_b_turb", 0.0))
            
            model_profile *= np.exp(-tau_DLA_theta)
            if model_uncertainty:
                model_profile_low *= np.exp(-tau_DLA_theta)
                model_profile_up *= np.exp(-tau_DLA_theta)
        
        if self.add_IGM:
            # Add the "standard" prescription for IGM absorption as well as a bespoke damping-wing absorption (interpolated from pre-computed grid)
            igm_transmission = igm_absorption(self.wl_obs_model, z) * self.IGM_damping_transmission(self.wl_obs_model, theta)

            model_profile *= igm_transmission
            if model_uncertainty:
                model_profile_low *= igm_transmission
                model_profile_up *= igm_transmission
        
        if self.convolve:
            model_profile = gaussian_filter1d(model_profile, sigma=self.n_res/(2.0 * np.sqrt(2.0 * np.log(2))),
                                                mode="nearest", truncate=5.0)
            if model_uncertainty:
                model_profile_low = gaussian_filter1d(model_profile_low, sigma=self.n_res/(2.0 * np.sqrt(2.0 * np.log(2))),
                                                        mode="nearest", truncate=5.0)
                model_profile_up = gaussian_filter1d(model_profile_up, sigma=self.n_res/(2.0 * np.sqrt(2.0 * np.log(2))),
                                                        mode="nearest", truncate=5.0)
            
            # Rebin to input wavelength array
            model_profile = spectres(self.wl_obs, self.wl_obs_model, model_profile)
            if model_uncertainty:
                model_profile_low = spectres(self.wl_obs, self.wl_obs_model, model_profile_low)
                model_profile_up = spectres(self.wl_obs, self.wl_obs_model, model_profile_up)
        
        if model_uncertainty:
            return (model_profile, model_profile_low, model_profile_up)
        else:
            return model_profile

    def LogLikelihood(self, cube):
        # Likelihood from fitting the IGM and/or DLA transmission
        if self.model_uncertainty:
            model_profile, model_profile_low, model_profile_up = self.get_profile(cube, model_uncertainty=True)
            flux_intrinsic_err = np.mean([model_profile-model_profile_low, model_profile_up-model_profile], axis=0)
            return -0.5 * np.nansum((self.flux - model_profile)**2 / (self.flux_err**2 + flux_intrinsic_err**2))
        else:
            model_profile = self.get_profile(cube, model_uncertainty=False)
            return -0.5 * np.nansum(((self.flux - model_profile) / self.flux_err)**2)


