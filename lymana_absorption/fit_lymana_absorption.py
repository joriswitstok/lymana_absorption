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
from scipy.stats import norm, skewnorm, gamma

from astropy import units
from astropy.cosmology import FLRW, z_at_value

from scipy.ndimage import median_filter, gaussian_filter1d
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import solve_ivp

from spectres import spectres

from .constants import m_H, wl_Lya, E_Lya, Myr_to_seconds
from .stats import get_mode_hdi
from .mean_IGM_absorption import igm_absorption
from .lymana_optical_depth import tau_IGM, tau_DLA
from .recombination_emissivity import f_recA, f_recB, alpha_A_HII_Draine2011, alpha_B_HII_Draine2011

def import_matplotlib():
    from matplotlib import pyplot as plt
    import seaborn as sns
    sns.set_style("ticks")

    return (plt, sns)

def import_corner():
    import corner
    return corner

class MN_IGM_DLA_solver(Solver):
    def __init__(self, wl_emit_intrinsic, flux_intrinsic, wl_obs, flux, flux_err, redshift_prior, intrinsic_spectrum={}, add_IGM={}, add_DLA={}, convolve={},
                    conv=None, print_setup=True, plot_setup=False, mpl_style=None, verbose=True,
                    mpi_run=False, mpi_comm=None, mpi_rank=0, mpi_ncores=1, mpi_synchronise=None, **solv_kwargs):
        self.mpi_run = mpi_run
        self.mpi_comm = mpi_comm
        self.mpi_rank = mpi_rank
        self.mpi_ncores = mpi_ncores
        self.mpi_synchronise = lambda _: None if mpi_synchronise is None else mpi_synchronise
        self.mpi_run_str = " with {:d} cores".format(self.mpi_ncores) if self.mpi_run else ''

        self.mpl_style = mpl_style

        self.verbose = verbose
        
        if self.mpi_rank == 0 and self.verbose:
            print("Initialising MultiNest Solver object{}...".format(self.mpi_run_str))

        # Intrinsic wavelength and flux (both in rest frame to allow variation in redshift)
        self.wl_emit_intrinsic = wl_emit_intrinsic # Angstrom
        self.flux_intrinsic = flux_intrinsic # in units of F_λ
        self.model_intrinsic_spectrum = self.wl_emit_intrinsic is None or self.flux_intrinsic is None
        if self.model_intrinsic_spectrum:
            # If intrinsic spectrum not provided, parameters to model it should be given
            assert intrinsic_spectrum
            assert "wl0_emit" in intrinsic_spectrum or "wl0_observed" in intrinsic_spectrum

        # Observed wavelength and flux
        self.flux = flux # same units of F_λ as the intrinsic spectrum
        self.flux_err = flux_err # same units of F_λ as the intrinsic spectrum
        # Is error is given as inverse covariance matrix?
        self.invcov = self.flux_err.ndim == 2
        self.conv = conv
        
        # Information on model parameters
        self.redshift_prior = redshift_prior
        self.intrinsic_spectrum = intrinsic_spectrum
        self.add_IGM = add_IGM
        self.add_DLA = add_DLA
        self.add_Lya = self.model_intrinsic_spectrum and self.intrinsic_spectrum["add_Lya"]
        self.convolve = convolve
        assert self.add_IGM or self.add_DLA
        
        self.cosmo = None
        self.coupled_R_ion = {}
        self.fixed_f_esc = None
        if self.add_IGM:
            self.cosmo = self.add_IGM["cosmo"]
            assert isinstance(self.cosmo, FLRW)
            self.IGM_damping_interp = None
            if "coupled_R_ion" in self.add_IGM:
                self.add_IGM["vary_R_ion"] = True
                self.coupled_R_ion = self.add_IGM["coupled_R_ion"]
                self.fixed_f_esc = self.coupled_R_ion.get("fixed_f_esc", None)
                assert self.add_Lya
        
        self.set_prior()
        self.set_wl_arrays(wl_obs)
        if self.add_IGM:
            self.compute_IGM_curves()
        
        if print_setup and self.mpi_rank == 0 and self.verbose:
            priors_minmax = [self.get_prior_extrema(param) for param in self.params]
            mean_params = [np.mean(r) for r in priors_minmax]
            min_params = [r[0] for r in priors_minmax]
            max_params = [r[1] for r in priors_minmax]
            print("\n{:<5}\t{:<30}\t{:<20}\t{:<20}\t{:<20}".format('', "Parameter", "Minimum prior value", "Mean prior value", "Maximum prior value"))
            print(*["{:<5}\t{:<30}\t{:<20}\t{:<20}\t{:<20}".format(*x) for x in zip(range(1, self.n_dims+1), self.params, min_params, mean_params, max_params)], sep='\n')
            print("\n{:<5}\t{:<30}\t{:<20}\t{:<20}\t{:<20}\n".format('', "Log-likelihood", self.LogLikelihood(min_params), self.LogLikelihood(mean_params), self.LogLikelihood(max_params)))
        
        if plot_setup and self.mpi_rank == 0:
            plt, sns = import_matplotlib()
            if self.mpl_style:
                plt.style.use(self.mpl_style)
            sns.set_style("ticks")

            fig, axes = plt.subplots(ncols=2, squeeze=True, gridspec_kw=dict(width_ratios=[3, 1]))
            ax = axes[0]
            ax_leg = axes[1]
            ax_leg.axis("off")
            handles = []
            
            z_fid = self.fixed_redshift if self.fixed_redshift else np.mean(self.get_prior_extrema("redshift"))
            ax.plot(np.nan, np.nan, color="None", label=r"Fiducial $z = {:.6g}$:".format(z_fid))
            if not self.model_intrinsic_spectrum:
                ax.errorbar(self.wl_emit_intrinsic*(1.0 + z_fid), self.flux_intrinsic/(1.0 + z_fid),
                            color='k', alpha=0.8, label="Intrinsic")
            handles.append(ax.errorbar(self.wl_obs, self.flux, yerr=None if self.invcov else self.flux_err,
                                        linestyle="None", marker='o', markersize=0.5, color='k', alpha=0.8,
                                        label="Measurements"))
            
            wl_emit_range = self.wl_obs_model / (1.0 + z_fid)
            if self.add_DLA:
                for N_HI in np.geomspace(*self.get_prior_extrema("N_HI"), 5):
                    wl_emit_array = wl_emit_range if self.model_intrinsic_spectrum else self.wl_emit_intrinsic
                    taui = tau_DLA(wl_emit_array=wl_emit_array, N_HI=N_HI, T=self.add_DLA["T_HI"], b_turb=0.0)
                    
                    theta = [np.mean(self.get_prior_extrema(p)) if p in ["C0", "beta_UV"] else np.nan for p in self.params]
                    
                    handles.append(ax.plot(wl_emit_array, np.exp(-taui)*self.get_intrinsic_profile(theta, wl_emit_range, z=z_fid),
                                            alpha=0.8, label=r"$N_\mathrm{{HI}} = 10^{{{:.1f}}} \, \mathrm{{cm^{{-2}}}}$".format(np.log10(N_HI)))[0])

            priors_minmax = [self.get_prior_extrema(param) for param in self.params]
            for z, ls in zip([self.fixed_redshift] if self.fixed_redshift else [z_fid, *self.get_prior_extrema("redshift")], ['-', '--', ':']):
                ax.plot(np.nan, np.nan, color="None", alpha=0, label=r"$z = {:.6g}$:".format(z))
                z_list = [] if self.fixed_redshift else [z]
                
                model_spectrum = self.get_profile(z_list + [r[0] for r, p in zip(priors_minmax, self.params) if p != "redshift"], return_profile="model_spectrum")
                label = ",\n".join([r"{} = {}$".format(l[:-1], r[0]) for r, p, l in zip(priors_minmax, self.params, self.math_labels) if p != "redshift"])
                handles.append(ax.plot(self.wl_obs_model, model_spectrum, drawstyle="steps-mid", linestyle=ls, alpha=0.8, label=label)[0])
                
                model_spectrum = self.get_profile(z_list + [0.5*(r[0]+r[1]) for r, p in zip(priors_minmax, self.params) if p != "redshift"], return_profile="model_spectrum")
                label = ",\n".join([r"{} = {}$".format(l[:-1], 0.5*(r[0]+r[1])) for r, p, l in zip(priors_minmax, self.params, self.math_labels) if p != "redshift"])
                handles.append(ax.plot(self.wl_obs_model, model_spectrum, drawstyle="steps-mid", linestyle=ls, alpha=0.8, label=label)[0])
                
                model_spectrum = self.get_profile(z_list + [r[1] for r, p in zip(priors_minmax, self.params) if p != "redshift"], return_profile="model_spectrum")
                label = ",\n".join([r"{} = {}$".format(l[:-1], r[1]) for r, p, l in zip(priors_minmax, self.params, self.math_labels) if p != "redshift"])
                handles.append(ax.plot(self.wl_obs_model, model_spectrum, drawstyle="steps-mid", linestyle=ls, alpha=0.8, label=label)[0])

            ax.set_xlabel(r"$\lambda_\mathrm{obs} \, (\mathrm{\AA})$")
            ax.set_ylabel(r"$F_\lambda$")

            ax.set_xlim(1100*(1.0 + z_fid), 1500*(1.0 + z_fid))
            ax.set_ylim(np.min(self.flux), np.max(self.flux))

            ax_leg.legend(handles=handles, loc="center", fontsize="xx-small")
            
            plt.show()
            plt.close(fig)
            exit()

        self.mpi_synchronise(self.mpi_comm)
        super().__init__(n_dims=self.n_dims, use_MPI=self.mpi_run, **solv_kwargs)
        if self.mpi_run:
            self.samples = self.mpi_comm.bcast(self.samples, root=0)
        self.mpi_synchronise(self.mpi_comm)
        self.fitting_complete = True

    def mpi_calculate(self, func, samples=None, fill_value=np.nan, **func_kwargs):
        if samples is None:
            assert self.fitting_complete and hasattr(self, "samples")
            samples = self.samples
        
        n_samples = samples.shape[0]
        sample_indices_rank = np.arange(self.mpi_rank, n_samples, self.mpi_ncores)
        sample_indices_list = self.mpi_comm.gather(sample_indices_rank, root=0)
        
        n_samples_rank = sample_indices_rank.size
        sample_indices_rank_list = self.mpi_comm.gather(n_samples_rank, root=0)

        func_out = func(samples[0], **func_kwargs)
        dict_return = isinstance(func_out, dict)
        if dict_return:
            func_samples_rank_dict = {key: np.tile(fill_value, (n_samples_rank, *val.shape)) for key, val in func_out.items()}
        else:
            func_samples_rank_dict = {"main": np.tile(fill_value, (n_samples_rank, *func_out.shape))}

        self.mpi_synchronise(self.mpi_comm)
        for idx, si in enumerate(sample_indices_rank):
            sample = func(samples[si], **func_kwargs)
            if dict_return:
                for key in func_samples_rank_dict:
                    func_samples_rank_dict[key][idx] = sample[key]
            else:
                func_samples_rank_dict["main"][idx] = sample
        
        func_samples_dict = {}
        for key in func_samples_rank_dict:
            func_samples_rank = func_samples_rank_dict[key]

            self.mpi_synchronise(self.mpi_comm)
            if self.mpi_rank == 0:
                # Receive arrays from each core and combine
                func_samples = np.tile(fill_value, (n_samples, *func_samples_rank.shape[1:]))
                for corei in range(self.mpi_ncores):
                    sample_indices_rank = sample_indices_list[corei]
                    n_samples_rank = sample_indices_rank_list[corei]
                    if corei > 0:
                        self.mpi_comm.Recv(sample_indices_rank, source=corei, tag=corei+self.mpi_ncores)
                        self.mpi_comm.Recv(func_samples_rank, source=corei, tag=corei+2*self.mpi_ncores)
                    
                    func_samples[sample_indices_rank[:n_samples_rank]] = func_samples_rank[:n_samples_rank]
            else:
                self.mpi_comm.Send(sample_indices_rank, dest=0, tag=self.mpi_rank+self.mpi_ncores)
                self.mpi_comm.Send(func_samples_rank, dest=0, tag=self.mpi_rank+2*self.mpi_ncores)
                func_samples = None
            
            func_samples_dict[key] = func_samples
        
        if dict_return:
            return func_samples_dict
        else:
            return func_samples_dict["main"]

    def analyse_posterior(self, hdi_params=None, limit_params=None, logval_params=["N_HI", "F_Lya", "xi_ion"],
                          plot_corner=True, fig=None, figsize=None, figname=None, showfig=False,
                          color=None, axeslabelsize=None, annlabelsize=None,
                          IGM_DLA_npz_file=None, x_HI_values=[0.0, 0.01, 0.1, 1.0]):
        assert self.fitting_complete and hasattr(self, "samples")
        n_samples = self.samples.shape[0]
        
        if self.mpi_rank == 0:
            if self.verbose:
                print("\nAnalysing {:d} samples of posterior distributions{}...".format(n_samples, self.mpi_run_str))

            params = self.params.copy()
            labels = self.labels.copy()
            math_labels = self.math_labels.copy()
            data = self.samples.copy().transpose()
            priors_minmax = [self.get_prior_extrema(param) for param in params]
            
            if self.add_Lya:
                if "F_Lya" in self.params:
                    if self.mpi_rank == 0 and self.verbose:
                        print("Calculating ionising photon production efficiencies...")
                    # Convert observed flux (in erg/s/cm^2/Å) at 1500 Å to erg/s/cm^2/Hz (skip conversion to specific luminosity in erg/s/Hz)
                    F_nu_UV_samples = self.get_intrinsic_profile(data, 1500.0, frame="intrinsic", units="F_nu")
                    F_ion_samples = self.get_ion_strength(data, luminosity=False)
                    xi_ion_samples = F_ion_samples / F_nu_UV_samples
                    
                    params.append("xi_ion")
                    if "xi_ion" not in logval_params: logval_params.append("xi_ion")
                    labels.append("Ion. photon prod. eff.\n")
                    math_labels.append(r"$\xi_\mathrm{{ion}} \, (\mathrm{{Hz \, erg^{{-1}}}})$")
                    data = np.append(data, xi_ion_samples.reshape(1, n_samples), axis=0)
                    priors_minmax.append([np.min(xi_ion_samples), np.max(xi_ion_samples)])
                    del xi_ion_samples
                else:
                    if self.mpi_rank == 0 and self.verbose:
                        print("Calculating observed Lyα fluxes...")
                    assert self.coupled_R_ion and "xi_ion" in self.params
                    # Observed Lyα flux in erg/s/cm^2
                    F_Lya_samples = self.get_Lya_strength(data)
                    
                    params.append("F_Lya")
                    if "F_Lya" not in logval_params: logval_params.append("F_Lya")
                    labels.append(r"Intrinsic Ly$\mathrm{\alpha}$ flux" + '\n')
                    math_labels.append(r"$F_\mathrm{{Ly \alpha}} \, (\mathrm{{erg \, s^{{-1}} \, cm^{{-2}}}})$")
                    data = np.append(data, F_Lya_samples.reshape(1, n_samples), axis=0)
                    priors_minmax.append([np.min(F_Lya_samples), np.max(F_Lya_samples)])
                    del F_Lya_samples
                
                # Apply unit conversion to the Lya flux
                data[params.index("F_Lya")] = data[params.index("F_Lya")] / self.conv
                priors_minmax[params.index("F_Lya")] = tuple(f/self.conv for f in priors_minmax[params.index("F_Lya")])
        
        if self.add_Lya:
            if self.coupled_R_ion:
                if self.mpi_rank == 0 and self.verbose:
                    print("Solving for ionised bubble sizes in {:d} samples...".format(n_samples))
                
                R_ion_samples = self.mpi_calculate(self.get_coupled_R_ion)
                
                if self.mpi_rank == 0:
                    params.append("R_ion")
                    labels.append("Ionised bubble radius\n")
                    math_labels.append(r"$R_\mathrm{{ion}} \, (\mathrm{{pMpc}})$")
                    data = np.append(data, R_ion_samples.reshape(1, n_samples), axis=0)
                    priors_minmax.append((np.nanmin(R_ion_samples), np.nanmax(R_ion_samples)))
                    del R_ion_samples
        
        if self.mpi_rank == 0:
            sort_indices = [params.index(p) for p in sorted(params, key=lambda p: self.all_params.index(p))]
            data = data[sort_indices]
            params = [params[idx] for idx in sort_indices]
            labels = [labels[idx] for idx in sort_indices]
            math_labels = [math_labels[idx] for idx in sort_indices]
            priors_minmax = [priors_minmax[idx] for idx in sort_indices]
            
            for param in logval_params:
                math_labels[params.index(param)] = r"$\log_{{10}} \left( " + math_labels[params.index(param)][1:-1] + r" \right)$"
                data[params.index(param)] = np.log10(data[params.index(param)])
                priors_minmax[params.index(param)] = tuple(np.log10(priors_minmax[params.index(param)]))
            
            n_dims = len(params)
            if limit_params is None:
                param_limits = {}
            if hdi_params is None:
                hdi_params = {}
                for param in params:
                    if param not in limit_params:
                        hdi_params[param] = 0.682689
            
            # Deselect non-finite data for histograms
            select_data = np.prod([np.isfinite(d) for d in data], axis=0).astype(bool)
            data = [d[select_data] for d in data]
            del select_data
            
            if self.verbose:
                print("Finding best-fit values of {:d} parameters...".format(n_dims))
        
        param_limits = {}
        params_hpds = {}
        params_perc = {}
        if self.mpi_rank == 0:
            for pi, param in enumerate(params):
                if param in limit_params:
                    param_limits[param] = np.quantile(data[pi], limit_params[param])
                if param in hdi_params:
                    params_hpds[param] = get_mode_hdi(data[pi], prob=hdi_params[param])
                params_perc[param] = ([np.median(data[pi])], [np.percentile(data[pi], [0.5*(100-68.2689), 0.5*(100+68.2689)])])
        
        self.mpi_synchronise(self.mpi_comm)
        if self.mpi_run:
            param_limits = self.mpi_comm.bcast(param_limits, root=0)
            params_hpds = self.mpi_comm.bcast(params_hpds, root=0)
        self.mpi_synchronise(self.mpi_comm)

        if self.mpi_rank != 0:
            params_vals = None
            params_labs = None
        else:
            if plot_corner:
                plt, sns = import_matplotlib()
                if self.mpl_style:
                    plt.style.use(self.mpl_style)
                sns.set_style("ticks")
                if color is None:
                    color = sns.color_palette()[0]
                corner = import_corner()
            
                if self.verbose:
                    print("Producing {:d}-parameter corner plot...".format(n_dims))
                
                n_bins = max(50, n_samples//500)
                bins = [n_bins] * n_dims
                
                if figsize is None:
                    figsize = (8.27/2, 8.27/2)
                if fig is None:
                    fig = plt.figure(figsize=figsize)
                if axeslabelsize is None:
                    axeslabelsize = plt.rcParams("axes.labelsize")
                if annlabelsize is None:
                    annlabelsize = plt.rcParams("font.size")
                
                if n_dims > 1:
                    fig = corner.corner(np.transpose(data), labels=params, bins=bins, range=priors_minmax,
                                        fig=fig, color=color, show_titles=False)
                else:
                    ax = fig.add_subplot(fig.add_gridspec(nrows=2, ncols=1, hspace=0, height_ratios=[1, 0])[0, :])
                    hist, bin_edges = np.histogram(data[0], bins=bins[0], range=priors_minmax[0])
                    ax.plot(0.5*(bin_edges[1:]+bin_edges[:-1]), gaussian_filter1d(hist, sigma=0.5), drawstyle="steps-mid", color=color)
                
                del data, bins

                # Extract the axes
                axes_c = np.array(fig.axes).reshape((n_dims, n_dims))
                
                for ri in range(n_dims):
                    for ci in range(ri+1):
                        axes_c[ri, ci].tick_params(axis="both", which="both", direction="in", top=ri!=ci, right=ri!=ci, labelsize=axeslabelsize)
                        axes_c[ri, ci].set_axisbelow(False)
                
                for ci in range(n_dims): 
                    axes_c[-1, ci].set_xlabel(labels[ci] + math_labels[ci], fontsize=axeslabelsize)
                    for ax in axes_c[ci:, ci]:
                        ax.set_xlim(priors_minmax[ci])
                for ri in range(1, n_dims):
                    axes_c[ri, 0].set_ylabel(labels[ri] + math_labels[ri], fontsize=axeslabelsize)
                    for ax in axes_c[ri, :ri]:
                        ax.set_ylim(priors_minmax[ri])
            
            if self.verbose:
                print("Obtaining (labels for) best-fit values...\n")

            params_vals = {}
            params_labs = {}
            for pi, param in enumerate(params):
                if param in limit_params:
                    params_vals[param + "_value"] = [param_limits[param], np.nan, np.nan]
                    prec = max(0, 1-math.floor(np.log10(params_vals[param + "_value"][0]))) if params_vals[param + "_value"][0] > 0 else 0
                    params_labs[param + "_label"] = r"{} {} {:.{prec}f}$".format(math_labels[pi][:-1], '>' if limit_params[param] < 0.5 else '<',
                                                                                 *params_vals[param + "_value"], prec=prec)
                else:
                    modes_mu, hpd_mu = params_hpds.get(param, params_perc[param])
                    
                    params_vals[param + "_value"] = modes_mu[0], modes_mu[0]-hpd_mu[0][0], hpd_mu[0][1]-modes_mu[0]
                    prec = max(0, 1-math.floor(np.log10(min(params_vals[param + "_value"][1:])))) if min(params_vals[param + "_value"][1:]) > 0 else 0
                    params_labs[param + "_label"] = r"{} = {:.{prec}f}_{{-{:.{prec}f}}}^{{+{:.{prec}f}}}$".format(math_labels[pi][:-1],
                                                                                                                  *params_vals[param + "_value"], prec=prec)
                    
                    if self.verbose:
                        print_str = params_labs[param + "_label"].replace('$', '').replace('{', '').replace("_-", ' -').replace('^', ' ')
                        print_str = print_str.replace('}', '').replace(r"\mathrm", '').replace(r"\AA", 'Å')
                        print_str = print_str.replace(r"\left", '').replace(r"\right", '').replace(r"\, ", '').replace(r"\log", "log")
                        print("Best-fit values of {}:\n{}".format(param, print_str))
                
                if plot_corner:
                    ha = "left" if param in logval_params or param == "C0" else "center"
                    axes_c[pi, pi].annotate(text=params_labs[param + "_label"],
                                            xy=(0 if ha == "left" else 0.5, 1), xytext=(4 if ha == "left" else 0, 4),
                                            xycoords="axes fraction", textcoords="offset points",
                                            va="bottom", ha="left" if ha == "left" else "center",
                                            size=annlabelsize, alpha=0.8, bbox=dict(boxstyle="Round, pad=0.05", facecolor='w', edgecolor="None", alpha=0.8))
                    if param == "redshift_DLA" and self.fixed_redshift:
                        axes_c[pi, pi].axvline(x=self.fixed_redshift, linestyle=':', color='k', alpha=0.8)
                    
                    if param in limit_params:
                        axes_c[pi, pi].axvline(x=params_vals[param + "_value"][0], linestyle='--', color="grey", alpha=0.8)
                    else:
                        if len(modes_mu) == 1:
                            for ax in axes_c[pi:, pi]:
                                ax.set_xlim(max(priors_minmax[pi][0], params_vals[param + "_value"][0]-5*params_vals[param + "_value"][1]),
                                            min(priors_minmax[pi][1], params_vals[param + "_value"][0]+5*params_vals[param + "_value"][2]))
                            for ax in axes_c[pi, :pi]:
                                ax.set_ylim(max(priors_minmax[pi][0], params_vals[param + "_value"][0]-5*params_vals[param + "_value"][1]),
                                            min(priors_minmax[pi][1], params_vals[param + "_value"][0]+5*params_vals[param + "_value"][2]))

                        axes_c[pi, pi].axvline(x=modes_mu[0], color="grey", alpha=0.8)
                        axes_c[pi, pi].axvline(x=hpd_mu[0][0], linestyle='--', color="grey", alpha=0.8)
                        axes_c[pi, pi].axvline(x=hpd_mu[0][1], linestyle='--', color="grey", alpha=0.8)
            
            if self.verbose:
                print("\nAnnotating corner plot...")
            
            if plot_corner:
                # Loop over the histograms
                for ri in range(n_dims):
                    for ci in range(ri):
                        if params[ci] == "redshift_DLA" and self.fixed_redshift:
                            axes_c[ri, ci].axvline(x=self.fixed_redshift, linestyle=':', color='k', alpha=0.8)
                        if params[ri] == "redshift_DLA" and self.fixed_redshift:
                            axes_c[ri, ci].axhline(y=self.fixed_redshift, linestyle=':', color='k', alpha=0.8)
                        
                        if params[ri] in limit_params:
                            axes_c[ri, ci].axvline(x=params_vals[params[ri] + "_value"][0], linestyle='--', color="grey", alpha=0.8)
                        else:
                            modes_rmu, hpd_rmu = params_hpds.get(params[ri], params_perc[params[ri]])
                            
                            axes_c[ri, ci].axhline(y=modes_rmu[0], color="grey", alpha=0.8)
                            axes_c[ri, ci].axhline(y=hpd_rmu[0][0], linestyle='--', color="grey", alpha=0.8)
                            axes_c[ri, ci].axhline(y=hpd_rmu[0][1], linestyle='--', color="grey", alpha=0.8)
                        
                        if params[ci] in limit_params:
                            axes_c[ri, ci].axvline(x=params_vals[params[ci] + "_value"][0], linestyle='--', color="grey", alpha=0.8)
                        else:
                            modes_cmu, hpd_cmu = params_hpds.get(params[ci], params_perc[params[ci]])
                            
                            axes_c[ri, ci].axvline(x=modes_cmu[0], color="grey", alpha=0.8)
                            axes_c[ri, ci].axvline(x=hpd_cmu[0][0], linestyle='--', color="grey", alpha=0.8)
                            axes_c[ri, ci].axvline(x=hpd_cmu[0][1], linestyle='--', color="grey", alpha=0.8)
                            
                        if not params[ri] in limit_params and not params[ci] in limit_params:
                            # Modes/medians
                            axes_c[ri, ci].plot(modes_cmu[0], modes_rmu[0], color="grey", marker='s', mfc="None", mec="grey", alpha=0.8)
            
                if figname:
                    fig.savefig(figname, bbox_inches="tight")
                if showfig:
                    plt.show()

                plt.close(fig)
                del fig, axes_c, ax, params, labels, math_labels, priors_minmax
                
                if self.verbose:
                    print("Saved and closed corner plot!")
        
        self.mpi_synchronise(self.mpi_comm)
        if self.mpi_run:
            params_vals = self.mpi_comm.bcast(params_vals, root=0)
            params_labs = self.mpi_comm.bcast(params_labs, root=0)
        self.mpi_synchronise(self.mpi_comm)

        rdict = {}
        if IGM_DLA_npz_file:
            if self.mpi_rank == 0 and self.verbose:
                print("Calculating best-fit spectra{}{}{} of size {:d}{}...".format(", IGM transmission curves" if self.add_IGM else '',
                                                                                    ", DLA transmission curves" if self.add_DLA else '',
                                                                                    ", Lyα profiles" if self.add_DLA else '',
                                                                                    self.wl_obs_model.size, self.mpi_run_str))

            profiles = self.mpi_calculate(self.get_profile, return_profile="all")
            if self.mpi_rank == 0:
                observed_spectrum_samples = profiles["observed_spectrum"]
                model_spectrum_samples = profiles["model_spectrum"]
                if self.add_IGM:
                    igm_transm_samples = profiles["igm_transmission"]
                if self.add_DLA:
                    dla_transm_samples = profiles["dla_transmission"]
                if self.add_Lya:
                    # Only consider wavelength region of ±5000 km/s around Lyα
                    Lya_reg = (self.wl_obs_model/(1.0+self.z_min) >= 1196.0) * (self.wl_obs_model/(1.0+self.z_max) <= 1236.0)
                    rdict["wl_obs_model_Lya"] = self.wl_obs_model[Lya_reg]
                    Lya_intr_profile_samples = profiles["Lya_profile"][:, Lya_reg]
                del profiles
            
                if self.verbose:
                    print("Finished calculations! Analysing profiles...")

                rdict["model_spectrum_median"] = np.median(model_spectrum_samples, axis=0)
                rdict["observed_spectrum_median"] = np.median(observed_spectrum_samples, axis=0)
                rdict["chi2"] = self.calc_chi2(self.flux - rdict["observed_spectrum_median"])
                n_dof = self.n_chan - self.n_dims
                rdict["red_chi2"] = rdict["chi2"] / n_dof
                if self.verbose:
                    print("Best-fit χ^2 ({:d} wavelength bins, {:d} free parameters): {:.1f}".format(self.n_chan, self.n_dims, rdict["chi2"]))
                    print("Best-fit reduced χ^2 ({:d} DOF): {:.1f}".format(n_dof, rdict["red_chi2"]))

                rdict["model_spectrum_lowerr"] = rdict["model_spectrum_median"] - np.percentile(model_spectrum_samples, 0.5*(100-68.2689), axis=0)
                rdict["model_spectrum_uperr"] = np.percentile(model_spectrum_samples, 0.5*(100+68.2689), axis=0) - rdict["model_spectrum_median"]
                del model_spectrum_samples
                rdict["observed_spectrum_lowerr"] = rdict["observed_spectrum_median"] - np.percentile(observed_spectrum_samples, 0.5*(100-68.2689), axis=0)
                rdict["observed_spectrum_uperr"] = np.percentile(observed_spectrum_samples, 0.5*(100+68.2689), axis=0) - rdict["observed_spectrum_median"]
                del observed_spectrum_samples
                
                if self.add_DLA:
                    rdict["dla_transm_median"] = np.median(dla_transm_samples, axis=0)
                    rdict["dla_transm_lowerr"] = rdict["dla_transm_median"] - np.percentile(dla_transm_samples, 0.5*(100-68.2689), axis=0)
                    rdict["dla_transm_uperr"] = np.percentile(dla_transm_samples, 0.5*(100+68.2689), axis=0) - rdict["dla_transm_median"]
                    del dla_transm_samples
                
                if self.add_Lya:
                    Lya_F_intr_samples = self.get_Lya_strength(self.samples.transpose()) / self.conv
                    rdict["Lya_F_intr_perc"] = np.percentile(Lya_F_intr_samples, [0.5*(100-68.2689), 50, 0.5*(100+68.2689)])
                    rdict["Lya_F_intr_median"] = rdict["Lya_F_intr_perc"][1]
                    rdict["Lya_F_intr_lowerr"], rdict["Lya_F_intr_uperr"] = np.diff(rdict["Lya_F_intr_perc"])
                    
                    norm = np.max(Lya_intr_profile_samples, axis=1).reshape(n_samples, 1)
                    Lya_intr_profile_samples_perc = np.percentile(Lya_intr_profile_samples/norm, [0.5*(100-68.2689), 50, 0.5*(100+68.2689)], axis=0)
                    rdict["Lya_intr_profile_median"] = Lya_intr_profile_samples_perc[1]
                    rdict["Lya_intr_profile_lowerr"], rdict["Lya_intr_profile_uperr"] = np.diff(Lya_intr_profile_samples_perc, axis=0)
                    
                    Lya_transm_profile_samples = Lya_intr_profile_samples * igm_transm_samples[:, Lya_reg]
                    Lya_transm_profile_samples_perc = np.percentile(Lya_transm_profile_samples/norm, [0.5*(100-68.2689), 50, 0.5*(100+68.2689)], axis=0)
                    rdict["Lya_transm_profile_median"] = Lya_transm_profile_samples_perc[1]
                    rdict["Lya_transm_profile_lowerr"], rdict["Lya_transm_profile_uperr"] = np.diff(Lya_transm_profile_samples_perc, axis=0)
                    
                    Lya_f_esc_samples = np.sum(Lya_transm_profile_samples, axis=1) / np.sum(Lya_intr_profile_samples, axis=1)
                    rdict["Lya_intr_profile_samples"] = Lya_intr_profile_samples
                    rdict["Lya_transm_profile_samples"] = Lya_transm_profile_samples
                    del Lya_intr_profile_samples, Lya_transm_profile_samples
                    rdict["Lya_f_esc_perc"] = np.percentile(Lya_f_esc_samples, [0.5*(100-68.2689), 50, 0.5*(100+68.2689)])
                    rdict["Lya_f_esc_median"] = rdict["Lya_f_esc_perc"][1]
                    rdict["Lya_f_esc_lowerr"], rdict["Lya_f_esc_uperr"] = np.diff(rdict["Lya_f_esc_perc"])
                    
                    Lya_F_obs_samples = Lya_f_esc_samples * Lya_F_intr_samples
                    rdict["Lya_F_obs_perc"] = np.percentile(Lya_F_obs_samples, [0.5*(100-68.2689), 50, 0.5*(100+68.2689)])
                    rdict["Lya_F_obs_median"] = rdict["Lya_F_obs_perc"][1]
                    rdict["Lya_F_obs_lowerr"], rdict["Lya_F_obs_uperr"] = np.diff(rdict["Lya_F_obs_perc"])
                    if self.verbose:
                        print("Observed Lyα flux: {:.3g} -{:.3g} +{:.3g} 10^{:d} erg/s/cm^2".format(rdict["Lya_F_obs_perc"][1]*self.conv,
                                                                                                    *np.diff(rdict["Lya_F_obs_perc"])*self.conv,
                                                                                                    int(-np.log10(self.conv))))
                        print("Lyα escape fraction: {:.3g} -{:.3g} +{:.3g}".format(rdict["Lya_f_esc_perc"][1], *np.diff(rdict["Lya_f_esc_perc"])))
                    
                    # Find the observed Lyα wavelength (in micron)
                    z_samples = self.fixed_redshift if self.fixed_redshift else self.samples[:, self.params.index("redshift")]
                    Lya_wl_obs_samples = self.get_Lya_wl_emit(self.samples.transpose(), z=z_samples) * (1.0 + z_samples) / 1e4
                    
                    rdict["Lya_wl_obs_perc"] = np.percentile(Lya_wl_obs_samples, [0.5*(100-68.2689), 50, 0.5*(100+68.2689)])
                    rdict["Lya_wl_obs_median"] = rdict["Lya_wl_obs_perc"][1]
                    rdict["Lya_wl_obs_lowerr"], rdict["Lya_wl_obs_uperr"] = np.diff(rdict["Lya_wl_obs_perc"])
                    if self.verbose:
                        print("Observed Lyα wavelength: {:.3g} -{:.3g} +{:.3g} μm".format(rdict["Lya_wl_obs_perc"][1],
                                                                                            *np.diff(rdict["Lya_wl_obs_perc"])))

                if self.add_IGM:
                    rdict["igm_transm_median"] = np.median(igm_transm_samples, axis=0)
                    rdict["igm_transm_lowerr"] = rdict["igm_transm_median"] - np.percentile(igm_transm_samples, 0.5*(100-68.2689), axis=0)
                    rdict["igm_transm_uperr"] = np.percentile(igm_transm_samples, 0.5*(100+68.2689), axis=0) - rdict["igm_transm_median"]
                    del igm_transm_samples
                    
                    # Compute flux profiles at fixed x_HI = 0.01, 0.1, 1.0; fix redshift and intrinsic spectrum, but record initial values
                    redshift_prior_init = self.redshift_prior.copy()
                    self.redshift_prior = {"type": "fixed", "params": [self.fixed_redshift if self.fixed_redshift else params_vals["redshift_value"][0]]}
                    if self.model_intrinsic_spectrum:
                        intrinsic_spectrum_init = self.intrinsic_spectrum.copy()
                        self.intrinsic_spectrum["fixed_C0"] = params_vals["C0_value"][0]
                        self.intrinsic_spectrum["fixed_beta_UV"] = params_vals["beta_UV_value"][0]
                    
                    # Reset IGM properties, but record initial values
                    add_IGM_init = self.add_IGM.copy()
                    self.add_IGM["vary_R_ion"] = False
                    self.add_IGM["fixed_R_ion"] = 0.0
                    self.add_IGM["vary_x_HI_global"] = False
                    
                    # Remove DLA, Lyα emission, and ionised bubble, but record their initial values
                    add_DLA_init = self.add_DLA.copy()
                    add_Lya_init = self.add_Lya
                    coupled_R_ion_init = self.coupled_R_ion.copy()
                    self.add_DLA = {}
                    self.add_Lya = False
                    self.coupled_R_ion = {}
                    
                    self.set_prior()
                    for x_HI in x_HI_values:
                        x_HI_key = "{:.5g}".format(x_HI).replace("0.", '')
                        self.add_IGM["fixed_x_HI_global"] = x_HI

                        self.compute_IGM_curves(verbose=False)
                        rdict["model_spectrum_fixed_xHI" + x_HI_key] = self.get_profile(theta=None, return_profile="model_spectrum")
                        rdict["chi2_fixed_xHI" + x_HI_key] = self.calc_chi2(self.flux - self.get_profile(theta=None))
                        print("χ^2 for fixed x_HI = {:.3g}: {:.1f}".format(x_HI, rdict["chi2_fixed_xHI" + x_HI_key]))
                    
                    # Reset values
                    self.redshift_prior = redshift_prior_init
                    if self.model_intrinsic_spectrum:
                        self.intrinsic_spectrum = intrinsic_spectrum_init
                    self.add_IGM = add_IGM_init
                    self.add_DLA = add_DLA_init
                    self.add_Lya = add_Lya_init
                    self.coupled_R_ion = coupled_R_ion_init
                    self.set_prior()

                np.savez_compressed(IGM_DLA_npz_file, wl_obs_model=self.wl_obs_model,
                                    grid_file_name=self.add_IGM.get("grid_file_name", None),
                                    **rdict, **params_vals, **params_labs)
                print("Saved results to {}".format(IGM_DLA_npz_file))
        
        self.mpi_synchronise(self.mpi_comm)
        if self.mpi_run:
            rdict = self.mpi_comm.bcast(rdict, root=0)
        self.mpi_synchronise(self.mpi_comm)
        
        if self.mpi_rank == 0 and self.verbose:
            print("Finished evaluating posterior distributions!")
        
        return (rdict, params_vals, params_labs)
    
    def set_wl_arrays(self, wl_obs):
        # Set observed wavelength array, assuring it is covered by the intrinsic spectrum and does not have any major gaps
        self.wl_obs = wl_obs
        self.n_chan = self.wl_obs.size
        if not self.model_intrinsic_spectrum:
            assert np.min(self.wl_emit_intrinsic) <= np.min(self.wl_obs / (1.0 + self.z_max))
            assert np.max(self.wl_emit_intrinsic) >= np.max(self.wl_obs / (1.0 + self.z_min))
        assert np.all(np.diff(self.wl_obs) < 2 * median_filter(self.wl_obs[:-1], size=10))

        if self.convolve:
            self.wl_obs_model = self.get_highres_wl_array(wl_obs=self.wl_obs)
        else:
            self.wl_obs_model = self.wl_obs
    
    def get_highres_wl_array(self, wl_obs):
        # Create higher-resolution wavelength array to convolve to lower resolution
        assert self.convolve
        wl_obs_bin_edges = []
        wl = np.min(wl_obs) - 5 * np.min(wl_obs) / np.min(self.convolve["resolution_curve"])
        
        # Require a spectral resolution of at least R_min, need to increase the number of wavelength bins
        dl_obs = self.wl_obs / np.interp(self.wl_obs, self.convolve["wl_obs_res"], self.convolve["resolution_curve"])
        dl_obs_min = wl_Lya * (1.0+self.z_min) / self.convolve.get("R_min", 1000.0)
        if self.add_Lya:
            dl_obs_min = min(dl_obs_min, self.get_Lya_sigma_l(theta=None, get_minimum=True))
        self.n_res = max(5.0, *dl_obs/dl_obs_min)
        
        while wl < np.max(wl_obs) + 5 * np.max(wl_obs) / np.min(self.convolve["resolution_curve"]):
            wl_obs_bin_edges.append(wl)
            wl += wl / (self.n_res * np.interp(wl, self.convolve["wl_obs_res"], self.convolve["resolution_curve"]))
        
        wl_obs_bin_edges = np.array(wl_obs_bin_edges)
        assert wl_obs_bin_edges.size > 1
        
        return 0.5 * (wl_obs_bin_edges[:-1] + wl_obs_bin_edges[1:])

    def get_Lya_wl_emit(self, theta, z, get_minimum=False):
        # Obtain wavelength of Lyα emission line in the rest frame, taking any velocity shift into account
        if "fixed_wl_emit_Lya" in self.intrinsic_spectrum:
            wl_emit_Lya = self.intrinsic_spectrum["fixed_wl_emit_Lya"]
        elif "fixed_wl_obs_Lya" in self.intrinsic_spectrum:
            wl_emit_Lya = self.intrinsic_spectrum["fixed_wl_obs_Lya"] / (1.0 + (self.z_max if get_minimum else z))
        else:
            delta_v = self.get_prior_extrema("delta_v_Lya")[0] if get_minimum else self.get_val(theta, "delta_v_Lya")
            wl_emit_Lya = wl_Lya / (1.0 - delta_v/299792.458)
        
        return wl_emit_Lya
    
    def get_Lya_sigma_l(self, theta, z=None, wl_emit_Lya=None, get_minimum=False):
        if get_minimum:
            wl_emit_Lya = self.get_Lya_wl_emit(theta=None, z=z, get_minimum=True)
        else:
            assert wl_emit_Lya is not None
        
        if "fixed_sigma_l_Lya" in self.intrinsic_spectrum:
            sigma_l = self.intrinsic_spectrum["fixed_sigma_l_Lya"]
        elif "fixed_sigma_v_Lya" in self.intrinsic_spectrum:
            assert wl_emit_Lya is not None
            sigma_l = wl_emit_Lya / (299792.458/self.intrinsic_spectrum["fixed_sigma_v_Lya"] - 1.0)
        else:
            sigma_v = self.get_prior_extrema("sigma_v_Lya")[0] if get_minimum else self.get_val(theta, "sigma_v_Lya")
            sigma_l = wl_emit_Lya / (299792.458/sigma_v - 1.0)
        
        return sigma_l
    
    def get_Lya_profile(self, theta, wl_obs_model=None, z=None, frame="observed"):
        if wl_obs_model is None:
            wl_obs_model = self.wl_obs_model
        if z is None:
            z = self.fixed_redshift if self.fixed_redshift else self.get_val(theta, "redshift")
        wl_emit_model = wl_obs_model / (1.0 + z) # in Angstrom
        
        A = self.get_Lya_strength(theta, z=z)
        wl_emit_Lya = self.get_Lya_wl_emit(theta, z=z)
        sigma_l = self.get_Lya_sigma_l(theta, z=z, wl_emit_Lya=wl_emit_Lya)

        profile = A / (np.sqrt(2*np.pi)*sigma_l) * np.exp( -(wl_emit_model-wl_emit_Lya)**2 / (2.0*sigma_l**2) )
        if frame == "observed":
            # Convert intrinsic flux density between the rest frame, as provided, and the observed frame, in which the
            # flux density in units of F_λ decreases by a factor (1+z), while the wavelength increases by the same factor
            profile /= (1.0 + z)
        else:
            assert frame == "intrinsic"
        
        return profile

    def get_intrinsic_profile(self, theta, wl_emit_model, z=None, get_maximum=False, frame="observed", units="F_lambda"):
        if z is None and not get_maximum:
            z = self.fixed_redshift if self.fixed_redshift else self.get_val(theta, "redshift")
        if get_maximum:
            z = self.z_min
        
        if self.model_intrinsic_spectrum:
            C0 = (self.intrinsic_spectrum["fixed_C0"] if "fixed_C0" in self.intrinsic_spectrum else self.get_val(theta, "C0", get_maximum=get_maximum)) * (1.0 + z)
            
            if "wl0_observed" in self.intrinsic_spectrum:
                wl0 = self.intrinsic_spectrum["wl0_observed"] / (1.0 + z)
            else:
                wl0 = self.intrinsic_spectrum["wl0_emit"]
            
            if "fixed_beta_UV" in self.intrinsic_spectrum:
                beta = self.intrinsic_spectrum["fixed_beta_UV"]
            else:
                if get_maximum:
                    beta = np.where(wl_emit_model > wl0, self.get_prior_extrema("beta_UV")[1], self.get_prior_extrema("beta_UV")[0])
                else:
                    beta = self.get_val(theta, "beta_UV")
            
            profile = C0 * (wl_emit_model/wl0)**beta
        else:
            profile = np.interp(wl_emit_model, self.wl_emit_intrinsic, self.flux_intrinsic)
        
        if frame == "observed":
            # Convert intrinsic flux density between the rest frame, as provided, and the observed frame, in which the
            # flux density in units of F_λ decreases by a factor (1+z), while the wavelength increases by the same factor
            profile /= (1.0 + z)
        else:
            assert frame == "intrinsic"
        
        if units.startswith('L'):
            # Convert flux 1/conv erg/s/cm^2 to luminosity in erg/s
            profile *= 1.0 / self.conv * 4.0 * np.pi * self.cosmo.luminosity_distance(z).to("cm").value**2 # in erg/s
        if units.endswith("_nu"):
            # Convert from F_lambda to F_nu
            profile *= wl_emit_model**2 / 299792458.0e10
        
        return profile
    
    def compute_IGM_curves(self, verbose=None):
        if verbose is None:
            verbose = self.verbose

        wl_obs_array = np.arange(0.95*np.min(self.wl_obs_model), 1.05*np.max(self.wl_obs_model), 0.25 * (1.0 + self.z_min))
        points = [wl_obs_array]
        self.IGM_params = ["observed_wavelength"]

        if not self.fixed_redshift or self.add_IGM["vary_R_ion"] or self.add_IGM["vary_x_HI_global"]:
            if not self.fixed_redshift:
                dz = min(0.05, (self.z_max-self.z_min)/20.0)

                z_array = np.arange(self.z_min-0.05, self.z_max+0.1, dz)
                points.append(z_array)
                self.IGM_params.append("redshift")
            if self.add_IGM["vary_R_ion"]:
                if self.coupled_R_ion:
                    minR, maxR = (1e-2, self.get_coupled_R_ion(theta=None, get_maximum=True))
                else:
                    minR, maxR = self.get_prior_extrema("R_ion")
                max_logR = np.log10(maxR)
                min_logR = min(np.log10(minR), max_logR-2.0)
                dlogR = min(0.2, (max_logR-min_logR)/20.0)
                
                R_ion_array = np.concatenate([[0.0], 10**np.arange(min_logR-0.2, max_logR+0.4, dlogR)], axis=0)
                points.append(R_ion_array)
                self.IGM_params.append("R_ion")
            if self.add_IGM["vary_x_HI_global"]:
                dx_HI = min(0.01, (self.add_IGM["max_x_HI_global"]-self.add_IGM["min_x_HI_global"])/20.0)

                x_HI_global_array = np.arange(self.add_IGM["min_x_HI_global"]-0.01, self.add_IGM["max_x_HI_global"]+0.02, dx_HI)
                points.append(x_HI_global_array[(x_HI_global_array >= 0)])
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
                        correct_shape = np.all([s == grid_points[si].size for si, s in enumerate(array_sizes)])
                        if correct_shape and np.all([np.allclose(p, gp) for p, gp in zip(points, grid_points)]):
                            calc_curves = False
                            if verbose:
                                print("IGM damping-wing transmission curve(s) have successfully been loaded from {}...".format(self.add_IGM["grid_file_name"].split('/')[-1]))
                        else:
                            if verbose:
                                print("Failed to load IGM damping-wing transmission curve(s) from {}:".format(self.add_IGM["grid_file_name"].split('/')[-1]))
                                if correct_shape:
                                    print_str = "the grid points did not match up"
                                else:
                                    print_str = "there is a shape mismatch (expected {} but loaded {} from disk)".format(tuple(array_sizes), tuple(gp.size for gp in grid_points))
                                print("{}, need to recompute the curve(s)...".format(print_str, tuple(array_sizes), tuple(gp.size for gp in grid_points)))
            else:
                if self.mpi_rank == 0 and verbose:
                    print("IGM damping-wing transmission curve(s) will be calculated (but not saved afterwards)...")
            
            if self.mpi_run:
                calc_curves = self.mpi_comm.bcast(calc_curves, root=0)
            
            if calc_curves:
                n_curves = np.prod(array_sizes[1:])
                points_mg = np.meshgrid(*points, indexing="ij")
                if self.mpi_rank == 0 and verbose:
                    print("Computing {:d} IGM damping-wing transmission curve(s) of size {:d}{}...".format(n_curves, array_sizes[0], self.mpi_run_str))

                mg_indices_rank = [np.arange(corei, n_curves, self.mpi_ncores) for corei in range(self.mpi_ncores)]
                IGM_damping_arrays = np.tile(np.nan, (array_sizes[0], n_curves))

                self.mpi_synchronise(self.mpi_comm)
                for mgi in mg_indices_rank[self.mpi_rank]:
                    ind = (0,) + np.unravel_index(mgi, array_sizes[1:])
                    z = self.fixed_redshift if self.fixed_redshift else points_mg[self.IGM_params.index("redshift")][ind]
                    R_ion = points_mg[self.IGM_params.index("R_ion")][ind] if self.add_IGM["vary_R_ion"] else self.add_IGM["fixed_R_ion"]
                    x_HI_global = points_mg[self.IGM_params.index("x_HI_global")][ind] if self.add_IGM["vary_x_HI_global"] else self.add_IGM["fixed_x_HI_global"]
                    
                    IGM_damping_arrays[:, mgi] = np.exp(-tau_IGM(wl_obs_array=wl_obs_array, z_s=z, R_ion=R_ion, x_HI_global=x_HI_global,
                                                                 cosmo=self.cosmo, use_vector=True))
                
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
            IGM_damping_arrays = np.exp(-tau_IGM(wl_obs_array=wl_obs_array, z_s=self.fixed_redshift,
                                                 R_ion=self.add_IGM["fixed_R_ion"], x_HI_global=self.add_IGM["fixed_x_HI_global"],
                                                 cosmo=self.cosmo, use_vector=True))
        
        self.IGM_damping_interp = RegularGridInterpolator(points=points, values=IGM_damping_arrays, method="linear", bounds_error=False)

        if self.mpi_rank == 0 and verbose:
            print("IGM damping-wing transmission curve(s) ready!")
    
    def IGM_damping_transmission(self, wl_obs_model, theta_IGM):
        assert self.IGM_damping_interp is not None
        points = np.moveaxis(np.meshgrid(wl_obs_model, *[[theta_IGM[p]] for p in self.IGM_params if p != "observed_wavelength"], indexing="ij"), 0, -1).squeeze()
        
        return self.IGM_damping_interp(points)

    def get_Lya_strength(self, theta, z=None, luminosity=False, get_maximum=False):
        if "F_Lya" in self.params:
            # Use Lyα observed flux in erg/s/cm^2 to get an observed flux of ionising photons
            F_Lya = self.get_val(theta, "F_Lya", get_maximum=get_maximum)

            if luminosity:
                # Calculate the Lyα luminosity in erg/s (from its flux in 1/conv erg/s/cm^2)
                if z is None:
                    z = self.fixed_redshift if self.fixed_redshift else self.get_val(theta, "redshift", get_maximum=get_maximum)
                F_Lya = F_Lya / self.conv * 4.0 * np.pi * self.cosmo.luminosity_distance(z).to("cm").value**2 # in erg/s
        else:
            assert self.coupled_R_ion and "xi_ion" in self.params
            
            f_rec = {'A': f_recA, 'B': f_recB}[self.coupled_R_ion["case"]]
            f_esc = self.fixed_f_esc if self.fixed_f_esc else self.get_val(theta, "f_esc", get_minimum=get_maximum)

            F_ion = self.get_ion_strength(theta, z=z, luminosity=luminosity, get_maximum=get_maximum)
            
            # Convert ionising photon production rate to Lyα luminosity
            F_Lya = F_ion * (f_rec(self.coupled_R_ion["T_gas"]) * E_Lya) * (1.0 - f_esc)

        return F_Lya

    def get_ion_strength(self, theta, z=None, luminosity=True, get_maximum=False):
        if z is None:
            z = self.fixed_redshift if self.fixed_redshift else self.get_val(theta, "redshift", get_maximum=get_maximum)
        
        if "F_Lya" in self.params:
            # Lyα observed luminosity (flux) in 1/conv erg/s/cm^2 to get the rate (flux) of ionising photons in 1/s(/conv/cm^2)
            L_Lya = self.get_Lya_strength(theta, z=z, luminosity=luminosity, get_maximum=get_maximum)

            # Convert Lyα luminosity (flux) to an ionising photon rate (flux)
            f_rec = {'A': f_recA, 'B': f_recB}[self.coupled_R_ion["case"]]
            f_esc = self.fixed_f_esc if self.fixed_f_esc else self.get_val(theta, "f_esc", get_maximum=get_maximum)
            F_ion = L_Lya / (f_rec(self.coupled_R_ion["T_gas"]) * E_Lya) / (1.0 - f_esc)
        else:
            assert self.coupled_R_ion and "xi_ion" in self.params
            
            f_rec = {'A': f_recA, 'B': f_recB}[self.coupled_R_ion["case"]]
            f_esc = self.fixed_f_esc if self.fixed_f_esc else self.get_val(theta, "f_esc", get_minimum=get_maximum)
        
            # Convert intrinsic flux at 1500 Å to ionising photon rate (flux)
            F_nu_UV = self.get_intrinsic_profile(theta, 1500.0, frame="intrinsic", units="L_nu" if luminosity else "F_nu",
                                                 get_maximum=get_maximum)
            F_ion = F_nu_UV * self.get_val(theta, "xi_ion", get_maximum=get_maximum)

        return F_ion

    def get_coupled_R_ion(self, theta, full_integration=None, z=None, get_maximum=False):
        if z is None and not get_maximum:
            z = self.fixed_redshift if self.fixed_redshift else self.get_val(theta, "redshift")
        
        dN_ion_dt_Myr = self.get_ion_strength(theta, z=z, get_maximum=get_maximum) * Myr_to_seconds # from 1/s to 1/Myr

        # Mean hydrogen number density (in 1/Mpc^3) at redshift z = 0, case-B recombination rate
        n_H_0 = (self.coupled_R_ion["f_H"] * self.cosmo.critical_density(0) * self.cosmo.Ob(0) / (m_H * units.kg)).to("1/Mpc^3").value
        alpha_B = (alpha_B_HII_Draine2011(self.coupled_R_ion["T_gas"]) * (units.cm**3/units.s)).to("Mpc^3/Myr").value # from cm^3/s to Mpc^3/Myr
        age_Myr = self.coupled_R_ion["age_Myr"]

        if full_integration is None:
            full_integration = self.coupled_R_ion.get("full_integration", True)
        if get_maximum:
            full_integration = False
            z = self.z_min
            f_esc = self.fixed_f_esc if self.fixed_f_esc else 1.0
        else:
            f_esc = self.fixed_f_esc if self.fixed_f_esc else self.get_val(theta, "f_esc")

        if full_integration:
            # Solve ODE to get ionised bubble size in Mpc
            def dRdt(t, R_ion3):
                if hasattr(z, "__len__"):
                    z_t = []
                    for zi in z:
                        z_t.append(z_at_value(lambda z_t: self.cosmo.age(zi) - self.cosmo.age(z_t), (age_Myr-t)*units.Myr,
                                              zmin=z-0.1, zmax=1000, method="bounded").value)
                    z_t = np.reshape(z_t, z.shape)
                else:
                    z_t = z_at_value(lambda z_t: self.cosmo.age(z) - self.cosmo.age(z_t), (age_Myr-t)*units.Myr,
                                     zmin=z-0.1, zmax=1000, method="bounded").value

                n_H_bar = n_H_0 * (1.0 + z_t)**3
                Hubble_rec = (3.0 * self.cosmo.H(z_t).to("1/Myr").value - self.coupled_R_ion["C_HII"] * n_H_bar * alpha_B) * R_ion3
                ion = 3.0 * f_esc * dN_ion_dt_Myr / (4.0 * np.pi * n_H_bar)
                return Hubble_rec + ion
            
            R_ion3 = solve_ivp(fun=dRdt, t_span=(0.0, age_Myr), y0=[0.0], t_eval=[age_Myr],
                                dense_output=True, vectorized=True, method="RK45").y[0, 0]
        else:
            # Neglect Hubble flow and recombinations in Eq. (3) of Cen & Haiman (2000) to get ionised bubble size in Mpc
            n_H_bar = n_H_0 * (1.0 + z)**3
            R_ion3 = 3.0 * f_esc * dN_ion_dt_Myr * age_Myr / (4.0 * np.pi * n_H_bar)
        
        return R_ion3**(1.0/3.0)

    def get_igm_transmission(self, theta, wl_obs_model=None, z=None):
        if wl_obs_model is None:
            wl_obs_model = self.wl_obs_model
        if z is None:
            z = self.fixed_redshift if self.fixed_redshift else self.get_val(theta, "redshift")
        
        theta_IGM = {}
        for p in self.IGM_params:
            if p == "observed_wavelength":
                continue
            elif p == "R_ion":
                theta_IGM[p] = self.get_coupled_R_ion(theta) if self.coupled_R_ion else self.get_val(theta, p)
            else:
                theta_IGM[p] = self.get_val(theta, p)
        
        return igm_absorption(wl_obs_model, z) * self.IGM_damping_transmission(wl_obs_model, theta_IGM)

    def get_dla_transmission(self, theta, wl_obs_model=None):
        if wl_obs_model is None:
            wl_obs_model = self.wl_obs_model
        
        if self.add_DLA["vary_redshift"]:
            wl_emit_array = wl_obs_model / (1.0 + self.get_val(theta, "redshift_DLA"))
        else:
            z = self.fixed_redshift if self.fixed_redshift else self.get_val(theta, "redshift")
            wl_emit_array = wl_obs_model / (1.0 + z)
        
        tau_DLA_theta = tau_DLA(wl_emit_array=wl_emit_array, N_HI=self.get_val(theta, "N_HI"), T=self.add_DLA["T_HI"],
                                b_turb=self.get_val(theta, "b_turb") if self.add_DLA["vary_b_turb"] else self.add_DLA.get("fixed_b_turb", 0.0))
        
        return np.exp(-tau_DLA_theta)

    def get_profile(self, theta, wl_obs=None, return_profile="observed_spectrum"):
        z = self.fixed_redshift if self.fixed_redshift else self.get_val(theta, "redshift")
        wl_emit_model = self.wl_obs_model / (1.0 + z) # in Angstrom
        if wl_obs is None:
            wl_obs = self.wl_obs
        
        profiles = {}
        
        # Obtain model profile (converted to the observed frame)
        model_spectrum = self.get_intrinsic_profile(theta, wl_emit_model=wl_emit_model, z=z)
        
        if self.add_DLA:
            dla_transmission = self.get_dla_transmission(theta)
            model_spectrum *= dla_transmission
            if return_profile in ["dla_transmission", "all"]:
                profiles["dla_transmission"] = dla_transmission
        
        if self.add_Lya:
            Lya_profile = self.get_Lya_profile(theta)
            model_spectrum += Lya_profile
            if return_profile in ["Lya_profile", "all"]:
                profiles["Lya_profile"] = Lya_profile
        
        if self.add_IGM:
            # Add the "standard" prescription for IGM absorption as well as a bespoke damping-wing absorption (interpolated from pre-computed grid)
            igm_transmission = self.get_igm_transmission(theta)
            model_spectrum *= igm_transmission
            if return_profile in ["igm_transmission", "all"]:
                profiles["igm_transmission"] = igm_transmission
        
        if self.convolve:
            model_spectrum = gaussian_filter1d(model_spectrum, sigma=self.n_res/(2.0 * np.sqrt(2.0 * np.log(2))),
                                                mode="nearest", truncate=5.0)
            
            if return_profile in ["model_spectrum", "all"]:
                profiles["model_spectrum"] = model_spectrum
            if return_profile in ["observed_spectrum", "all"]:
                # Rebin to input wavelength array
                profiles["observed_spectrum"] = spectres(wl_obs, self.wl_obs_model, model_spectrum)
        else:
            profiles["model_spectrum"] = model_spectrum
            profiles["observed_spectrum"] = model_spectrum
        
        if return_profile == "all":
            return profiles
        else:
            return profiles[return_profile]
    
    def set_prior(self):
        self.params = []
        self.priors = []
        self.labels = []
        self.math_labels = []

        self.fixed_redshift = float(self.redshift_prior["params"][0]) if self.redshift_prior["type"] == "fixed" else None
        if self.fixed_redshift:
            self.z_min, self.z_max = self.fixed_redshift, self.fixed_redshift
        else:
            self.params.append("redshift")
            self.priors.append(self.redshift_prior)
            self.z_min, self.z_max = self.get_prior_extrema("redshift")
            self.labels.append("Redshift ")
            self.math_labels.append(r"$z$")

        if self.model_intrinsic_spectrum:
            if not "fixed_C0" in self.intrinsic_spectrum:
                self.params.append("C0")
                self.priors.append(self.intrinsic_spectrum["C0_prior"])
                self.labels.append("Cont. normalisation\n")
                self.math_labels.append(r"$C \, (10^{{{:d}}} \, \mathrm{{erg \, s^{{-1}} \, cm^{{-2}} \, \AA^{{-1}}}})$".format(int(-np.log10(self.conv))))
            if not "fixed_beta_UV" in self.intrinsic_spectrum:
                self.params.append("beta_UV")
                self.priors.append(self.intrinsic_spectrum["beta_UV_prior"])
                self.labels.append("UV slope ")
                self.math_labels.append(r"$\beta_\mathrm{{UV}}$")
            if self.add_Lya:
                self.params.append("F_Lya")
                self.priors.append(self.intrinsic_spectrum["F_Lya_prior"])
                self.labels.append(r"Intrinsic Ly$\mathrm{\alpha}$ flux" + '\n')
                self.math_labels.append(r"$F_\mathrm{{Ly \alpha}} \, (\mathrm{{erg \, s^{{-1}} \, cm^{{-2}}}})$")
            if not "fixed_wl_emit_Lya" in self.intrinsic_spectrum and not "fixed_wl_obs_Lya" in self.intrinsic_spectrum:
                assert "delta_v_Lya_prior" in self.intrinsic_spectrum
                if self.intrinsic_spectrum["delta_v_Lya_prior"]["type"] == "fixed":
                    self.intrinsic_spectrum["fixed_wl_emit_Lya"] = wl_Lya / (1.0 - self.intrinsic_spectrum["delta_v_Lya_prior"]["params"][0]/299792.458)
                else:
                    self.params.append("delta_v_Lya")
                    self.priors.append(self.intrinsic_spectrum["delta_v_Lya_prior"])
                    self.labels.append(r"Ly$\mathrm{\alpha}$ velocity offset" + '\n')
                    self.math_labels.append(r"$\Delta v_\mathrm{{Ly \alpha}} \, (\mathrm{{km \, s^{{-1}}}})$")
            if not "fixed_sigma_l_Lya" in self.intrinsic_spectrum and not "fixed_sigma_v_Lya" in self.intrinsic_spectrum:
                self.priors.append(self.intrinsic_spectrum["sigma_v_Lya_prior"])
                self.params.append("sigma_v_Lya")
                self.labels.append(r"Ly$\mathrm{\alpha}$ velocity dispersion" + '\n')
                self.math_labels.append(r"$\sigma_\mathrm{{Ly \alpha}} \, (\mathrm{{km \, s^{{-1}}}})$")

        if self.add_DLA:
            self.params.append("N_HI")
            self.priors.append(self.add_DLA["N_HI_prior"])
            self.labels.append("HI column density\n")
            self.math_labels.append(r"$N_\mathrm{{HI}} \, (\mathrm{{cm^{{-2}}}})$")
            if self.add_DLA["vary_redshift"]:
                self.params.append("redshift_DLA")
                self.priors.append(self.add_DLA["redshift_DLA_prior"])
                self.labels.append("DLA redshift ")
                self.math_labels.append(r"$z_\mathrm{{DLA}}$")
            if self.add_DLA["vary_b_turb"]:
                self.params.append("b_turb")
                self.priors.append(self.add_DLA["b_turb_prior"])
                self.labels.append("DLA turbulent velocity\n")
                self.math_labels.append(r"$b_\mathrm{{turb, \, DLA}} \, (\mathrm{{km \, s^{{-1}}}})$")
        if self.add_IGM:
            if self.add_IGM["vary_R_ion"]:
                if self.coupled_R_ion:
                    if not self.fixed_f_esc:
                        # Instead of varying F_Lya, vary the ionising photon production efficiency
                        F_Lya_idx = self.params.index("F_Lya")
                        del self.params[F_Lya_idx], self.priors[F_Lya_idx], self.labels[F_Lya_idx], self.math_labels[F_Lya_idx]
                        
                        self.params.append("xi_ion")
                        self.priors.append(self.coupled_R_ion["xi_ion_prior"])
                        self.labels.append("Ion. photon prod. eff.\n")
                        self.math_labels.append(r"$\xi_\mathrm{{ion}} \, (\mathrm{{Hz \, erg^{{-1}}}})$")

                        self.params.append("f_esc")
                        self.priors.append(self.coupled_R_ion["f_esc_prior"])
                        self.labels.append("LyC escape fraction\n")
                        self.math_labels.append(r"$f_\mathrm{{esc, \, LyC}}$")
                else:
                    self.params.append("R_ion")
                    self.priors.append(self.add_IGM["R_ion_prior"])
                    self.labels.append("Ionised bubble radius\n")
                    self.math_labels.append(r"$R_\mathrm{{ion}} \, (\mathrm{{pMpc}})$")
            if self.add_IGM["vary_x_HI_global"]:
                self.params.append("x_HI_global")
                self.priors.append(self.add_IGM["x_HI_global_prior"])
                self.labels.append("IGM HI fraction ")
                self.math_labels.append(r"$\bar{{x}}_\mathrm{{HI}}$")
        
        # Sort parameters according to predetermined order
        self.all_params = ["redshift", "N_HI", "redshift_DLA", "b_turb", "C0", "beta_UV",
                            "x_HI_global", "R_ion", "xi_ion", "f_esc", "F_Lya", "delta_v_Lya", "sigma_v_Lya"]
        sort_indices = [self.params.index(p) for p in sorted(self.params, key=lambda p: self.all_params.index(p))]
        self.params = [self.params[idx] for idx in sort_indices]
        self.priors = [self.priors[idx] for idx in sort_indices]
        self.labels = [self.labels[idx] for idx in sort_indices]
        self.math_labels = [self.math_labels[idx] for idx in sort_indices]

        self.n_dims = len(self.params)
    
    def get_val(self, theta, param, get_minimum=False, get_maximum=False):
        if get_minimum or get_maximum:
            return self.get_prior_extrema(param)[1 if get_maximum else 0]
        else:
            return theta[self.params.index(param)]
    
    def get_prior_extrema(self, param):
        """Function for getting the minimum and maximum of a prior. Intended for internal use.

        """
        prior = self.priors[self.params.index(param)]
        if prior["type"].lower() == "uniform":
            minimum = float(prior["params"][0])
            maximum = float(prior["params"][1])
        elif prior["type"].lower() == "loguniform":
            # Set the minimum/maximum value as the edge of the uniform distribution or as a 3σ outlier
            minimum = 10**(float(prior["params"][0]))
            maximum = 10**(float(prior["params"][1]))
        elif prior["type"].lower() == "normal":
            # Set the minimum/maximum value as the edge of the uniform distribution or as a 3σ outlier
            minimum = float(prior["params"][0]) - 3 * float(prior["params"][1])
            maximum = float(prior["params"][0]) + 3 * float(prior["params"][1])
        elif prior["type"].lower() == "lognormal":
            # Set the minimum/maximum value as the edge of the uniform distribution or as a 3σ outlier
            minimum = 10**(float(prior["params"][0]) - 3 * float(prior["params"][1]))
            maximum = 10**(float(prior["params"][0]) + 3 * float(prior["params"][1]))
        elif prior["type"].lower() == "gamma":
            minimum = 0
            maximum = float(prior["params"][0]) * float(prior["params"][1]) + 5 * float(prior["params"][1])
        elif prior["type"].lower() == "fixed":
            minimum = float(prior["params"][0])
            maximum = float(prior["params"][0])
        else:
            raise TypeError("prior type '{}' not recognised".format(prior["type"]))
        
        return (minimum, maximum)

    def Prior(self, cube):
        """Function for calculating a prior. Intended for internal use.

        """
        assert hasattr(self, "priors")
        assert hasattr(self, "params")

        # Scale the input unit cube to apply priors across all parameters
        for di in range(len(cube)):
            if self.priors[di]["type"].lower() == "uniform":
                # Uniform distribution as prior
                cube[di] = cube[di] * (float(self.priors[di]["params"][1]) - float(self.priors[di]["params"][0])) + float(self.priors[di]["params"][0])
            elif self.priors[di]["type"].lower() == "loguniform":
                # Log-uniform distribution as prior
                cube[di] = 10**(cube[di] * (float(self.priors[di]["params"][1]) - float(self.priors[di]["params"][0])) + float(self.priors[di]["params"][0]))
            elif self.priors[di]["type"].lower() == "normal":
                # Normal distribution as prior
                cube[di] = norm.ppf(cube[di], loc=float(self.priors[di]["params"][0]), scale=float(self.priors[di]["params"][1]))
            elif self.priors[di]["type"].lower() == "lognormal":
                # Log-normal distribution as prior
                cube[di] = 10**norm.ppf(cube[di], loc=float(self.priors[di]["params"][0]), scale=float(self.priors[di]["params"][1]))
            elif self.priors[di]["type"].lower() == "gamma":
                # Gamma distribution as prior (prior belief: unlikely to be very high)
                cube[di] = gamma.ppf(cube[di], a=float(self.priors[di]["params"][0]), loc=0, scale=float(self.priors[di]["params"][1]))
            else:
                raise TypeError("prior type '{}' of parameter {:d}, {}, not recognised".format(self.priors[di]["type"], di+1, self.params[di]))
        
        return cube

    def calc_chi2(self, diff, err=None):
        if err is None:
            err = self.flux_err
        
        if self.invcov:
            diff = np.where(np.isnan(diff), 0, diff)
            return np.linalg.multi_dot([diff, err, diff])
        else:
            return np.nansum((diff / err)**2)

    def LogLikelihood(self, theta):
        # Likelihood from fitting the IGM and/or DLA transmission
        diff = self.flux - self.get_profile(theta)
        
        return -0.5 * self.calc_chi2(diff=diff)


