import numpy as np
import math
import os
from scipy import interpolate
from scipy.linalg import sqrtm
from dix_inversion.raylee_sensitivity import raylee_sensitivity
from dix_inversion.utils import (
    update_covariances, check_model_validity, compute_misfit, print_convergence_summary, check_nans
)
from dix_inversion.linvers import linvers

# -------------------------------------------------------------------------------------
# Function to compute Rayleigh wave phase and group velocity kernels for a homogeneous half-space
# -------------------------------------------------------------------------------------
# Based on the Dix formulation from Haney & Tsai (2017)
def compute_homogeneous_dix_kernels(v: float, kz: np.ndarray):
    """
    Compute Rayleigh wave phase and group velocity kernels for a homogeneous half-space
    based on the Dix formulation from Haney & Tsai (2017).

    Parameters
    ----------
    v : float
        Poisson's ratio (typically between 0.25 and 0.49)
    kz : np.ndarray
        2D array of wavenumber * depth values (shape: [n_freqs, n_depths])

    Returns
    -------
    f_homo : np.ndarray
        Phase velocity kernel matrix (same shape as kz)
    f_homo_U : np.ndarray
        Group velocity kernel matrix (same shape as kz)

    Raises
    ------
    ValueError
        If input leads to complex or unstable results.

    Notes
    -----
    Based on: Haney & Tsai (2017), Geophysics, 82(3), F15–F28.
    """

    # Step 1: Compute Rayleigh/shear velocity ratio (f) analytically
    f = np.sqrt((8 / 3) - ((-16 + 56 * v - 40 * (v ** 2)) /
    (3 * (2 ** (2 / 3)) * (-1 + v) * (-11 + 78 * v - 123 * (v ** 2) + 56 * (v ** 3) +
    3 * np.sqrt(3) * np.sqrt(-5 + 36 * v - 94 * (v ** 2) + 148 * (v ** 3) - 165 * (v ** 4) + 112 * (v ** 5) - 32 * (v ** 6))) ** (1 / 3))) +
    ((1 / (3 * (-1 + v))) * (2 ** (2 / 3)) * (-11 + 78 * v - 123 * (v ** 2) + 56 * (v ** 3) +
    3 * np.sqrt(3) * np.sqrt(-5 + 36 * v - 94 * (v ** 2) + 148 * (v ** 3) - 165 * (v ** 4) + 112 * (v ** 5) - 32 * (v ** 6))) ** (1 / 3)))
    
    # Ensure there isn't a small imaginary part
    f = np.real(f)

    # Step 2: Compute denominator g of the Dix expression in Haney and Tsai (2015)
    try:
        g = -((((-2 + f ** 2) ** 3) / ((1 - f ** 2) ** (3 / 2))) - 
              (((2 * np.sqrt(2) * np.sqrt((-2 + f ** 2 + 2 * v - 2 * (f ** 2) * v) / (-1 + v)) * 
                 (4 - 4 * v + (f ** 2) * (-1 + 2 * v)))) / ((2 - 2 * v + (f ** 2) * (-1 + 2 * v)))) +
                 ((8 * (-2 + f ** 2) * (2 - 2 * f ** 2 + np.sqrt(2 - 2 * (f ** 2)) * np.sqrt((-2 + f ** 2 + 2 * v - 2 * (f ** 2) * v) / (-1 + v)))) /
        ((-1 + f ** 2) * (2 * np.sqrt(1 - f ** 2) + np.sqrt(2) * np.sqrt((-2 + f ** 2 + 2 * v - 2 * (f ** 2) * v) / (-1 + v))))))
    except Exception:
        raise ValueError("Numerical error in computing denominator g.")

    # Step 3: Compute coefficients and exponents used in the homogeneous formulation
    fctr1 = -((f ** 2 - 2) ** 2) * (8 - 8 * (f ** 2) + f ** 4) / ((1 - f ** 2) ** 1.5) / g
    exp1 = 2 * np.sqrt(1 - f ** 2)

    fctr2 = -8 * (f ** 2 - 2) * 2 * (1 + np.sqrt(1 - f ** 2) * np.sqrt(1 + ((f ** 2) * ((2 * v - 1) / (2 - 2 * v))))) / np.sqrt(1 - f ** 2) / g
    exp2 = np.sqrt(1 - f ** 2) + np.sqrt(1 + ((f ** 2) * ((2 * v - 1) / (2 - 2 * v))))

    fctr3 = 4 * np.sqrt(1 + ((f ** 2) * ((2 * v - 1) / (2 - 2 * v)))) * (16 * (v - 1) + (f ** 2) * (f ** 2 - 8) * (2 * v - 1)) / (2 - 2 * v + (f ** 2) * (2 * v - 1)) / g
    exp3 = 2 * np.sqrt(1 + ((f ** 2) * ((2 * v - 1) / (2 - 2 * v))))

    # Step 4: Construct phase and group velocity kernels
    f_homo = fctr3 * np.exp(-exp3 * kz) + fctr2 * np.exp(-exp2 * kz) + fctr1 * np.exp(-exp1 * kz)
    f_homo[:, -1] = 0  # truncate final depth

    f_homo_U = fctr3 * (1 - exp3 * kz) * np.exp(-exp3 * kz) + \
               fctr2 * (1 - exp2 * kz) * np.exp(-exp2 * kz) + \
               fctr1 * (1 - exp1 * kz) * np.exp(-exp1 * kz)
    f_homo_U[:, -1] = 0

    return f_homo, f_homo_U

# --------------------------------------------------------------------------------------------------
# ----------------------------
# Initialization of Dix Kernels
# ----------------------------

def rayleigh_phase_homogeneous(v, kz, alpha):
    # compute and return f_ray for inv_flag = 1 
    f_homo, _ = compute_homogeneous_dix_kernels(v, kz)
    G = np.diff(f_homo, axis=1)
    return G

def rayleigh_phase_powerlaw_025(v, kz, alpha):
    # inv_flag = 2
    f_ray = -103.14 * np.exp(-1.8586 * kz) + 6.1446 * np.exp(-1.7714 * kz) + \
        217.120 * np.exp(-1.7555 * kz) - 10.312 * np.exp(-1.7012 * kz) - \
        160.68 * np.exp(-1.6842 * kz) + 1.2856 * np.exp(-1.6683 * kz) - \
        115.00 * np.exp(-1.6524 * kz) + 294.66 * np.exp(-1.6140 * kz) + \
        4.0924 * np.exp(-1.5981 * kz) - 135.49 * np.exp(-1.5438 * kz)
    G = np.diff(f_ray, axis=1)
    return G

def love_phase_powerlaw(v, kz, alpha):
    # inv_flag = 3
    f_love = -(1 + 0.85 ** 2) * np.exp(-2 * 0.85 * kz)
    G = np.diff(f_love, axis=1)
    return G

def rayleigh_group_homogeneous(v, kz, alpha):
    # inv_flag = 4
    _, f_homo_U = compute_homogeneous_dix_kernels(v, kz)
    G = np.diff(f_homo_U, axis=1)
    return G

def rayleigh_phase_powerlaw_030(v, kz, alpha):
    # inv_flag = 5
    f_ray = -130.253 * np.exp(-1.8362 * kz) + 6.88812 * np.exp(-1.7556 * kz) + \
        271.540 * np.exp(-1.7380 * kz) - 12.1077 * np.exp(-1.6859 * kz) - \
        183.478 * np.exp(-1.6750 * kz) + 1.70238 * np.exp(-1.6574 * kz) - \
        142.361 * np.exp(-1.6398 * kz) + 341.126 * np.exp(-1.6053 * kz) + \
        4.76194 * np.exp(-1.5877 * kz) - 159.030 * np.exp(-1.5356 * kz)
    G = np.diff(f_ray, axis=1)
    return G

def rayleigh_group_powerlaw_025(v, kz, alpha):
    # inv_flag = 6
    G = ((1 - alpha) ** 2) * rayleigh_phase_powerlaw_025(v, kz, alpha)
    return G

def love_group_powerlaw(v, kz, alpha):
    # inv_flag = 7
    G = ((1 - alpha) ** 2) * love_phase_powerlaw(v, kz, alpha)
    return G

def rayleigh_group_powerlaw_030(v, kz, alpha):
    # inv_flag = 8
    G = ((1 - alpha) ** 2) * rayleigh_phase_powerlaw_030(v, kz, alpha)
    return G

# # ----------------------------
# # Dispatcher
# # ----------------------------

def compute_dix_kernel(inv_flag, v, kz, alpha=None):
    """
    Dispatches the appropriate kernel function based on inv_flag.
    """
    if inv_flag == 1:
        return rayleigh_phase_homogeneous(v, kz, alpha)
    elif inv_flag == 2:
        return rayleigh_phase_powerlaw_025(v, kz, alpha)
    elif inv_flag == 3:
        return love_phase_powerlaw(v, kz, alpha)
    elif inv_flag == 4:
        return rayleigh_group_homogeneous(v, kz, alpha)
    elif inv_flag == 5:
        return rayleigh_phase_powerlaw_030(v, kz, alpha)
    elif inv_flag == 6:
        return rayleigh_group_powerlaw_025(v, kz, alpha)
    elif inv_flag == 7:
        return love_group_powerlaw(v, kz, alpha)
    elif inv_flag == 8:
        return rayleigh_group_powerlaw_030(v, kz, alpha)
    else:
        raise ValueError(f"Invalid inv_flag: {inv_flag}")

# --------------------------------------------------------------------------------------------------

# -------------------------------------------------------
# Function for Xia interpolation method to initialize model
# -------------------------------------------------------

import numpy as np
from scipy import interpolate

def compute_xia_interpolation(c, fr, xcof, z_mid, empirical_ratio=0.88):
    """
    Perform Xia-type interpolation and extrapolation for estimating apparent 
    shear velocity structure at depth from surface dispersion data.

    Parameters
    ----------
    c : np.ndarray
        Observed Rayleigh wave phase velocities (1D array of length Nf).
    fr : np.ndarray
        Corresponding frequencies (Hz) for each phase velocity (1D array of length Nf).
    xcof : float
        Scaling coefficient for estimating pseudo-depth as c / fr.
    z_mid : np.ndarray
        Midpoints of the depth layers used for inversion (1D array of length Nz).

    empirical_ratio : float, optional
        Empirically determined ratio of shear wave velocity to Rayleigh phase 
        velocity (default is 0.88, typical for soft sediments).

    Returns
    -------
    xia_int : np.ndarray
        Interpolated shear velocity profile at depth midpoints (`z_mid`), extrapolated
        to deep and shallow ends based on empirical trends.

    Notes
    -----
    This follows the approach proposed in Xia et al. (1999, 2002), where 
    dispersion observations are transformed to approximate depth-velocity 
    structure using scaling and extrapolation.

    The extrapolation ensures stability near the surface and at depth, 
    beyond the nominal resolution of the dispersion data.
    """
    # Step 1: Estimate pseudo-depth from dispersion data
    pseudo_depth = xcof * c / fr

    # Step 2: Fit log-log and linear relationships to guide extrapolation
    aap, bbp = np.polyfit(np.log(pseudo_depth), np.log(c / empirical_ratio), 1)
    aal, bbl = np.polyfit(pseudo_depth, c / empirical_ratio, 1)

    # Step 3: Estimate extrapolated shear velocity at bottom (czro)
    czro = max(0, c[-1] / 0.88 - np.exp(bbp) * aap * ((xcof * c[-1] / fr[-1]) ** aap))

    # Step 4: Estimate extrapolated shear velocity at top (cdeep)
    cdeep = max(0, c[0] / 0.88 - np.exp(bbp) * aap * ((xcof * c[0] / fr[0]) ** (aap - 1)) * (xcof * c[0] / fr[0] - np.max(z_mid)))

    # Step 5: Construct interpolation input arrays
    xiatrap = np.concatenate(([cdeep], c / empirical_ratio, [czro]))
    xp = np.concatenate(([np.max(z_mid)], pseudo_depth, [0]))

    # Step 6: Create interpolator and evaluate at layer midpoints
    f_interp = interpolate.interp1d(
        xp,
        xiatrap,
        kind='linear',
        fill_value='extrapolate',
        assume_sorted=False
    )
    xia_int = f_interp(z_mid)
    z_mid_mat = np.abs(z_mid[:, None] - z_mid[None, :])

    return xia_int, z_mid_mat

# # ---------------------------------------------------------------------------------------
# # Function for damped linear inversion (Tarantola and Valette, 1982)
# # ---------------------------------------------------------------------------------------
# # File: damped_inversion.py

# class DampedInversion:
#     def __init__(
#         self, vsv_init, vpvsratio, U_data, U_data_errs, fks, modn, vflg,
#         rhov, h, vpfv, rhofv, hfv, pratioflag,
#         msigmaf, lsmth, chilo, chihi, nupds, hs, hss
#     ):
#         """
#         Initialize inversion parameters and preallocate storage.
#         """
#         self.vsv = vsv_init.copy()
#         self.vpvsratio = vpvsratio
#         self.U_data = U_data
#         self.U_data_errs = U_data_errs
#         self.fks = fks
#         self.modn = modn
#         self.vflg = vflg
#         self.rhov = rhov
#         self.h = h
#         self.vpfv = vpfv
#         self.rhofv = rhofv
#         self.hfv = hfv
#         self.pratioflag = pratioflag

#         self.msigmaf = msigmaf
#         self.lsmth = lsmth
#         self.chilo = chilo
#         self.chihi = chihi
#         self.nupds = nupds

#         self.hs = hs
#         self.hss = hss
#         self.Nn = len(h)
#         self.Nnf = len(hfv)
#         self.Nf = len(fks)

#         # History arrays
#         self.vsv_update = np.zeros((nupds + 2, self.Nn))
#         self.chisqurd = np.zeros(nupds + 2)
#         self.rmserror = np.zeros(nupds + 2)
#         self.Nfrv = np.zeros(nupds + 2, dtype=int)

#     def run(self):
#         """
#         Perform iterative damped least-squares inversion.
#         Returns:
#             - Final model history
#             - Misfit history
#             - Posterior statistics
#         """
#         # Initial setup
#         vpv = self.vpvsratio * self.vsv if self.pratioflag == 1 else None
#         U, snsmf_vstot, _ = raylee_sensitivity(
#             self.Nn, self.vsv, vpv, self.rhov, self.fks, self.h,
#             self.modn, self.vflg, self.Nnf, self.vpfv, self.rhofv, self.hfv, self.pratioflag
#         )

#         Ur, U_datar, fksr, fksri, modnr, vflgr, snsmf_vstotr = check_nans(
#             U, self.U_data, self.fks, self.modn, self.vflg, snsmf_vstot
#         )

#         # Store filtered quantities
#         Nfr = len(fksr)
#         mcm, mcmisr, dcm, dcmisr = update_covariances(self.U_data_errs, fksri, self.msigmaf, self.hs, self.lsmth)

#         # First inversion step
#         dvs = linvers(U_datar, Ur, snsmf_vstot, mcmisr, dcmisr, self.Nn, self.vsv, self.vsv)
#         self.vsv += dvs
#         vpv = self.vpvsratio * self.vsv if self.pratioflag == 1 else vpv

#         # Forward model with updated Vs
#         U, snsmf_vstot, _ = raylee_sensitivity(
#             self.Nn, self.vsv, vpv, self.rhov, self.fks, self.h,
#             self.modn, self.vflg, self.Nnf, self.vpfv, self.rhofv, self.hfv, self.pratioflag
#         )

#         Ur, U_datar, fksr, fksri, modnr, vflgr, snsmf_vstotr = check_nans(
#             U, self.U_data, self.fks, self.modn, self.vflg, snsmf_vstot
#         )

#         if len(fksr) != Nfr:
#             Nfr = len(fksr)
#             mcm, mcmisr, dcm, dcmisr = update_covariances(self.U_data_errs, fksri, self.msigmaf, self.hs, self.lsmth)

#         rmserrorp, chisqurdp = compute_misfit(Ur, U_datar, self.U_data_errs, fksri)
#         self.vsv_update[1, :] = self.vsv
#         self.rmserror[2] = rmserrorp
#         self.chisqurd[2] = chisqurdp
#         self.Nfrv[1] = Nfr

#         # Call internal loop
#         return self._iterate_updates(
#             U_datar, Ur, snsmf_vstot, mcm, mcmisr, dcm, dcmisr,
#             snsmf_vstotr, Nfr, fksr, fksri, modnr, vflgr, vpv
#         )

#     def _iterate_updates(self, U_datar, Ur, snsmf_vstot, mcm, mcmisr, dcm, dcmisr,
#                          snsmf_vstotr, Nfr, fksr, fksri, modnr, vflgr, vpv):
#         """
#         Perform iterative inversion with line search if needed.
#         """
#         nupdat = 1

#         while self.chisqurd[nupdat + 1] / Nfr > self.chihi and nupdat < self.nupds:
#             dvs = linvers(U_datar, Ur, snsmf_vstot, mcmisr, dcmisr, self.Nn, self.vsv, self.vsv_update[nupdat])
#             self.vsv += dvs
#             vpv = self.vpvsratio * self.vsv if self.pratioflag == 1 else vpv

#             U, snsmf_vstot, _ = raylee_sensitivity(
#                 self.Nn, self.vsv, vpv, self.rhov, fksr, self.h,
#                 modnr, vflgr, self.Nnf, self.vpfv, self.rhofv, self.hfv, self.pratioflag
#             )
#             Ur, _, _, _, _, _, snsmf_vstotr = check_nans(U, U_datar, fksr, modnr, vflgr, snsmf_vstot)

#             rmserrorp, chisqurdp = compute_misfit(Ur, U_datar, self.U_data_errs, fksri)
#             self.vsv_update[nupdat + 1] = self.vsv
#             self.rmserror[nupdat + 2] = rmserrorp
#             self.chisqurd[nupdat + 2] = chisqurdp
#             self.Nfrv[nupdat + 1] = Nfr

#             nupdat += 1
#             check_model_validity(self.vsv, vpv)

#         # Final covariance and resolution
#         G = snsmf_vstotr.T
#         Cd_inv = np.linalg.inv(dcm)
#         Cm_inv = np.linalg.inv(mcm)
#         C_post = np.linalg.inv(G.T @ Cd_inv @ G + Cm_inv)
#         R_model = C_post @ G.T @ Cd_inv @ G
#         R_data = G @ C_post @ G.T @ Cd_inv

#         print_convergence_summary(Nfr, self.Nf, self.chisqurd[nupdat + 1], self.chilo, self.chihi, nupdat, dvs)

#         return {
#         "vsv_guess": self.vsv_init,
#         "vsv_final": self.vsv,
#         "U_initial": self.U_guess,
#         "U_final": self.U,
#         "U_data": self.U_data,
#         "U_data_errs": self.U_data_errs,
#         "fks": self.fks,
#         "hss": self.hss,
#         "vs_posterior_std": self.vs_post_std,
#         "G": self.snsmf_vstot,  # sensitivity
#         "Cm": self.Cm,
#         "C_post": self.C_post,
#         "R_model": self.R_model,
#         "R_data": self.R_data,
#         "Nfr": self.Nfr,
#     }


