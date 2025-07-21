# raylee_sensitivity.py
# Ported from Haney & Tsai (2017), raylee_sensitivity.m
# Computes Rayleigh wave phase and group velocity sensitivity kernels

import numpy as np
from dix_inversion.raylee_lysmer import raylee_lysmer  # Make sure this is implemented
# If not available yet, you can temporarily replace this with a dummy placeholder

def raylee_sensitivity(Nn, vsv, vpv, rhov, fks_orig, hv, modnv_orig, vflg,
                       Nnf, vpfv, rhofv, hfv, pratioflag):
    """
    Compute Rayleigh wave sensitivity kernels for phase and group velocities.

    Parameters
    ----------
    Nn : int
        Number of elements in the solid model.
    vsv, vpv, rhov : array_like
        Shear velocity, P-wave velocity, and density in solid (length Nn).
    fks_orig : array_like
        Frequency vector (Hz).
    hv : array_like
        Grid spacing in solid model (m).
    modnv_orig : array_like
        Mode numbers (1 = fundamental).
    vflg : array_like
        0 = phase kernel, 1 = group kernel.
    Nnf : int
        Number of elements in fluid layer.
    vpfv, rhofv, hfv : array_like
        Fluid P-wave velocity, density, and grid spacing.
    pratioflag : int
        0 = fixed Vp, 1 = fixed Poisson’s ratio.

    Returns
    -------
    Uf : ndarray
        Modeled phase/group velocities for each frequency in fks_orig.
    snsmf_vstotf : ndarray
        Sensitivity kernels (shear velocity, absolute).
    snsmf_htotf : ndarray
        Sensitivity kernels (thickness, absolute).
    """

    # ----------------------------
    # Frequency vector augmentation
    # ----------------------------

    fks = []
    modnv = []

    for i, (f, mode, flag) in enumerate(zip(fks_orig, modnv_orig, vflg)):
        if flag == 1:
            fks.extend([f * 0.999, f, f * 1.001])
            modnv.extend([mode, mode, mode])
        else:
            fks.append(f)
            modnv.append(mode)

    fks = np.array(fks)
    modnv = np.array(modnv)

    # ----------------------------------------------
    # Forward modeling for each frequency (main loop)
    # ----------------------------------------------

    vp = np.zeros(len(fks))
    U = np.zeros(len(fks))
    xm = np.zeros((2 * Nn, len(fks)))

    for i, (f, mode) in enumerate(zip(fks, modnv)):
        _, vpk, vgk, x = raylee_lysmer(Nn, vsv, vpv, rhov, f, hv, mode,
                                       Nnf, vpfv, rhofv, hfv)
        vp[i] = vpk
        U[i] = vgk
        xm[:, i] = x

    snsmf = np.zeros((Nn, len(fks)))
    snsmflam = np.zeros((Nn, len(fks)))
    snsmfrho = np.zeros((Nn, len(fks)))
    snsmfh = np.zeros((Nn, len(fks)))

    # ----------------------------------------------
    # Sensitivity kernel calculations
    # ----------------------------------------------
    for ii in range(Nn - 1):
        h = hv[ii]
        for count, f in enumerate(fks):
            # Extract wavefunction values
            x = xm[:, count]

            # === DENSITY SENSITIVITY ===
            snsmfrho[ii, count] = - (vp[count] / (2 * U[count])) * (
                x[2*ii]   * (h/2) * x[2*ii]   +
                x[2*ii+1] * (h/2) * x[2*ii+1] +
                x[2*ii+2] * (h/2) * x[2*ii+2] +
                x[2*ii+3] * (h/2) * x[2*ii+3]
            ) * rhov[ii]

            # === LAMBDA SENSITIVITY ===
            lam_factor = rhov[ii] * (vpv[ii]**2 - 2 * vsv[ii]**2)

            # k^2 term
            snsmflam[ii, count] = (1 / (2 * vp[count] * U[count])) * (
                x[2*ii]   * (h/3) * x[2*ii]   +
                x[2*ii+2] * (h/3) * x[2*ii+2] +
                x[2*ii+2] * (h/6) * x[2*ii]   +
                x[2*ii]   * (h/6) * x[2*ii+2]
            ) * lam_factor

            # k^1 term
            omega = 2 * np.pi * f
            snsmflam[ii, count] += (1 / (2 * omega * U[count])) * (
                x[2*ii]   * ( 0.5) * x[2*ii+1] +
                x[2*ii]   * (-0.5) * x[2*ii+3] +
                x[2*ii+1] * ( 0.5) * x[2*ii]   +
                x[2*ii+1] * ( 0.5) * x[2*ii+2] +
                x[2*ii+2] * ( 0.5) * x[2*ii+1] +
                x[2*ii+2] * (-0.5) * x[2*ii+3] +
                x[2*ii+3] * (-0.5) * x[2*ii]   +
                x[2*ii+3] * (-0.5) * x[2*ii+2]
            ) * lam_factor

            # k^0 term
            snsmflam[ii, count] += (vp[count] / (2 * omega**2 * U[count])) * (
                x[2*ii+1] * (1/h) * x[2*ii+1] +
                x[2*ii+3] * (1/h) * x[2*ii+3] +
                x[2*ii+1] * (-1/h) * x[2*ii+3] +
                x[2*ii+3] * (-1/h) * x[2*ii+1]
            ) * lam_factor

            # === MU (SHEAR) SENSITIVITY ===
            mu_factor = rhov[ii] * (vsv[ii]**2)

            # k^2 term
            snsmf[ii, count] = (1 / (2 * vp[count] * U[count])) * (
                x[2*ii]   * (2*h/3) * x[2*ii]   +
                x[2*ii+2] * (2*h/3) * x[2*ii+2] +
                x[2*ii+2] * (h/3)   * x[2*ii]   +
                x[2*ii]   * (h/3)   * x[2*ii+2] +
                x[2*ii+1] * (h/3)   * x[2*ii+1] +
                x[2*ii+3] * (h/3)   * x[2*ii+3] +
                x[2*ii+1] * (h/6)   * x[2*ii+3] +
                x[2*ii+3] * (h/6)   * x[2*ii+1]
            ) * mu_factor

            # k^1 term
            snsmf[ii, count] += (1 / (2 * omega * U[count])) * (
                x[2*ii]   * (-0.5) * x[2*ii+1] +
                x[2*ii]   * (-0.5) * x[2*ii+3] +
                x[2*ii+1] * (-0.5) * x[2*ii]   +
                x[2*ii+1] * ( 0.5) * x[2*ii+2] +
                x[2*ii+2] * ( 0.5) * x[2*ii+1] +
                x[2*ii+2] * ( 0.5) * x[2*ii+3] +
                x[2*ii+3] * (-0.5) * x[2*ii]   +
                x[2*ii+3] * ( 0.5) * x[2*ii+2]
            ) * mu_factor

            # k^0 term
            snsmf[ii, count] += (vp[count] / (2 * omega**2 * U[count])) * (
                x[2*ii+1] * (2/h) * x[2*ii+1] +
                x[2*ii+3] * (2/h) * x[2*ii+3] +
                x[2*ii+1] * (-2/h) * x[2*ii+3] +
                x[2*ii+3] * (-2/h) * x[2*ii+1] +
                x[2*ii]   * (1/h) * x[2*ii]   +
                x[2*ii+2] * (1/h) * x[2*ii+2] +
                x[2*ii]   * (-1/h) * x[2*ii+2] +
                x[2*ii+2] * (-1/h) * x[2*ii]
            ) * mu_factor

            # === THICKNESS SENSITIVITY ===
            # omega^2 term
            snsmfh[ii, count] = - (vp[count] / (2 * U[count])) * (
                x[2*ii]   * (rhov[ii]/2) * x[2*ii]   +
                x[2*ii+1] * (rhov[ii]/2) * x[2*ii+1] +
                x[2*ii+2] * (rhov[ii]/2) * x[2*ii+2] +
                x[2*ii+3] * (rhov[ii]/2) * x[2*ii+3]
            ) * h

            pmod = rhov[ii] * (vpv[ii]**2)
            smod = rhov[ii] * (vsv[ii]**2)

            # k^2 term
            snsmfh[ii, count] += (1 / (2 * vp[count] * U[count])) * (
                x[2*ii]   * (pmod/3) * x[2*ii]   +
                x[2*ii+2] * (pmod/3) * x[2*ii+2] +
                x[2*ii+2] * (pmod/6) * x[2*ii]   +
                x[2*ii]   * (pmod/6) * x[2*ii+2] +
                x[2*ii+1] * (smod/3) * x[2*ii+1] +
                x[2*ii+3] * (smod/3) * x[2*ii+3] +
                x[2*ii+1] * (smod/6) * x[2*ii+3] +
                x[2*ii+3] * (smod/6) * x[2*ii+1]
            ) * h

            # k^0 term
            snsmfh[ii, count] += (vp[count] / (2 * omega**2 * U[count])) * (
                x[2*ii+1] * pmod * (-1/h**2) * x[2*ii+1] +
                x[2*ii+3] * pmod * (-1/h**2) * x[2*ii+3] +
                x[2*ii+1] * -pmod * (-1/h**2) * x[2*ii+3] +
                x[2*ii+3] * -pmod * (-1/h**2) * x[2*ii+1] +
                x[2*ii]   * smod * (-1/h**2) * x[2*ii]   +
                x[2*ii+2] * smod * (-1/h**2) * x[2*ii+2] +
                x[2*ii]   * -smod * (-1/h**2) * x[2*ii+2] +
                x[2*ii+2] * -smod * (-1/h**2) * x[2*ii]
            ) * h
    # ----------------------------------------------
    # bottom layer sensitivity
    # ----------------------------------------------
    ii = Nn - 1
    h = hv[ii]

    for count, f in enumerate(fks):
        x = xm[:, count]
        omega = 2 * np.pi * f

        # === DENSITY SENSITIVITY (2 DOFs only) ===
        snsmfrho[ii, count] = - (vp[count] / (2 * U[count])) * (
            x[2*ii]   * (h/2) * x[2*ii] +
            x[2*ii+1] * (h/2) * x[2*ii+1]
        ) * rhov[ii]

        # === LAMBDA SENSITIVITY ===
        lam_factor = rhov[ii] * (vpv[ii]**2 - 2 * vsv[ii]**2)

        # k^2 term
        snsmflam[ii, count] = (1 / (2 * vp[count] * U[count])) * (
            x[2*ii] * (h/3) * x[2*ii]
        ) * lam_factor

        # k^1 term
        snsmflam[ii, count] += (1 / (2 * omega * U[count])) * (
            x[2*ii]   * 0.5 * x[2*ii+1] +
            x[2*ii+1] * 0.5 * x[2*ii]
        ) * lam_factor

        # k^0 term
        snsmflam[ii, count] += (vp[count] / (2 * omega**2 * U[count])) * (
            x[2*ii+1] * (1/h) * x[2*ii+1]
        ) * lam_factor

        # === MU SENSITIVITY ===
        mu_factor = rhov[ii] * (vsv[ii]**2)

        # k^2 term
        snsmf[ii, count] = (1 / (2 * vp[count] * U[count])) * (
            x[2*ii] * (2*h/3) * x[2*ii] +
            x[2*ii+1] * (h/3) * x[2*ii+1]
        ) * mu_factor

        # k^1 term
        snsmf[ii, count] += (1 / (2 * omega * U[count])) * (
            x[2*ii]   * (-0.5) * x[2*ii+1] +
            x[2*ii+1] * (-0.5) * x[2*ii]
        ) * mu_factor

        # k^0 term
        snsmf[ii, count] += (vp[count] / (2 * omega**2 * U[count])) * (
            x[2*ii+1] * (2/h) * x[2*ii+1] +
            x[2*ii]   * (1/h) * x[2*ii]
        ) * mu_factor

        # === THICKNESS SENSITIVITY ===
        # omega^2 term
        snsmfh[ii, count] = - (vp[count] / (2 * U[count])) * (
            x[2*ii]   * (rhov[ii]/2) * x[2*ii] +
            x[2*ii+1] * (rhov[ii]/2) * x[2*ii+1]
        ) * h

        pmod = rhov[ii] * (vpv[ii]**2)
        smod = rhov[ii] * (vsv[ii]**2)

        # k^2 term
        snsmfh[ii, count] += (1 / (2 * vp[count] * U[count])) * (
            x[2*ii]   * (pmod/3) * x[2*ii] +
            x[2*ii+1] * (smod/3) * x[2*ii+1]
        ) * h

        # k^0 term
        snsmfh[ii, count] += (vp[count] / (2 * omega**2 * U[count])) * (
            x[2*ii+1] * pmod * (-1/h**2) * x[2*ii+1] +
            x[2*ii]   * smod * (-1/h**2) * x[2*ii]
        ) * h

    # ---------------------------------------------
    # Step 1: Shear velocity kernel (Vs sensitivity)
    # ---------------------------------------------
    if pratioflag == 0:
        # Vp fixed: use chain rule
        snsmf_vs = 2 * snsmf - 4 * (snsmflam.T @ np.diag((vsv**2) / (vpv**2 - 2 * vsv**2))).T
    elif pratioflag == 1:
        # Poisson's ratio fixed
        snsmf_vs = 2 * (snsmf + snsmflam)
    else:
        raise ValueError("Invalid pratioflag: must be 0 or 1")

    # ---------------------------------------------
    # Step 2: Frequency map vfi
    # Maps original fks to indices in extended fks
    # ---------------------------------------------
    vfi = np.zeros(len(fks_orig), dtype=int)

    if vflg[0] == 0:
        vfi[0] = 0
    else:
        vfi[0] = 1

    for ii in range(1, len(vflg)):
        if vflg[ii] == 1 and vflg[ii - 1] == 0:
            vfi[ii] = vfi[ii - 1] + 2
        elif vflg[ii] == 1 and vflg[ii - 1] == 1:
            vfi[ii] = vfi[ii - 1] + 3
        elif vflg[ii] == 0 and vflg[ii - 1] == 0:
            vfi[ii] = vfi[ii - 1] + 1
        else:
            vfi[ii] = vfi[ii - 1] + 2

    # ---------------------------------------------
    # Step 3: Output kernels and velocity vector
    # ---------------------------------------------
    Uf = np.zeros(len(fks_orig))
    snsmf_vstotf = np.zeros((Nn, len(fks_orig)))
    snsmf_htotf = np.zeros((Nn, len(fks_orig)))

    for count in range(len(fks_orig)):
        i_vfi = vfi[count]

        if vflg[count] == 0:
            # Phase kernel
            Uf[count] = vp[i_vfi]
            snsmf_vstotf[:, count] = vp[i_vfi] * (snsmf_vs[:, i_vfi] / vsv)
            snsmf_htotf[:, count] = vp[i_vfi] * (snsmfh[:, i_vfi] / hv)

        else:
            # Group kernel using Rodi et al. 1975 finite-diff
            df = fks[i_vfi + 1] - fks[i_vfi - 1]
            if np.isnan(U[i_vfi + 1]) or np.isnan(U[i_vfi - 1]):
                Uf[count] = np.nan
            else:
                Uf[count] = U[i_vfi]

            dVs = (snsmf_vs[:, i_vfi + 1] - snsmf_vs[:, i_vfi - 1])
            snsmf_vstot = snsmf_vs[:, i_vfi] + \
                          (U[i_vfi] / vp[i_vfi]) * omega * dVs / df
            snsmf_vstotf[:, count] = vp[i_vfi] * (snsmf_vstot / vsv)

            dH = (snsmfh[:, i_vfi + 1] - snsmfh[:, i_vfi - 1])
            snsmf_htot = snsmfh[:, i_vfi] + \
                         (U[i_vfi] / vp[i_vfi]) * omega * dH / df
            snsmf_htotf[:, count] = U[i_vfi] * (snsmf_htot / hv)

    return Uf, snsmf_vstotf, snsmf_htotf