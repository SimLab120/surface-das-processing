import numpy as np
from scipy.linalg import sqrtm

# -------------------------------------------------------------------------------------
# Function to check for NaN values in Rayleigh wave inversion results
# -------------------------------------------------------------------------------------

def check_nans(U, U_data, fks, modn, vflg, snsmf_vstot):
    """
    Check for NaN values in the results of Rayleigh wave inversion and filter out invalid entries.

    Parameters:
    U (np.ndarray): Phase velocity array.
    U_data (np.ndarray): Data array corresponding to phase velocities.
    fks (np.ndarray): Frequency array.
    modn (np.ndarray): Model number array.
    vflg (np.ndarray): Flag indicating valid models.
    snsmf_vstot (np.ndarray): Sensitivity matrix.

    Returns:
    tuple: Cleaned arrays without NaN values.
    """
    mask = ~np.isnan(U) & ~np.isnan(U_data)
    
    Ur = U[mask]
    U_datar = U_data[mask]
    fksr = fks[mask]
    fksri = np.where(mask)[0]  # Indices of valid entries
    modnr = modn[mask]
    vflgr = vflg[mask]
    snsmf_vstotr = snsmf_vstot[:, mask]

    return Ur, U_datar, fksr, fksri, modnr, vflgr, snsmf_vstotr


# -------------------------------------------------------------------------------------
# Function to build an exponential layer model for Rayleigh or Love wave inversion
# -------------------------------------------------------------------------------------

def build_exponential_layer_model(c, fr, xcof, lrho, zminfact, zmaxfact):
    """
    Construct an exponentially spaced layered model for Rayleigh or Love wave inversion.

    This function creates a 1D layered Earth model where layer thickness increases
    exponentially with depth. The total depth and resolution are determined based on
    the minimum and maximum penetration depths of the input dispersion data.

    Parameters
    ----------
    c : np.ndarray
        Array of Rayleigh (or Love) wave phase velocities [m/s].
    fr : np.ndarray
        Array of corresponding frequencies [Hz].
    xcof : float, optional
        Scaling coefficient for converting wavelength to depth. 
        Typically between 0.3–0.5 for surface waves.
    lrho : float, optional
        Controls the number of layers via logarithmic spacing (higher = more layers).
    zminfact : float, optional
        Multiplier for the shallowest penetration depth to set minimum model depth.
    zmaxfact : float, optional
        Multiplier for the deepest penetration depth to set maximum model depth.

    Returns
    -------
    thks : np.ndarray
        Layer thicknesses [m] (length N).
    z : np.ndarray
        Layer boundary depths [m] (length N+2, including 0 and inf).
    
    Notes
    -----
    This approach is based on the method used in:
    Haney, M. M., & Tsai, V. C. (2017). Perturbational and nonperturbational inversion 
    of Rayleigh-wave velocities. *Geophysics*, 82(3), F15–F28.
    """

    # Estimate penetration depth range from wavelength = c / fr
    zmin = np.min(xcof * (c / fr)) * zminfact
    zmax = np.max(xcof * (c / fr)) * zmaxfact

    # Estimate number of exponential layers needed
    Nmax = int((xcof * lrho) * np.log(zmax / zmin))

    # Generate exponentially increasing layer boundaries
    exp_boundaries = zmin * np.exp(np.arange(0, Nmax + 1) / (xcof * lrho))

    # Create thickness array from boundaries, including 0 and zmax
    thkso = np.diff(np.concatenate(([0], exp_boundaries, [zmax])))

    # Repeat shallowest layer to reach zmin
    num_shallow = int(np.ceil(zmin / thkso[1]))
    thks = np.concatenate(([thkso[1]] * num_shallow, thkso[1:]))

    # Adjust first layer so total matches zmin exactly
    thks[0] += zmin - thkso[1] * num_shallow

    # Final layer boundaries (0 to sum(thks), plus ∞)
    z = np.concatenate(([0], np.cumsum(thks), [np.inf]))

    return thks, z


# -----------------------------------------------------------------------------------------


def update_covariances(U_data_errs, fksri, msigmaf, hs, lsmth):
    """
    Compute model and data covariance matrices and their inverse square roots.

    Parameters:
    - U_data_errs: Array of uncertainty in observed dispersion
    - fksri: Filtered index set of usable frequencies
    - msigmaf: Scalar factor for model standard deviation
    - hs: Mid-depth vector of layers
    - lsmth: Smoothing scale (in meters)

    Returns:
    - mcm: Model covariance matrix
    - mcmisr: Inverse sqrt of model covariance
    - dcm: Data covariance matrix
    - dcmisr: Inverse sqrt of data covariance
    """
    msigma = np.mean(U_data_errs[fksri]) * msigmaf
    mcm = (msigma ** 2) * np.exp(-np.abs(hs[:, None] - hs[None, :]) / lsmth)
    mcmisr = sqrtm(np.linalg.inv(mcm))
    dcm = np.diag(U_data_errs[fksri] ** 2)
    dcmisr = np.diag(1. / U_data_errs[fksri])
    return mcm, mcmisr, dcm, dcmisr


def check_model_validity(vsv, vpv):
    """
    Ensure physical validity of the current model.

    Raises error if:
    - Shear velocity (Vs) is non-positive
    - Poisson ratio is outside physical range [-1, 0.5]
    """
    if np.any(vsv <= 0):
        raise ValueError("Negative Vs values encountered.")

    pratio = (vpv**2 - 2 * vsv**2) / (2 * (vpv**2 - vsv**2))
    if np.any((pratio <= -1) | (pratio >= 0.5)):
        raise ValueError("Unrealistic Poisson's ratio values.")


def compute_misfit(U_model, U_obs, U_errs, fksri):
    """
    Compute RMS and chi-squared misfit between observed and modeled dispersion.

    Returns:
    - rmserror: Root-mean-square error
    - chisqurd: Chi-squared misfit
    """
    residual = U_model - U_obs
    rmserror = np.sqrt(np.mean(residual ** 2))
    chisqurd = np.sum((residual / U_errs[fksri]) ** 2)
    return rmserror, chisqurd


def print_convergence_summary(Nfr, Nf, chisq, chilo, chihi, nupdat, dvs):
    """
    Print summary of final inversion step.
    """
    print(f"Number of frequencies used: {Nfr} out of {Nf}")
    print(f"Chi-squared per freq (iteration {nupdat}): {chisq:.3f}")
    print(f"Max |ΔVs| in last update: {np.max(np.abs(dvs)):.4f}")
    if chisq / Nfr < chilo:
        print("WARNING: Inversion did not converge to stopping criterion and overfitted data. Increase number of reduction steps.")
    elif chisq / Nfr > chihi:
        print("WARNING: Inversion did not converge to stopping criterion and underfitted data. Increase the number of updates.")

    
