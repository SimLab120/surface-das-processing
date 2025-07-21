import numpy as np
import scipy.linalg as la
from scipy.interpolate import interp1d
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import time
import dix_inversion as dix
import os

# Start timing
t_start = time.time()

# Helper to construct full path
def full_path(fname):
    return os.path.join('initial_model_output', fname)

# Helper to read a vector from file
def read_vector_from_file(filepath, N):
    with open(filepath, 'r') as f:
        return np.array([float(f.readline()) for _ in range(N)])

# Read parameters
inp = np.loadtxt(full_path('input_params.txt'), comments='%')
pratioflag = int(inp[0])
lsmth = inp[1]
msigmaf = inp[2]
nupds = int(inp[3])
Nf = int(inp[4])
Nn = int(inp[5])
Nnf = int(inp[6])
chilo = inp[7]
chihi = inp[8]


# Read solid and fluid grid spacing
h = read_vector_from_file(full_path('grid_values_solid.txt'), Nn)
hfv = read_vector_from_file(full_path('grid_values_fluid.txt'), Nnf)

# Frequency and data files
fks = read_vector_from_file(full_path('frequency_values.txt'), Nf)
U_data = read_vector_from_file(full_path('velocity_values.txt'), Nf)
U_data_errs = read_vector_from_file(full_path('velocity_values_errs.txt'), Nf)
modn = read_vector_from_file(full_path('mode_values.txt'), Nf).astype(int)
vflg = read_vector_from_file(full_path('vtype_values.txt'), Nf).astype(int)

# Initial Vs model
vsv = read_vector_from_file(full_path('vs_init.txt'), Nn)

# Initial Vp model
if pratioflag == 0:
    vpv = read_vector_from_file(full_path('vp_init.txt'), Nn)
elif pratioflag == 1:
    with open(full_path('vp_init.txt'), 'r') as f:
        vpvsratio = float(f.readline())
    vpv = vpvsratio * vsv
else:
    raise ValueError("Invalid value for pratioflag")

# Initial density
rhov = read_vector_from_file(full_path('rho_init.txt'), Nn)

# Fluid Vp and density
vpfv = read_vector_from_file(full_path('vpf.txt'), Nnf)
rhofv = read_vector_from_file(full_path('rhof.txt'), Nnf)

# Initialize model update array
vsv_update = np.zeros((nupds + 1, Nn))

# Node depths
hs = np.zeros(Nn)
hs[0] = 0
for i in range(1, Nn):
    hs[i] = np.sum(h[:i])

# Element center depths
hss = np.zeros(Nn)
hss[0] = h[0] / 2
for i in range(1, Nn):
    hss[i] = np.sum(h[:i]) + h[i] / 2

# ========== Sanity Checks ==========
if np.any(rhov <= 0):
    raise ValueError("Negative density values exist in initial guess")
if np.any(vsv <= 0):
    raise ValueError("Negative shear velocity values exist in initial guess")
pratio = (vpv**2 - 2 * vsv**2) / (2 * (vpv**2 - vsv**2))
if np.any((pratio <= -1) | (pratio >= 0.5)):
    raise ValueError("Impossible Poisson ratio values exist in initial guess")
if np.any(rhofv <= 0):
    raise ValueError("Negative density values exist in fluid layer")


# ------------------------------------------------------
# Compute sensitivity kernel using the initial Vs model
# ------------------------------------------------------
U, snsmf_vstot, _ = dix.raylee_sensitivity(
    Nn, vsv, vpv, rhov, fks, h, modn, vflg,
    Nnf, vpfv, rhofv, hfv, pratioflag
)

# ------------------------------------------------------------------------
# Identify frequencies where both the predicted and observed data are valid
# This removes any NaNs due to ill-posed sensitivity or bad measurements
# ------------------------------------------------------------------------
Ur, U_datar, fksr, fksri, modnr, vflgr, snsmf_vstotr = dix.check_nans(
    U, U_data, fks, modn, vflg, snsmf_vstot
)
Nfr = len(fksr)  # Number of "retained" (non-NaN) frequencies

# ------------------------------------------------------
# Save current model and prediction as baseline "guess"
# ------------------------------------------------------
vsv_guess = vsv.copy()
U_guess = Ur.copy()
fksr_guess = fksr.copy()

# ------------------------------------------------------------------------
# A priori model covariance matrix (MCM):
# Exponential spatial correlation based on depth differences
# Equivalent to: mcm(i,j) = (σ²) * exp(-|zi - zj| / lsmth)
# ------------------------------------------------------------------------
msigma = np.mean(U_data_errs[fksri]) * msigmaf
depth_matrix_diff = np.abs(np.subtract.outer(hs, hs))  # [Nn x Nn] matrix
mcm = (msigma ** 2) * np.exp(-depth_matrix_diff / lsmth)

# Inverse square root of model covariance (for regularization)
mcmisr = la.sqrtm(la.inv(mcm))  # [Nn x Nn]

# ------------------------------------------------------------------------
# A priori data covariance (DCM) is diagonal (uncorrelated errors)
# DCM inverse square root = diag(1 / σi)
# ------------------------------------------------------------------------
dcm = np.diag(U_data_errs[fksri] ** 2)     # [Nfr x Nfr]
dcmisr = np.diag(1.0 / U_data_errs[fksri]) # [Nfr x Nfr]

# ------------------------------------------------------
# RMS error (normalized residual) of the initial guess
# ------------------------------------------------------
rmserror = np.zeros(nupds + 2)
chisqurd = np.zeros(nupds + 2)
Nfrv = np.zeros(nupds + 2, dtype=int)

rmserror[0] = np.sqrt(np.mean((U_guess - U_datar) ** 2))

# Chi-squared = (residual^T * DCM⁻¹ * DCM⁻¹ * residual)
# Same as: residual.T @ (DCMisr @ DCMisr) @ residual
residual = U_guess - U_datar
chisqurd[0] = residual @ dcmisr @ dcmisr @ residual
Nfrv[0] = Nfr

# ------------------------------------------------------
# Check if initial model already satisfies the chi² bounds
# If so, inversion is either unnecessary or ill-posed
# ------------------------------------------------------
reduced_chi2 = chisqurd[0] / Nfr
if reduced_chi2 < chilo:
    raise ValueError('Initial model fits data to less than 1 chi-squared')
elif reduced_chi2 < chihi:
    raise ValueError('Initial model fits data within acceptable chi-squared window')
else:
    pass  # Proceed with inversion
# ------------------------------------------------------

# ------------------------------------------------------
# Inversion Begins
# ------------------------------------------------------

# Initial damped linear inversion (Tarantola and Valette, 1982)
dvs = dix.linvers(U_datar, Ur, snsmf_vstotr, mcmisr, dcmisr, Nn, vsv, vsv_guess)

vsv = dvs + vsv_guess
if pratioflag == 1:
    vpv = vpvsratio * vsv

# Compute new sensitivity kernel
U, snsmf_vstot, _ = dix.raylee_sensitivity(Nn, vsv, vpv, rhov, fksr, h, modnr, vflgr,
                                    Nnf, vpfv, rhofv, hfv, pratioflag)
U, U_datar, fksr, fksri, modnr, vflgr, snsmf_vstot = dix.check_nans(
    U, U_datar, fksr, modnr, vflgr, snsmf_vstot)

# Recompute covariances if needed
if len(fksr) != Nfr:
    Nfr = len(fksr)
    msigma = np.mean(U_data_errs[fksri]) * msigmaf
    mcm = (msigma ** 2) * np.exp(-np.abs(hs[:, None] - hs[None, :]) / lsmth)
    mcmisr = sqrtm(np.linalg.inv(mcm))
    dcm = np.diag(U_data_errs[fksri] ** 2)
    dcmisr = np.diag(1. / U_data_errs[fksri])

rmserrorp = np.sqrt(np.mean((U - U_datar) ** 2))
residual = U - U_datar
chisqurdp = np.sum((residual / U_data_errs[fksri]) ** 2)

nreds = 0
while ((chisqurdp >= chisqurd[0] and nreds < nupds) or
       ((chisqurdp / Nfr) < 1 and nreds < nupds)):
    nreds += 1
    dvs /= 2
    vsv = vsv_guess + dvs
    if pratioflag == 1:
        vpv = vpvsratio * vsv

    # Forward modeling with updated model
    U, snsmf_vstot, _ = dix.raylee_sensitivity(Nn, vsv, vpv, rhov, fksr, h, modnr, vflgr,
                                           Nnf, vpfv, rhofv, hfv, pratioflag)
    U, U_datar, fksr, fksri, modnr, vflgr, snsmf_vstot = dix.check_nans(
        U, U_datar, fksr, modnr, vflgr, snsmf_vstot)

    # If number of data changed, recompute covariances
    if len(fksr) != Nfr:
        Nfr = len(fksr)
        msigma = np.mean(U_data_errs[fksri]) * msigmaf
        mcm = (msigma ** 2) * np.exp(-np.abs(hs[:, None] - hs[None, :]) / lsmth)
        mcmisr = sqrtm(np.linalg.inv(mcm))
        dcm = np.diag(U_data_errs[fksri] ** 2)
        dcmisr = np.diag(1. / U_data_errs[fksri])

    # Compute RMS and chi^2 for this potential update
    rmserrorp = np.sqrt(np.mean((U - U_datar) ** 2))
    residual = U - U_datar
    chisqurdp = np.sum((residual / U_data_errs[fksri]) ** 2)


# Shear velocity must be greater than zero
if (np.any(vsv <= 0)):
    raise ValueError("Negative shear velocity values encountered in inversion")
# Poisson's ratio between two bounds
pratio = (vpv**2 - 2 * vsv**2) / (2 * (vpv**2 - vsv**2))
if (np.any((pratio <= -1) | (pratio >= 0.5))):
    raise ValueError("Impossible Poisson ratio values encountered in inversion")

# the upated model, print the number to update on the screen
nupdat = 1
vsv_update[nupdat, :] = vsv

rmserror[nupdat + 1] = rmserrorp
chisqurd[nupdat + 1] = chisqurdp

# now full modeling
U, snsmf_vstot, _ = dix.raylee_sensitivity(Nn, vsv, vpv, rhov, fks, h, modn, vflg,
                                       Nnf, vpfv, rhofv, hfv, pratioflag)
# Check for NaNs in the full model
Ur, U_datar, fksr, fksri, modnr, vflgr, snsmf_vstotr = dix.check_nans(
    U, U_data, fks, modn, vflg, snsmf_vstot)

# If number of Nans changed, recompute data and model covariances
if len(fksr) != Nfr:
    Nfr = len(fksr)
    msigma = np.mean(U_data_errs[fksri]) * msigmaf
    mcm = (msigma ** 2) * np.exp(-np.abs(hs[:, None] - hs[None, :]) / lsmth)
    mcmisr = sqrtm(np.linalg.inv(mcm))
    dcm = np.diag(U_data_errs[fksri] ** 2)
    dcmisr = np.diag(1. / U_data_errs[fksri])

# Compute RMS and chi^2 for the model
rmserrorp = np.sqrt(np.mean((Ur - U_datar) ** 2))
residual = Ur - U_datar
chisqurdp = np.sum((residual / U_data_errs[fksri]) ** 2)

# rms of the update
rmserror[nupdat + 1] = rmserrorp
chisqurd[nupdat + 1] = chisqurdp


# -----------------------
# Now an Iterative Inversion
# -----------------------
while chisqurdp/ Nfr > chihi and nupdat < nupds:

    Nfrv[nupdat] = Nfr

    dvs = dix.linvers(U_datar, Ur, snsmf_vstotr, mcmisr, dcmisr, Nn, vsv, vsv_guess)

    # add to initial model
    vsv = dvs + vsv_guess

    if pratioflag == 1:
        vpv = vpvsratio * vsv

    # call sensitivity kernel
    U, snsmf_vstot, _ = dix.raylee_sensitivity(Nn, vsv, vpv, rhov, fksr, h, modnr, vflgr,
                                           Nnf, vpfv, rhofv, hfv, pratioflag)
    Ur, U_datar, fksr, fksri, modnr, vflgr, snsmf_vstotr = dix.check_nans(
        U, U_datar, fksr, modnr, vflgr, snsmf_vstot)
    
    # If number of data changed, recompute covariances
    if len(fksr) != Nfr:
        Nfr = len(fksr)
        msigma = np.mean(U_data_errs[fksri]) * msigmaf
        mcm = (msigma ** 2) * np.exp(-np.abs(hs[:, None] - hs[None, :]) / lsmth)
        mcmisr = sqrtm(np.linalg.inv(mcm))
        dcm = np.diag(U_data_errs[fksri] ** 2)
        dcmisr = np.diag(1. / U_data_errs[fksri])

    # Compute RMS and chi^2 for this potential update
    rmserrorp = np.sqrt(np.mean((Ur - U_datar) ** 2))
    residual = Ur - U_datar
    chisqurdp = np.sum((residual / U_data_errs[fksri]) ** 2)

    print(f"Iteration {nupdat}")
    print("  max(abs(dvs)):", np.max(np.abs(dvs)))
    print("  chisqurdp / Nfr:", chisqurdp / Nfr)
    print("  vsv_guess[:3]:", vsv_guess[:3])
    print("  vsv[:3]:", vsv[:3])


    # A reduced line search if chi^2 is not lower
    nreds = 0
    dvs = vsv - vsv_update[nupdat, :]  # gradient: difference from previous update

    while ((chisqurdp >= 1.01 * chisqurd[nupdat + 1] and nreds < nupds) or
           ((chisqurdp / Nfr) < chilo and nreds < nupds)):
        nreds += 1
        dvs /= 2
        vsv = vsv_update[nupdat, :] + dvs
        if pratioflag == 1:
            vpv = vpvsratio * vsv

        # call sensitivity kernel
        U, snsmf_vstot, _ = dix.raylee_sensitivity(Nn, vsv, vpv, rhov, fksr, h, modnr, vflgr,
                                               Nnf, vpfv, rhofv, hfv, pratioflag)
        U, U_datar, fksr, fksri, modnr, vflgr, snsmf_vstot = dix.check_nans(
            U, U_datar, fksr, modnr, vflgr, snsmf_vstot)
        # If number of data changed, recompute covariances
        if len(fksr) != Nfr:
            Nfr = len(fksr)
            msigma = np.mean(U_data_errs[fksri]) * msigmaf
            mcm = (msigma ** 2) * np.exp(-np.abs(hs[:, None] - hs[None, :]) / lsmth)
            mcmisr = sqrtm(np.linalg.inv(mcm))
            dcm = np.diag(U_data_errs[fksri] ** 2)
            dcmisr = np.diag(1. / U_data_errs[fksri])

        # Compute RMS and chi^2 for this potential update
        rmserrorp = np.sqrt(np.mean((U - U_datar) ** 2))
        residual = U - U_datar
        chisqurdp = np.sum((residual / U_data_errs[fksri]) ** 2)

        print(f"Iteration {nupdat}")
        print("  max(abs(dvs)):", np.max(np.abs(dvs)))
        print("  chisqurdp / Nfr:", chisqurdp / Nfr)
        print("  vsv_guess[:3]:", vsv_guess[:3])
        print("  vsv[:3]:", vsv[:3])


    # Shear velocity must be greater than zero
    if np.any(vsv <= 0):
        raise ValueError("Negative shear velocity values encountered in inversion")
    # Poisson's ratio between two bounds
    pratio = (vpv**2 - 2 * vsv**2) / (2 * (vpv**2 - vsv**2))
    if np.any((pratio <= -1) | (pratio >= 0.5)):
        raise ValueError("Impossible Poisson ratio values encountered in inversion")
    
    print("Line search reductions:", nreds)

    # next update
    Nfrv[nupdat + 1] = Nfr
    nupdat += 1
    vsv_update[nupdat, :] = vsv

    rmserror[nupdat + 1] = rmserrorp
    chisqurd[nupdat + 1] = chisqurdp

    # now full modeling
    U, snsmf_vstot, _ = dix.raylee_sensitivity(Nn, vsv, vpv, rhov, fks, h, modn, vflg,
                                           Nnf, vpfv, rhofv, hfv, pratioflag)
    Ur, U_datar, fksr, fksri, modnr, vflgr, snsmf_vstotr = dix.check_nans(
        U, U_data, fks, modn, vflg, snsmf_vstot)
    # If number of data changed, recompute covariances
    if len(fksr) != Nfr:
        Nfr = len(fksr)
        msigma = np.mean(U_data_errs[fksri]) * msigmaf
        mcm = (msigma ** 2) * np.exp(-np.abs(hs[:, None] - hs[None, :]) / lsmth)
        mcmisr = sqrtm(np.linalg.inv(mcm))
        dcm = np.diag(U_data_errs[fksri] ** 2)
        dcmisr = np.diag(1. / U_data_errs[fksri])
    # Compute RMS and chi^2 for the model   
    rmserrorp = np.sqrt(np.mean((Ur - U_datar) ** 2))
    residual = Ur - U_datar
    chisqurdp = np.sum((residual / U_data_errs[fksri]) ** 2)
    rmserror[nupdat + 1] = rmserrorp
    chisqurd[nupdat + 1] = chisqurdp
    
# print the measurements used in the inversion
print(f"Number of measurements used: {Nfr} out of {Nf} total frequencies")
print(f"{Nfr} of {Nf - np.sum(np.isnan(U_data))} measurements used")

if chisqurd[nupdat + 1]/ Nfr < chilo:
    print("WARNING: Inversion did not converge to stopping criterion and overfitted data. Increase number of reduction steps.")
if chisqurd[nupdat + 1]/ Nfr > chihi:
    print("WARNING: Inversion did not converge to stopping criterion and underfitted data. Increase the number of updates.")

print("U.shape:", U.shape)
print("U_datar.shape:", U_datar.shape)
print("U_data_errs[fksri].shape:", U_data_errs[fksri].shape)

print(U.dtype, U_datar.dtype, U_data_errs.dtype)

print("fksri:", fksri)

print("U[:5] =", U[:5])
print("U_datar[:5] =", U_datar[:5])
print("errs[:5] =", U_data_errs[fksri][:5])

print('Nfrv:', Nfrv)
# ---------------------- Convergence Plot ----------------------
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(np.arange(nupdat + 1), np.array(chisqurd[:nupdat + 1]) / np.array(Nfrv[:nupdat + 1]), '-o', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel(r'$\chi^2$ per frequency')
plt.title('Chi-squared Convergence')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(np.arange(nupdat + 1), rmserror[:nupdat + 1], '-s', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('RMS Error')
plt.title('RMS Error Convergence')
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"Chi-squared per frequency (iteration {nupdat + 1}): {chisqurd[nupdat + 1] / Nfr:.6f}")
# max abs dvs
print(f"Max abs(dvs): {np.max(np.abs(dvs)):.6f}")

# ---------------------- Posterior Covariance and Resolution Matrices ----------------------
depth_limit = 1000  # cutoff depth in meters

G = snsmf_vstotr.T    # Sensitivity matrix: shape [Nf, Nn]
Cd = dcm              # Data covariance: shape [Nf, Nf]
Cm = mcm              # Model covariance: shape [Nn, Nn]
z = hss               # Layer depths (used for plotting)

# Posterior model covariance
Cd_inv = np.linalg.inv(Cd)
Cm_inv = np.linalg.inv(Cm)
C_post = np.linalg.inv(G.T @ Cd_inv @ G + Cm_inv)

# Model resolution matrix
R_model = C_post @ G.T @ Cd_inv @ G

# Data resolution matrix
R_data = G @ C_post @ G.T @ Cd_inv

# Standard deviation (1-sigma uncertainty) for Vs model
vs_post_std = np.sqrt(np.diag(C_post))

# ---------------------- Data Fit Plots ----------------------
# With error bars
plt.figure()
plt.errorbar(fks, U_data, yerr=U_data_errs, fmt='bo', linewidth=2, markersize=6, label='Observed (with error)')
plt.plot(fks, U_guess, 'ro', linewidth=2, markersize=6, label='Initial')
plt.plot(fks, U, 'k-', linewidth=2, label='Final')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Velocity (m/s)')
plt.title('Observed vs Modeled Dispersion')
plt.legend()
plt.grid(True)
plt.gca().tick_params(labelsize=12)
plt.show()

# Without error bars
plt.figure()
plt.plot(fks, U_data, 'bo', linewidth=2, markersize=6, label='Observed')
plt.plot(fks, U_guess, 'ro', linewidth=2, markersize=6, label='Initial')
plt.plot(fks, U, 'k-', linewidth=2, label='Final')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Velocity (m/s)')
plt.title('Observed vs Modeled Dispersion')
plt.legend()
plt.grid(True)
plt.gca().tick_params(labelsize=12)
plt.show()

# Plot settings
depth_limit = 1000  # meters
depth_idx = hss <= depth_limit

# 1. Shear Velocity Model Plot (cut at 1000 m)
plt.figure()
plt.plot(vsv_guess[depth_idx], hss[depth_idx], 'r--o', linewidth=2, label='Initial Model')
plt.plot(vsv_update[nupdat, depth_idx], hss[depth_idx], 'k-o', linewidth=2, label='Final Inverted Model')
plt.gca().invert_yaxis()
plt.xlabel('Vs (m/s)')
plt.ylabel('Depth (m)')
plt.legend()
plt.title('Shear Wave Velocity Model (Top 1000 m)')
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Updated Shear Velocity Plot with Error Bars
plt.figure()
plt.errorbar(vsv_update[nupdat, depth_idx], hss[depth_idx], xerr=vs_post_std[depth_idx], fmt='k-o', linewidth=2, label='Inverted Vs ±1σ')
plt.plot(vsv_guess[depth_idx], hss[depth_idx], 'r--o', linewidth=2, label='Initial Model')
plt.gca().invert_yaxis()
plt.xlabel('Vs (m/s)')
plt.ylabel('Depth (m)')
plt.legend()
plt.title('Shear Wave Velocity with Posterior Uncertainty (Top 1000 m)')
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Sensitivity Kernel Plot (Top 1000 m)
depth_interp = np.arange(0, np.sum(h)+min(h), min(h))
depth_interp = depth_interp[depth_interp <= depth_limit]
snsmf_vstoti = np.zeros((len(depth_interp), snsmf_vstot.shape[1]))

for ii in range(snsmf_vstot.shape[1]):
    interp_func = interp1d(hs, snsmf_vstot[:, ii], kind='linear', fill_value='extrapolate')
    snsmf_vstoti[:, ii] = interp_func(depth_interp)

plt.figure()
plt.imshow(snsmf_vstoti / min(h), aspect='auto', extent=[fks[0], fks[-1], depth_interp[-1], depth_interp[0]])
plt.colorbar()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Depth (m)')
plt.title('Vs Sensitivity Kernel (Top 1000 m)')
plt.tight_layout()
plt.show()

# 4. Uncertainty Plot (Prior vs Posterior Relative Std Dev)
prior_std = np.sqrt(np.diag(Cm))
posterior_std = np.sqrt(np.diag(C_post))
prior_vs = vsv_guess[depth_idx]
posterior_vs = vsv_update[nupdat, depth_idx]
prior_std_rel = np.divide(prior_std[depth_idx], prior_vs, out=np.full_like(prior_vs, np.nan), where=prior_vs!=0)
posterior_std_rel = np.divide(posterior_std[depth_idx], posterior_vs, out=np.full_like(posterior_vs, np.nan), where=posterior_vs!=0)

plt.figure()
plt.plot(prior_std_rel, hss[depth_idx], 'r--', linewidth=1.5, label='Prior Relative Std Dev')
plt.plot(posterior_std_rel, hss[depth_idx], 'ko', linewidth=1.5, label='Posterior Relative Std Dev')
plt.gca().invert_yaxis()
plt.xlabel('Relative Std Dev (σ / Vs)')
plt.ylabel('Depth (m)')
plt.legend()
plt.title('Relative Model Uncertainty (Top 1000 m)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Save Covariance Matrices
np.savetxt('prior_model_covariance_example.csv', Cm, delimiter=',')
np.savetxt('posterior_model_covariance_Cpost_example.csv', C_post, delimiter=',')

# 5. Resolution Matrices
plt.figure()
plt.imshow(R_model, aspect='equal', extent=[z[0], z[-1], z[-1], z[0]])
plt.colorbar()
plt.xlabel('True Depth (m)')
plt.ylabel('Estimated Depth (m)')
plt.title('Model Resolution Matrix (Depth vs Depth)')
plt.tight_layout()
plt.show()

plt.figure()
plt.imshow(R_data, aspect='equal', extent=[fksr[0], fksr[-1], fksr[-1], fksr[0]])
plt.colorbar()
plt.xlabel('True Frequency (Hz)')
plt.ylabel('Estimated Frequency (Hz)')
plt.title('Data Resolution Matrix (Freq vs Freq)')
plt.tight_layout()
plt.show()
