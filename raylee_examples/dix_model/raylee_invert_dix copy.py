# import numpy as np
# import scipy.linalg as la
# from scipy.interpolate import interp1d
# from scipy.linalg import sqrtm
# import matplotlib.pyplot as plt
# import time
# from dix_inversion.raylee_sensitivity import raylee_sensitivity
# from dix_inversion.utils import check_nans
# from dix_inversion.linvers import linvers
# import os
# from dix_inversion.dix_inverse import DampedInversion


# # Start timing
# t_start = time.time()

# # Helper to construct full path
# def full_path(fname):
#     return os.path.join('initial_model_output', fname)

# # Helper to read a vector from file
# def read_vector_from_file(filepath, N):
#     with open(filepath, 'r') as f:
#         return np.array([float(f.readline()) for _ in range(N)])

# # Read parameters
# inp = np.loadtxt(full_path('input_params.txt'), comments='%')
# pratioflag = int(inp[0])
# lsmth = inp[1]
# msigmaf = inp[2]
# nupds = int(inp[3])
# Nf = int(inp[4])
# Nn = int(inp[5])
# Nnf = int(inp[6])
# chilo = inp[7]
# chihi = inp[8]


# # Read solid and fluid grid spacing
# h = read_vector_from_file(full_path('grid_values_solid.txt'), Nn)
# hfv = read_vector_from_file(full_path('grid_values_fluid.txt'), Nnf)

# # Frequency and data files
# fks = read_vector_from_file(full_path('frequency_values.txt'), Nf)
# U_data = read_vector_from_file(full_path('velocity_values.txt'), Nf)
# U_data_errs = read_vector_from_file(full_path('velocity_values_errs.txt'), Nf)
# modn = read_vector_from_file(full_path('mode_values.txt'), Nf).astype(int)
# vflg = read_vector_from_file(full_path('vtype_values.txt'), Nf).astype(int)

# # Initial Vs model
# vsv_init = read_vector_from_file(full_path('vs_init.txt'), Nn)

# # Initial Vp model
# if pratioflag == 0:
#     vpv = read_vector_from_file(full_path('vp_init.txt'), Nn)
# elif pratioflag == 1:
#     with open(full_path('vp_init.txt'), 'r') as f:
#         vpvsratio = float(f.readline())
#     vpv = vpvsratio * vsv_init
# else:
#     raise ValueError("Invalid value for pratioflag")

# # Initial density
# rhov = read_vector_from_file(full_path('rho_init.txt'), Nn)

# # Fluid Vp and density
# vpfv = read_vector_from_file(full_path('vpf.txt'), Nnf)
# rhofv = read_vector_from_file(full_path('rhof.txt'), Nnf)

# # Initialize model update array
# vsv_update = np.zeros((nupds + 1, Nn))

# # Node depths
# hs = np.zeros(Nn)
# hs[0] = 0
# for i in range(1, Nn):
#     hs[i] = np.sum(h[:i])

# # Element center depths
# hss = np.zeros(Nn)
# hss[0] = h[0] / 2
# for i in range(1, Nn):
#     hss[i] = np.sum(h[:i]) + h[i] / 2

# # ========== Sanity Checks ==========
# if np.any(rhov <= 0):
#     raise ValueError("Negative density values exist in initial guess")
# if np.any(vsv_init <= 0):
#     raise ValueError("Negative shear velocity values exist in initial guess")
# pratio = (vpv**2 - 2 * vsv_init**2) / (2 * (vpv**2 - vsv_init**2))
# if np.any((pratio <= -1) | (pratio >= 0.5)):
#     raise ValueError("Impossible Poisson ratio values exist in initial guess")
# if np.any(rhofv <= 0):
#     raise ValueError("Negative density values exist in fluid layer")


# # # ------------------------------------------------------
# # # Compute sensitivity kernel using the initial Vs model
# # # ------------------------------------------------------
# # U, snsmf_vstot, _ = raylee_sensitivity(
# #     Nn, vsv, vpv, rhov, fks, h, modn, vflg,
# #     Nnf, vpfv, rhofv, hfv, pratioflag
# # )

# # # ------------------------------------------------------------------------
# # # Identify frequencies where both the predicted and observed data are valid
# # # This removes any NaNs due to ill-posed sensitivity or bad measurements
# # # ------------------------------------------------------------------------
# # Ur, U_datar, fksr, fksri, modnr, vflgr, snsmf_vstotr = check_nans(
# #     U, U_data, fks, modn, vflg, snsmf_vstot
# # )
# # Nfr = len(fksr)  # Number of "retained" (non-NaN) frequencies

# # # ------------------------------------------------------
# # # Save current model and prediction as baseline "guess"
# # # ------------------------------------------------------
# # vsv_guess = vsv.copy()
# # U_guess = Ur.copy()
# # fksr_guess = fksr.copy()

# # # ------------------------------------------------------------------------
# # # A priori model covariance matrix (MCM):
# # # Exponential spatial correlation based on depth differences
# # # Equivalent to: mcm(i,j) = (σ²) * exp(-|zi - zj| / lsmth)
# # # ------------------------------------------------------------------------
# # msigma = np.mean(U_data_errs[fksri]) * msigmaf
# # depth_matrix_diff = np.abs(np.subtract.outer(hs, hs))  # [Nn x Nn] matrix
# # mcm = (msigma ** 2) * np.exp(-depth_matrix_diff / lsmth)

# # # Inverse square root of model covariance (for regularization)
# # mcmisr = la.sqrtm(la.inv(mcm))  # [Nn x Nn]

# # # ------------------------------------------------------------------------
# # # A priori data covariance (DCM) is diagonal (uncorrelated errors)
# # # DCM inverse square root = diag(1 / σi)
# # # ------------------------------------------------------------------------
# # dcm = np.diag(U_data_errs[fksri] ** 2)     # [Nfr x Nfr]
# # dcmisr = np.diag(1.0 / U_data_errs[fksri]) # [Nfr x Nfr]

# # # ------------------------------------------------------
# # # RMS error (normalized residual) of the initial guess
# # # ------------------------------------------------------
# # rmserror = np.zeros(nupds + 2)
# # chisqurd = np.zeros(nupds + 2)
# # Nfrv = np.zeros(nupds + 2, dtype=int)

# # rmserror[0] = np.sqrt(np.mean((U_guess - U_datar) ** 2))

# # # Chi-squared = (residual^T * DCM⁻¹ * DCM⁻¹ * residual)
# # # Same as: residual.T @ (DCMisr @ DCMisr) @ residual
# # residual = U_guess - U_datar
# # chisqurd[0] = residual @ dcmisr @ dcmisr @ residual
# # Nfrv[0] = Nfr

# # # ------------------------------------------------------
# # # Check if initial model already satisfies the chi² bounds
# # # If so, inversion is either unnecessary or ill-posed
# # # ------------------------------------------------------
# # reduced_chi2 = chisqurd[0] / Nfr
# # if reduced_chi2 < chilo:
# #     raise ValueError('Initial model fits data to less than 1 chi-squared')
# # elif reduced_chi2 < chihi:
# #     raise ValueError('Initial model fits data within acceptable chi-squared window')
# # else:
# #     pass  # Proceed with inversion
# # # ------------------------------------------------------



# inversion = DampedInversion(
#     vsv_init=vsv_init,
#     vpvsratio=vpvsratio,
#     U_data=U_data,
#     U_data_errs=U_data_errs,
#     fks=fks,
#     modn=modn,
#     vflg=vflg,
#     rhov=rhov,
#     h=h,
#     vpfv=vpfv,
#     rhofv=rhofv,
#     hfv=hfv,
#     pratioflag=pratioflag,
#     msigmaf=msigmaf,
#     lsmth=lsmth,
#     chilo=chilo,
#     chihi=chihi,
#     nupds=nupds,
#     hs=hs,
#     hss=hss,
# )

# results = inversion.run()
# # ------------------------------------------------------

# vsv_guess = results["vsv_guess"]
# vsv_final = results["vsv_final"]
# U_guess = results["U_initial"]
# U = results["U_final"]
# U_data = results["U_data"]
# U_data_errs = results["U_data_errs"]
# fks = results["fks"]
# hss = results["hss"]
# vs_post_std = results["vs_post_std"]
# G = results["G"]
# Cm = results["Cm"]
# C_post = results["C_post"]
# R_model = results["R_model"]
# R_data = results["R_data"]
# Nfr = results["Nfr"]


# # # ---------------------- Convergence Plot ----------------------
# # # ---------------------- Posterior Covariance and Resolution Matrices ----------------------
# # depth_limit = 150  # cutoff depth in meters

# # plt.figure()
# # plt.subplot(2, 1, 1)
# # plt.plot(np.arange(nupdat + 1), np.array(chisqurd[:nupdat + 1]) / np.array(Nfrv[:nupdat + 1]), '-o', linewidth=2)
# # plt.xlabel('Iteration')
# # plt.ylabel(r'$\chi^2$ per frequency')
# # plt.title('Chi-squared Convergence')
# # plt.grid(True)

# # plt.subplot(2, 1, 2)
# # plt.plot(np.arange(nupdat + 1), rmserror[:nupdat + 1], '-s', linewidth=2)
# # plt.xlabel('Iteration')
# # plt.ylabel('RMS Error')
# # plt.title('RMS Error Convergence')
# # plt.grid(True)
# # plt.tight_layout()
# # plt.show()

# # print(f"Chi-squared per frequency (iteration {nupdat + 1}): {chisqurd[nupdat + 1] / Nfr:.6f}")
# # # max abs dvs
# # print(f"Max abs(dvs): {np.max(np.abs(dvs)):.6f}")



# # ---------------------- Data Fit Plots ----------------------
# # With error bars
# plt.figure()
# plt.errorbar(fks, U_data, yerr=U_data_errs, fmt='bo', linewidth=2, markersize=6, label='Observed (with error)')
# plt.plot(fks, U_guess, 'ro', linewidth=2, markersize=6, label='Initial')
# plt.plot(fks, U, 'k-', linewidth=2, label='Final')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Velocity (m/s)')
# plt.title('Observed vs Modeled Dispersion')
# plt.legend()
# plt.grid(True)
# plt.gca().tick_params(labelsize=12)
# plt.show()

# # Without error bars
# plt.figure()
# plt.plot(fks, U_data, 'bo', linewidth=2, markersize=6, label='Observed')
# plt.plot(fks, U_guess, 'ro', linewidth=2, markersize=6, label='Initial')
# plt.plot(fks, U, 'k-', linewidth=2, label='Final')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Velocity (m/s)')
# plt.title('Observed vs Modeled Dispersion')
# plt.legend()
# plt.grid(True)
# plt.gca().tick_params(labelsize=12)
# plt.show()

# # Plot settings
# depth_limit = 150  # meters
# depth_idx = hss <= depth_limit

# # # 1. Shear Velocity Model Plot (cut at 150 m)
# # plt.figure()
# # plt.plot(vsv_guess[depth_idx], hss[depth_idx], 'r--o', linewidth=2, label='Initial Model')
# # plt.plot(vsv_update[nupdat, depth_idx], hss[depth_idx], 'k-o', linewidth=2, label='Final Inverted Model')
# # plt.gca().invert_yaxis()
# # plt.xlabel('Vs (m/s)')
# # plt.ylabel('Depth (m)')
# # plt.legend()
# # plt.title('Shear Wave Velocity Model (Top 150 m)')
# # plt.grid(True)
# # plt.tight_layout()
# # plt.show()

# # # 2. Updated Shear Velocity Plot with Error Bars
# # plt.figure()
# # plt.errorbar(vsv_update[nupdat, depth_idx], hss[depth_idx], xerr=vs_post_std[depth_idx], fmt='k-o', linewidth=2, label='Inverted Vs ±1σ')
# # plt.plot(vsv_guess[depth_idx], hss[depth_idx], 'r--o', linewidth=2, label='Initial Model')
# # plt.gca().invert_yaxis()
# # plt.xlabel('Vs (m/s)')
# # plt.ylabel('Depth (m)')
# # plt.legend()
# # plt.title('Shear Wave Velocity with Posterior Uncertainty (Top 150 m)')
# # plt.grid(True)
# # plt.tight_layout()
# # plt.show()

# # # 3. Sensitivity Kernel Plot (Top 150 m)
# # depth_interp = np.arange(0, np.sum(h)+min(h), min(h))
# # depth_interp = depth_interp[depth_interp <= depth_limit]
# # snsmf_vstoti = np.zeros((len(depth_interp), snsmf_vstot.shape[1]))

# # for ii in range(snsmf_vstot.shape[1]):
# #     interp_func = interp1d(hs, snsmf_vstot[:, ii], kind='linear', fill_value='extrapolate')
# #     snsmf_vstoti[:, ii] = interp_func(depth_interp)

# # plt.figure()
# # plt.imshow(snsmf_vstoti / min(h), aspect='auto', extent=[fks[0], fks[-1], depth_interp[-1], depth_interp[0]])
# # plt.colorbar()
# # plt.xlabel('Frequency (Hz)')
# # plt.ylabel('Depth (m)')
# # plt.title('Vs Sensitivity Kernel (Top 150 m)')
# # plt.tight_layout()
# # plt.show()

# # # 4. Uncertainty Plot (Prior vs Posterior Relative Std Dev)
# # prior_std = np.sqrt(np.diag(Cm))
# # posterior_std = np.sqrt(np.diag(C_post))
# # prior_vs = vsv_guess[depth_idx]
# # posterior_vs = vsv_update[nupdat, depth_idx]
# # prior_std_rel = np.divide(prior_std[depth_idx], prior_vs, out=np.full_like(prior_vs, np.nan), where=prior_vs!=0)
# # posterior_std_rel = np.divide(posterior_std[depth_idx], posterior_vs, out=np.full_like(posterior_vs, np.nan), where=posterior_vs!=0)

# # plt.figure()
# # plt.plot(prior_std_rel, hss[depth_idx], 'r--', linewidth=1.5, label='Prior Relative Std Dev')
# # plt.plot(posterior_std_rel, hss[depth_idx], 'ko', linewidth=1.5, label='Posterior Relative Std Dev')
# # plt.gca().invert_yaxis()
# # plt.xlabel('Relative Std Dev (σ / Vs)')
# # plt.ylabel('Depth (m)')
# # plt.legend()
# # plt.title('Relative Model Uncertainty (Top 150 m)')
# # plt.grid(True)
# # plt.tight_layout()
# # plt.show()

# # # Save Covariance Matrices
# # np.savetxt('prior_model_covariance_example.csv', Cm, delimiter=',')
# # np.savetxt('posterior_model_covariance_Cpost_example.csv', C_post, delimiter=',')

# # # 5. Resolution Matrices
# # plt.figure()
# # plt.imshow(R_model, aspect='equal', extent=[z[0], z[-1], z[-1], z[0]])
# # plt.colorbar()
# # plt.xlabel('True Depth (m)')
# # plt.ylabel('Estimated Depth (m)')
# # plt.title('Model Resolution Matrix (Depth vs Depth)')
# # plt.tight_layout()
# # plt.show()

# # plt.figure()
# # plt.imshow(R_data, aspect='equal', extent=[fksr[0], fksr[-1], fksr[-1], fksr[0]])
# # plt.colorbar()
# # plt.xlabel('True Frequency (Hz)')
# # plt.ylabel('Estimated Frequency (Hz)')
# # plt.title('Data Resolution Matrix (Freq vs Freq)')
# # plt.tight_layout()
# # plt.show()
