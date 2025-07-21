"""
make_initial_model_dix.py
This script prepares an initial model for Rayleigh wave 
inversion using the Dix method developed by Haney and 
Tsai (2015). 

This code is a Python version of the original 
Matlab script which can be found in the raylee_examples package. 
 http://software.seg.org/2017/0003/index.html
Reference:
Haney, M. M., Tsai, V. C. (2017) Perturbational and nonperturbational 
inversion of Rayleigh-wave velocities, Geophysics, 82(3), F15-F28.
doi: 10.1190/geo2016-0397.1

------------------------------
Input files:
modx_phase_vels.ascii: Phase velocities
modx_freqs.ascii: Frequencies
error files if applicable
------------------------------
Output files: (Saved in initial_model_output)

velocity_values.txt: Phase velocities
velocity_values_errs.txt: Phase velocity errors
frequency_values.txt: Frequencies
mode_values.txt: Mode numbers
vtype_values.txt: Velocity types
vp_init.txt: Initial P-wave velocity
vs_init.txt: Initial S-wave velocity
rho_init.txt: Initial density
vpf.txt: Initial P-wave velocity for fluid
rhof.txt: Initial density for fluid
grid_values_solid.txt: Grid values for solid
grid_values_fluid.txt: Grid values for fluid
input_params.txt: Input parameters for the inversion

-------------------------------

Inversion Flags based on type of velocity data:
1 = Rayleigh wave phase velocity, homogeneous form
2 = Rayleigh wave phase velocity, power-law form w/0.25 Poisson's ratio 
3 = Love wave phase velocity, power-law form 
4 = Rayleigh wave group velocity, homogeneous form
5 = Rayleigh wave phase velocity, power-law form w/0.3 Poisson's ratio
6 = Rayleigh wave group velocity, power-law form w/0.25 Poisson's ratio 
7 = Love wave group velocity, power-law form
8 = Rayleigh wave group velocity, power-law form w/0.3 Poisson's ratio

"""

# ------------------------------

import numpy as np
import os
import pandas as pd
from scipy import interpolate   
from scipy.interpolate import interp1d
from scipy.linalg import sqrtm
import dix_inversion as dix


# -----------------------------
# 1. Load the phase velocity and frequency data
# -----------------------------
c = np.loadtxt('modx_phase_vels.txt')
fr = np.loadtxt('modx_freqs.txt')
c_low = c * (1 - 0.02)
c_hi = c * (1 + 0.02)

# -----------------------------
# 2. Set up the parameters for the Dix inversion
# -----------------------------
# Inversion parameters
# These parameters can be adjusted based on the specific requirements of the inversion
# Inversion flag for inversion type
inv_flag_ar = 1 
# multiplicative coefficient between wavelength and sensitivity depth
xcof = 0.5
# number of layers per wavelength at the sensitivity depth
lrho = 20
# range of correlation length factors to scan over
# These are used to control the smoothness of the model
lcormultv = np.arange(10, 1001, 10)
# range of model variance factors to scan over
# These are used to control the model variance
sigmscalev = np.arange(1, 20.1, 0.1)
# low value of acceptable chi-squared
chiwinlo = 1.0
# high value of acceptable chi-squared
chiwinhi = 1.5
# multiple of minimum depth for model to extend to
zminfact = 1
# multiple of maximum depth for model to extend to
zmaxfact = 4
# Poisson's ratio (used for homogeneous model formulation and for building the output Vp model from the Vs model)
# This is typically set to 0.45 for Rayleigh wave inversions
v = 0.45
# Nominal power law exponent (used for group velocity power law forms)
alph = 0.3

# These parameters are needed simply to pass output to nonlinear inversion:
# Nominal density value (used for building the output density model)
rhod = 2000
# Number of fluid layers to place above model
Nnf = 0
# Thickness of fluid layers placed above model (if Nnf=0, not subsequently used)
hfval = 200
# Density of fluid layers (if Nnf=0, not subsequently used)
rhovfal = 1000
# Velocity of fluid layers (if Nnf=0, not subsequently used)
vpvfal = 1500
# Flag whether Vp/Vs ratio is to be fixed in nonlinear inversion
# 0 = no, 1 = yes
pratioflag = 1
# Max number of nonlinear updates
nupdats = 15
# Model standard deviation factor
# This is used to control the model variance in the nonlinear inversion
mstdfact = 2
# Smoothness scale, typically less than zmax
smscl = 10


# -----------------------------
# 3. Build the model based on the Dix method
# -----------------------------
# Build the exponential layer model based on the input phase velocities and frequencies
thks, z = dix.build_exponential_layer_model(c, fr, xcof, lrho, zminfact, zmaxfact)

# # -----------------------------
# 4. Set up the model parameters and factors
# -----------------------------
# Size of data and model
dsz = len(fr)
msz = len(z) - 1

# Angular Frequency, Wavenumber, and Wavenumber times Depth
w = 2 * np.pi * fr
k = w / c
kz = np.outer(k, z)

# calculate array of data std deviations based on the hi and low values of c
sigma_d_ar = ((c_hi ** 2 - c_low ** 2) / 2)
Cd = np.diag(sigma_d_ar ** 2)

# -------------------------------
# 5. Calculate the Kernel functions based on the inversion type
# -------------------------------
# The kernel functions are computed based on the inversion flag from the Dix method in dix_inverse.py
G = dix.compute_dix_kernel(inv_flag_ar, v, kz, alpha=alph)


# -------------------------------
# 6. Xia interpolation for initial model
"""
Create Xia model interp (0.5c/f) and midpoint distance matrix 
find best fit power law, find slope at shallow point, and 
do the extrapolation to the surface
if power law gives negative velocity at surface, use linear function
if linear function gives negative velocity at surface use shallowest 
velocity, repeat similarly for deepest point in model
"""
# -------------------------------
# compute midpoint depths for Xia interpolation from dix_inverse.py
z_mid = z[:-1]
xia_int, z_mid_mat = dix.compute_xia_interpolation(c, fr, xcof, z_mid)


# -------------------------------
# 7. Inverse the model using the Dix method
# -------------------------------
# Initialize counters and arrays
modcountr = 0  # Counter for acceptable (physically valid + good fit) models
chi_sqd_arr = np.zeros((len(lcormultv), len(sigmscalev)))  # Stores chi-squared values for each parameter combination
mod_neg_arr = np.zeros_like(chi_sqd_arr)  # Stores 1 if all model parameters > 0, else 0
beta_sq_r2_all = []  # List to store accepted beta^2 models

# Loop over correlation length scaling factors (lcormultv)
for lcntr, lcormult in enumerate(lcormultv):

    # Loop over model standard deviation scaling factors (sigmscalev)
    for mcntr, sigmscale in enumerate(sigmscalev):

        # --- Construct Model Covariance Matrix Cm ---
        l_cor = np.median(thks) * lcormult  # Characteristic correlation length
        sigma_m = np.mean(sigma_d_ar) * sigmscale  # Standard deviation of the model
        Cm = sigma_m ** 2 * np.exp(-z_mid_mat / l_cor)  # Exponential spatial covariance (Tarantola style)

        # --- Compute inverse square roots of Data and Model Covariance matrices ---
        Cd_mhalf = np.linalg.inv(sqrtm(Cd))  # Cd^{-1/2}
        Cm_mhalf = np.linalg.inv(sqrtm(Cm))  # Cm^{-1/2}

        # --- Construct augmented data and G matrices ---
        d_aug = np.vstack([
            Cd_mhalf @ (c ** 2).reshape(-1, 1),       # Transformed data vector: observed phase velocity squared
            Cm_mhalf @ (xia_int ** 2).reshape(-1, 1)  # Transformed prior model: interpolated shear profile squared
        ])
        G_aug = np.vstack([
            Cd_mhalf @ G,      # Transformed sensitivity kernel
            Cm_mhalf           # Regularization terms
        ])

        # --- Solve the regularized least-squares problem for beta^2 (shear velocity squared) ---
        beta_sq_r2 = np.linalg.solve(G_aug.T @ G_aug, G_aug.T @ d_aug).flatten()

        # --- Compute chi-squared misfit using original data covariance ---
        resid = (c ** 2 - G @ beta_sq_r2).reshape(-1, 1)  # Residual = observed - predicted (in c^2 space)
        chi_sqd = (resid.T @ np.linalg.inv(Cd) @ resid) / len(c)  # Normalized chi-squared
        chi_sqd = chi_sqd.item()  # Convert from 1x1 matrix to scalar

        # --- Store results ---
        chi_sqd_arr[lcntr, mcntr] = chi_sqd

        # Check that all beta^2 values are positive (i.e., physically valid shear velocities)
        mod_neg_arr[lcntr, mcntr] = np.min(beta_sq_r2 > 0)

        # If chi-squared is within acceptable bounds and model is physical, keep it
        if chiwinlo < chi_sqd < chiwinhi and mod_neg_arr[lcntr, mcntr] != 0:
            modcountr += 1
            beta_sq_r2_all.append(beta_sq_r2)

    # Log progress for each lcormult (row in the parameter space)
    print(f"Processed lcormult={lcormult}, total models={modcountr}")
# -----------------------------

print("Minimum chi-squared:", np.min(chi_sqd_arr))
print("Maximum chi-squared:", np.max(chi_sqd_arr))
print("Any negative beta^2 models:", np.sum(mod_neg_arr == 0))

# -----------------------------
# 9. Mean of acceptable models
# -----------------------------
if 'beta_sq_r2_all' in locals() and modcountr > 0:
    beta_sq_r2_all = np.array(beta_sq_r2_all).T
    beta_sq_r2m = np.mean(beta_sq_r2_all, axis=1)
elif 'chi_sqd' in locals() and np.min(chi_sqd) > chiwinhi:
    # An error message if no models found
    raise ValueError('No acceptable models found: All chi-squareds above acceptable window. Consider expanding search grid over regularization parameters or expanding acceptable chi-squared range.')
elif 'chi_sqd' in locals() and np.max(chi_sqd) < chiwinlo:
    # An error message if no models found
    raise ValueError('No acceptable models found: All chi-squareds below acceptable window. Consider expanding search grid over regularization parameters or expanding acceptable chi-squared range.')
else:
    # An error message if no models found
    raise ValueError('No acceptable models found: All chi-squareds above or below acceptable window. Consider densifying search grid over regularization parameters')

# -----------------------------
# 10. Plot the Chi-squared and range of acceptable models
# -----------------------------
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6), dpi=300)
plt.imshow(chi_sqd_arr * mod_neg_arr, aspect='auto',
           extent=[sigmscalev[0], sigmscalev[-1], lcormultv[0], lcormultv[-1]],
           origin='lower', cmap='jet')
plt.colorbar(label='Chi-squared', orientation='vertical')
plt.contour(sigmscalev, lcormultv, chi_sqd_arr * mod_neg_arr,
            levels=[chiwinlo, chiwinhi], colors='white', linestyles='--', linewidths=2)
plt.title('Chi-squared misfit for Dix inversion', fontsize=14, fontweight='bold')
plt.xlabel('Model standard deviation factor', fontsize=14, fontweight='bold')
plt.ylabel('Model correlation length factor', fontsize=14, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('chi_squared_range.png', dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------
# 11. Save the model parameters
# -----------------------------

# Define output directory
outdir = 'initial_model_output'
os.makedirs(outdir, exist_ok=True)

def write_txt(filename, array):
    filepath = os.path.join(outdir, filename)
    np.savetxt(filepath, array, fmt='%10.5f')

# Derived quantities
Nf = len(c)
Nn = len(thks)
vout = c
voute = c_hi - c
fks = fr
modnv = np.full(Nf, 1, dtype=int)  

if inv_flag_ar in [1, 2, 3, 5]:
    vtypv = np.zeros(Nf)
else:
    vtypv = np.ones(Nf)

vsv = np.sqrt(beta_sq_r2m[:Nn])
# print("Shear wave velocities (vsv):", vsv)
vpv = np.sqrt((2*v - 2)/(2*v - 1)) * vsv
rhov = rhod * np.ones(Nn)
vpvf = vpvfal * np.ones(Nnf)
rhovf = rhovfal * np.ones(Nnf)
h = thks
# print("Layer thicknesses (h):", h)
hfv = hfval * np.ones(Nnf)

# Write files
write_txt('velocity_values.txt', vout)
write_txt('velocity_values_errs.txt', voute)
write_txt('frequency_values.txt', fks)
write_txt('mode_values.txt', modnv)
write_txt('vtype_values.txt', vtypv)

if pratioflag == 0:
    write_txt('vp_init.txt', vpv)
elif pratioflag == 1:
    with open(os.path.join(outdir, 'vp_init.txt'), 'w') as f:
        f.write(f'{np.sqrt((2*v - 2)/(2*v - 1)):.5f}\n')

write_txt('vs_init.txt', vsv)
write_txt('rho_init.txt', rhov)
write_txt('vpf.txt', vpvf)
write_txt('rhof.txt', rhovf)
write_txt('grid_values_solid.txt', h)
write_txt('grid_values_fluid.txt', hfv)

# Write input_params.txt
with open(os.path.join(outdir, 'input_params.txt'), 'w') as f:
    f.write('%% input parameters for Rayleigh/Scholte wave inversion\n\n')
    f.write(f'{pratioflag}  %% flag for fixed poisson\'s ratio (0=no,1=yes)\n')
    f.write(f'{smscl:10.5f}  %% smoothness scale (m)\n')
    f.write(f'{mstdfact:10.5f}  %% a priori model standard deviation factor\n')
    f.write(f'{nupdats}  %% maximum number of updates (iterations)\n')
    f.write(f'{Nf}  %% number of measurements\n')
    f.write(f'{Nn}  %% number of elements in solid part of model\n')
    f.write(f'{Nnf}  %% number of elements in fluid part of model\n')
    f.write(f'{chiwinlo:10.5f}  %% lower chi squared window\n')
    f.write(f'{chiwinhi:10.5f}  %% higher chi squared window\n')

# -----------------------------
# End of the script
# -----------------------------
# Final Output files listed below:
# - velocity_values.txt
# - velocity_values_errs.txt
# - frequency_values.txt
# - mode_values.txt
# - vtype_values.txt
# - vp_init.txt     
# - vs_init.txt
# - rho_init.txt
# - vpf.txt
# - rhof.txt    
# - grid_values_solid.txt
# - grid_values_fluid.txt
# - input_params.txt
