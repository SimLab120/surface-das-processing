"""
numerical_tests.py

This script reproduces Jacobian tests from Table 2 of Cercato (2007, GJI) using
Rayleigh wave sensitivity kernels computed by the Raylee code.

It also validates sensitivity of phase velocity to crustal thickness using:
1. Perturbation-theory-based sensitivity kernels
2. Brute-force finite difference method (by altering crustal thickness)

Author: Adapted from Raylee package given by Matt Haney and Victor Tsai
"""

import numpy as np
from dix_inversion.raylee_sensitivity import raylee_sensitivity
from dix_inversion.raylee_lysmer import raylee_lysmer

# -------------------------------------------------------------------------------------
# STEP 1: Define Earth Model and Frequencies
# -------------------------------------------------------------------------------------

# Set the uniform grid spacing (250 m) for 1600 elements
h = np.full(1600, 250.0)  # in meters
Nn = len(h)               # total number of layers/nodes

# Frequencies corresponding to periods 40s, 30s, 20s (Hz)
fks = 1.0 / np.array([40.0, 30.0, 20.0])

# Mode numbers (1 = fundamental mode for all frequencies)
modnv = np.ones(len(fks), dtype=int)

# Define a 2-layer model (Cercato's Novotny crust-mantle profile)
# Crust: first 140 elements
vpv = np.zeros(Nn)  # P-wave velocity (m/s)
vsv = np.zeros(Nn)  # S-wave velocity (m/s)
rhov = np.zeros(Nn) # Density (kg/m³)

vpv[:140] = 6000.0
vsv[:140] = 3500.0
rhov[:140] = 2700.0

# Mantle: remaining elements
vpv[140:] = 8000.0
vsv[140:] = 4500.0
rhov[140:] = 3300.0

# Convert all quantities to Raylee's required units:
# meters → kilometers, m/s → km/s, kg/m³ → g/cm³
h /= 1000.0
vpv /= 1000.0
vsv /= 1000.0
rhov /= 1000.0

# Flag for phase velocity computation (0 = phase velocity)
vflg = np.zeros(len(fks), dtype=int)

# No fluid layer on top (Raylee can handle fluid-solid interfaces)
Nnf, vpfv, rhofv, hfv = 0, 0, 0, 0

# -------------------------------------------------------------------------------------
# STEP 2: Compute Sensitivity Kernels (Rayleigh wave Jacobian)
# -------------------------------------------------------------------------------------

# Returns:
#   U             - Phase velocities (km/s)
#   snsmf_vstot   - Sensitivity to shear velocity Vs
#   snsmf_h       - Sensitivity to layer thickness h
U, snsmf_vstot, snsmf_h = raylee_sensitivity(
    Nn, vsv, vpv, rhov, fks, h, modnv, vflg,
    Nnf, vpfv, rhofv, hfv, pratioflag=0
)

# -------------------------------------------------------------------------------------
# STEP 3: Compare with Cercato's Table 2 (Jacobian verification)
# -------------------------------------------------------------------------------------

# Sum sensitivity of Vs over crust and mantle separately
crust_vs_sensitivity = np.sum(snsmf_vstot[:140, :], axis=0)
mantle_vs_sensitivity = np.sum(snsmf_vstot[140:, :], axis=0)

print("\n=== Jacobian (Vs Sensitivity) Comparison with Cercato (2007) ===")
print("Format: [CrustSum  MantleSum] for each frequency")
print(np.vstack([crust_vs_sensitivity, mantle_vs_sensitivity]).T)

# -------------------------------------------------------------------------------------
# STEP 4: Derivative of Phase Velocity wrt Crust Thickness (Perturbation Method)
# -------------------------------------------------------------------------------------

# Approximate dU/dh for crust by subtracting adjacent sensitivity rows
# h[139] is top of mantle; h[138] is bottom of crust
thickness_perturb_sensitivity = snsmf_h[139, :] - snsmf_h[140, :]

print("\n=== dU/dh using Perturbation Sensitivity ===")
print(thickness_perturb_sensitivity)

# -------------------------------------------------------------------------------------
# STEP 5: Finite Difference Test - Decrease Crust Thickness by 250 m
# -------------------------------------------------------------------------------------

# Rebuild model: Crust = 34.75 km instead of 35 km
h = np.full(1600, 250.0)  # in meters

# Define thinner crust (139 layers instead of 140)
vpv[:139] = 6000.0
vsv[:139] = 3500.0
rhov[:139] = 2700.0

# Define mantle
vpv[139:] = 8000.0
vsv[139:] = 4500.0
rhov[139:] = 3300.0

# Convert units
h /= 1000.0
vpv /= 1000.0
vsv /= 1000.0
rhov /= 1000.0

# Calculate phase velocities after crust thinning
vpp_thin = []
for f in fks:
    modn = 1
    Uf, vpk, _, _ = raylee_lysmer(Nn, vsv, vpv, rhov, f, h, modn,
                                  Nnf, vpfv, rhofv, hfv)
    vpp_thin.append(vpk)
vpp_thin = np.array(vpp_thin)

# Finite difference approximation: (original - perturbed) / delta_thickness
delta_h = 35.0 - 34.75  # km
fdiff_thinner = (U - vpp_thin) / delta_h

print("\n=== Finite Difference (Thinner Crust, -250 m) ===")
print(fdiff_thinner)

# -------------------------------------------------------------------------------------
# STEP 6: Finite Difference Test - Increase Crust Thickness by 250 m
# -------------------------------------------------------------------------------------

# Rebuild model: Crust = 35.25 km instead of 35 km
h = np.full(1600, 250.0)

# Define thicker crust (141 layers instead of 140)
vpv[:141] = 6000.0
vsv[:141] = 3500.0
rhov[:141] = 2700.0

# Define mantle
vpv[141:] = 8000.0
vsv[141:] = 4500.0
rhov[141:] = 3300.0

# Convert units again
h /= 1000.0
vpv /= 1000.0
vsv /= 1000.0
rhov /= 1000.0

# Calculate phase velocities after crust thickening
vpp_thick = []
for f in fks:
    modn = 1
    Uf, vpk, _, _ = raylee_lysmer(Nn, vsv, vpv, rhov, f, h, modn,
                                  Nnf, vpfv, rhofv, hfv)
    vpp_thick.append(vpk)
vpp_thick = np.array(vpp_thick)

# Finite difference approximation: (original - perturbed) / delta_thickness
delta_h = 35.0 - 35.25  # km
fdiff_thicker = (U - vpp_thick) / delta_h

print("\n=== Finite Difference (Thicker Crust, +250 m) ===")
print(fdiff_thicker)
