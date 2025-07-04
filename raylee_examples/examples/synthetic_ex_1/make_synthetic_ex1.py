# =============================================================================
# make_synthetic_ex1.py
#
# This script generates synthetic Rayleigh-wave phase/group velocity data.
# It follows the methodology in Haney and Tsai (2017), Geophysics, 82(3), F15-F28.
#
# Output files:
#   - velocity_values.txt
#   - velocity_values_errs.txt
#   - frequency_values.txt
#   - mode_values.txt
#   - vtype_values.txt
# =============================================================================

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from raylee_processing.forward.raylee_lysmer import raylee_lysmer

# -----------------------------
# 1. Construct the model grid
# -----------------------------
Nn = 240                          # Number of solid elements
h = 250 * np.ones(Nn)            # Grid spacing in solid (meters)

# ----------------------------------
# 2. Frequency and mode definition
# ----------------------------------
Nf = 56
fmin = 0.10
df = 0.01
fks = fmin + df * np.arange(Nf)  # Frequencies for velocity measurement

# Model both fundamental (mode 1) and 1st higher mode (mode 2)
Nf *= 2
modnv = np.concatenate([np.ones(Nf // 2), 2 * np.ones(Nf // 2)])
fks = np.concatenate([fks, fks])
vtypv = np.zeros(Nf)             # 0: phase velocity, 1: group velocity

# ------------------------------
# 3. Define medium parameters
# ------------------------------
vpvsr = 1.7321                  # Vp/Vs ratio
gardc = 309.6                   # Gardner's constant
powr = 0.25                     # Gardner's exponent

# No water layer, so no fluid parameters needed
Nnf = 0
vpfv = 1
rhofv = 1
hfv = 1

# Define 4-layer velocity and density model
layrth1 = 5
layrth2 = 10
layrth3 = 50

vplay1 = 4000; vslay1 = vplay1 / vpvsr; rholay1 = gardc * (vplay1 ** powr)
vplay2 = 3396; vslay2 = vplay2 / vpvsr; rholay2 = gardc * (vplay2 ** powr)
vplay3 = 4500; vslay3 = vplay3 / vpvsr; rholay3 = gardc * (vplay3 ** powr)
vplay4 = 6000; vslay4 = vplay4 / vpvsr; rholay4 = gardc * (vplay4 ** powr)

vpv = np.concatenate([
    vplay1 * np.ones(layrth1),
    vplay2 * np.ones(layrth2),
    vplay3 * np.ones(layrth3),
    vplay4 * np.ones(Nn - (layrth1 + layrth2 + layrth3))
])
vsv = np.concatenate([
    vslay1 * np.ones(layrth1),
    vslay2 * np.ones(layrth2),
    vslay3 * np.ones(layrth3),
    vslay4 * np.ones(Nn - (layrth1 + layrth2 + layrth3))
])
rhov = np.concatenate([
    rholay1 * np.ones(layrth1),
    rholay2 * np.ones(layrth2),
    rholay3 * np.ones(layrth3),
    rholay4 * np.ones(Nn - (layrth1 + layrth2 + layrth3))
])

# ---------------------------
# 4. Forward modeling loop
# ---------------------------
vout = np.zeros(Nf)
for i, f in enumerate(fks):
    modn = int(modnv[i])
    # Set all fluid-related arguments to None or zero as they are unused
    kk, vpk, vgk, ev = raylee_lysmer(Nn, vsv, vpv, rhov, f, h, modn, Nnf, )
    vout[i] = vpk if vtypv[i] == 0 else vgk

# --------------------------------------
# 5. Optionally add 2.5% Gaussian noise
# --------------------------------------
add_noise = False                           # <-- Set to True if you want to add noise

if add_noise:
    np.random.seed(0)
    vout *= (1 + 0.025 * np.random.randn(Nf))

# ---------------------------------------------
# 6. Filter out NaNs and clean up the vectors
# ---------------------------------------------
valid = ~np.isnan(vout)
vout = vout[valid]
fks = fks[valid]
modnv = modnv[valid]
vtypv = vtypv[valid]
Nf = len(vout)

# -------------------
# 7. Save output data
# -------------------
np.savetxt('velocity_values.txt', vout, fmt='%.5f')
np.savetxt('velocity_values_errs.txt', vout * 0.025, fmt='%.5f')
np.savetxt('frequency_values.txt', fks, fmt='%.5f')
np.savetxt('mode_values.txt', modnv, fmt='%.5f')
np.savetxt('vtype_values.txt', vtypv, fmt='%.5f')

print("Synthetic Lysmer data generation complete. Output written to text files.")

# =============================================================================
# 8. Also compute using DISBA for comparison
# =============================================================================
from disba import PhaseDispersion

print("Computing DISBA phase velocities...")

# Use the same h, vpv, vsv, rhov as defined earlier
thickness = h
vp = vpv
vs = vsv
rho = rhov

# Frequency and sorted period array
frequencies_disba = np.linspace(0.10, 0.65, Nf // 2)
periods = np.sort(1.0 / frequencies_disba)

# Initialize DISBA model and compute phase velocities
pd = PhaseDispersion(thickness, vp, vs, rho)
disp0 = pd(periods, mode=0, wave="rayleigh")
disp1 = pd(periods, mode=1, wave="rayleigh")

# Reconvert to frequency axis
freqs_disba = 1/periods

# Assemble results in Raylee-compatible format
disba_velocities = np.concatenate([disp0.velocity, disp1.velocity])
disba_freqs = np.concatenate([freqs_disba, freqs_disba])
disba_modes = np.concatenate([
    np.ones_like(disp0.velocity),
    2 * np.ones_like(disp1.velocity)
])
disba_vtypes = np.zeros_like(disba_modes)

# Save to text files
np.savetxt("disba_frequency_values.txt", disba_freqs, fmt="%.5f")
np.savetxt("disba_velocity_values.txt", disba_velocities, fmt="%.5f")
np.savetxt("disba_mode_values.txt", disba_modes, fmt="%.5f")
np.savetxt("disba_vtype_values.txt", disba_vtypes, fmt="%.5f")

print("DISBA results saved as text files.")
