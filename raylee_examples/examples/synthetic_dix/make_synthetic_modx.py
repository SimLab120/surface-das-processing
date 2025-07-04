# =============================================================================
# make_synthetic_modx.py
#
# This script generates synthetic Rayleigh-wave phase and group velocity data
# for the Xia et al. (1999) "modx" model using the Lysmer FEM method, based on
# Haney and Tsai (2017), Geophysics. It also computes velocities using
# DISBA for comparison.
#
# Output files:
#   - modx_phase_vels.txt
#   - modx_group_vels.txt
#   - modx_freqs.txt
#   - disba_phase_vels.txt
#   - disba_freqs.txt
# =============================================================================

import numpy as np
from raylee_processing.forward.raylee_lysmer import raylee_lysmer

# -----------------------------
# 1. Construct the model grid
# -----------------------------
h = np.concatenate([0.1 * np.ones(200), 2.0 * np.ones(115)])  # Grid spacing in meters
Nn = len(h)  # Number of solid elements (nodes)

# -----------------------------
# 2. Frequency and mode setup
# -----------------------------
fks = np.arange(5, 30.1, 0.5)  # Frequencies (Hz)
modn = 1  # Fundamental mode only

# -----------------------------
# 3. No water layer on top
# -----------------------------
Nnf = 0
vpfv = 1
rhofv = 1
hfv = 1

# -----------------------------
# 4. Define the 6-layer earth model
# -----------------------------
layrth1, layrth2, layrth3, layrth4, layrth5 = 20, 23, 25, 28, 32

vplay = [650, 750, 1400, 1800, 2150, 2800]  # P-wave velocities in m/s
vslay = [194, 270, 367, 485, 603, 740]      # S-wave velocities in m/s
rholay = [1820, 1860, 1910, 1960, 2020, 2090]  # Densities in kg/m^3

vpv = np.concatenate([
    vplay[0] * np.ones(layrth1),
    vplay[1] * np.ones(layrth2),
    vplay[2] * np.ones(layrth3),
    vplay[3] * np.ones(layrth4),
    vplay[4] * np.ones(layrth5),
    vplay[5] * np.ones(Nn - sum([layrth1, layrth2, layrth3, layrth4, layrth5]))
])

vsv = np.concatenate([
    vslay[0] * np.ones(layrth1),
    vslay[1] * np.ones(layrth2),
    vslay[2] * np.ones(layrth3),
    vslay[3] * np.ones(layrth4),
    vslay[4] * np.ones(layrth5),
    vslay[5] * np.ones(Nn - sum([layrth1, layrth2, layrth3, layrth4, layrth5]))
])

rhov = np.concatenate([
    rholay[0] * np.ones(layrth1),
    rholay[1] * np.ones(layrth2),
    rholay[2] * np.ones(layrth3),
    rholay[3] * np.ones(layrth4),
    rholay[4] * np.ones(layrth5),
    rholay[5] * np.ones(Nn - sum([layrth1, layrth2, layrth3, layrth4, layrth5]))
])

# -----------------------------
# 5. Forward modeling loop (Raylee)
# -----------------------------
vp = []  # Phase velocities
U = []   # Group velocities

for f in fks:
    kk, vpk, vgk, ev = raylee_lysmer(Nn, vsv, vpv, rhov, f, h, modn, Nnf, vpfv, rhofv, hfv)
    vp.append(vpk)
    U.append(vgk)

vp = np.array(vp)
U = np.array(U)

# --------------------------------------
# 6. Optionally add 2% Gaussian noise
# --------------------------------------
add_noise = False  # Set to False if noise is not needed

if add_noise:
    np.random.seed(0)
    noise = 0.02 * np.random.randn(len(vp))
    vp *= (1 + noise)
    U *= (1 + noise)

# -----------------------------
# 7. Save Raylee output
# -----------------------------
np.savetxt("modx_phase_vels.txt", vp, fmt="%.5f")
np.savetxt("modx_group_vels.txt", U, fmt="%.5f")
np.savetxt("modx_freqs.txt", fks, fmt="%.5f")

print("Synthetic modx data saved.")

# =============================================================================
# 8. Also compute phase and group velocities using DISBA
# =============================================================================
print("Computing DISBA phase and group velocities...")

from disba import PhaseDispersion, GroupDispersion

thickness = h
vp_disba = vpv
vs_disba = vsv
rho_disba = rhov

periods = np.sort(1.0 / fks)  # Convert frequencies to periods  

# Phase velocities
phase_disp = PhaseDispersion(thickness, vp_disba, vs_disba, rho_disba)
vp_disba_out = phase_disp(periods, mode=0, wave="rayleigh").velocity

# Group velocities
group_disp = GroupDispersion(thickness, vp_disba, vs_disba, rho_disba)
U_disba_out = group_disp(periods, mode=0, wave="rayleigh").velocity

fks = 1.0 / periods  # Convert periods back to frequencies

# Save DISBA results
np.savetxt("disba_phase_vels.txt", vp_disba_out, fmt="%.5f")
np.savetxt("disba_group_vels.txt", U_disba_out, fmt="%.5f")
np.savetxt("disba_freqs.txt", fks, fmt="%.5f")

print("DISBA results saved as text files.")
