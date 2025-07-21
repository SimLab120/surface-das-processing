"""
Program: make_initial_model_ex1.py

Purpose:
Generate an initial model for iterative Rayleigh/Scholte wave
phase or group velocity inversion.

Inputs:
- velocity_values.txt (from make_synthetic_ex1.py)
- velocity_values_errs.txt (from make_synthetic_ex1.py)

Outputs:
- vp_init.txt, vs_init.txt, rho_init.txt
- vpf.txt, rhof.txt
- grid_values_solid.txt, grid_values_fluid.txt
- input_params.txt
"""

import numpy as np
import os

# Create output folder if not exists
output_dir = 'initial_model_output'
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Parameters
chilo = 1.0
chihi = 1.5
nupdats = 16
mstdfact = 2.0
smscl = 1000.0
pratioflag = 1

# Load number of frequency measurements
velocity_values = np.loadtxt( 'velocity_values.txt')
Nf = len(velocity_values)
frequency_values = np.loadtxt( 'frequency_values.txt')
velocity_values_errs = np.loadtxt('velocity_values_errs.txt')
vtype_values = np.loadtxt('vtype_values.txt')
mode_values = np.loadtxt('mode_values.txt')

# Solid grid
Nn = 240
h = 250.0 * np.ones(Nn)

# Fluid grid
Nnf = 0
hfv = 100.0 * np.ones(Nnf)

# Constants
vpvsr = 1.7321
gardc = 309.6
powr = 0.25

# Homogeneous solid model
vslay1 = 3400.0
vplay1 = vslay1 * vpvsr
rholay1 = gardc * (vplay1 ** powr)

vpv = vplay1 * np.ones(Nn)
vsv = vslay1 * np.ones(Nn)
rhov = rholay1 * np.ones(Nn)

# Fluid model
vplay4 = 1500.0
rholay4 = 1000.0
vpvf = vplay4 * np.ones(Nnf)
rhovf = rholay4 * np.ones(Nnf)

# -----------------------------
# Write input parameters (no headers, comment lines allowed)
with open(os.path.join(output_dir, 'input_params.txt'), 'w') as f:
    f.write("%% input parameters for Rayleigh/Scholte wave inversion\n\n")
    f.write(f"{pratioflag}\n")
    f.write(f"{smscl:10.5f}\n")
    f.write(f"{mstdfact:10.5f}\n")
    f.write(f"{nupdats}\n")
    f.write(f"{Nf}\n")
    f.write(f"{Nn}\n")
    f.write(f"{Nnf}\n")
    f.write(f"{chilo:10.5f}\n")
    f.write(f"{chihi:10.5f}\n")

# -----------------------------
# Save helper
def save_vector(filename, array):
    np.savetxt(os.path.join(output_dir, filename), array, fmt='%10.5f')

# Vp
if pratioflag == 1:
    save_vector('vp_init.txt', np.array([vpvsr]))
else:
    save_vector('vp_init.txt', vpv)

# Vs, Density (solid), Vp/Density (fluid), Grid thicknesses
save_vector('vs_init.txt', vsv)
save_vector('rho_init.txt', rhov)
save_vector('vpf.txt', vpvf)
save_vector('rhof.txt', rhovf)
save_vector('grid_values_solid.txt', h)
save_vector('grid_values_fluid.txt', hfv)
save_vector('velocity_values.txt', velocity_values)
save_vector('frequency_values.txt', frequency_values)
save_vector('velocity_values_errs.txt', velocity_values_errs)
save_vector('vtype_values.txt', vtype_values)
save_vector('mode_values.txt', mode_values)