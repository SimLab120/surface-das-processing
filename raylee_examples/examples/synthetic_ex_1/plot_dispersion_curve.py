# =============================================================================
# plot_dispersion_curve.py
#
# This script reads frequency, velocity, mode, and velocity type data
# from both Raylee and DISBA outputs and plots their dispersion curves.
#
# Usage: Run this after running make_synthetic_ex1.py
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import os

def load_dispersion_data(directory=".", prefix=""):
    """
    Load dispersion data files from the specified directory.
    `prefix` can be "" (for Raylee) or "disba_" (for DISBA).
    """
    freq_path = os.path.join(directory, f"{prefix}frequency_values.txt")
    vel_path = os.path.join(directory, f"{prefix}velocity_values.txt")
    mode_path = os.path.join(directory, f"{prefix}mode_values.txt")
    vtype_path = os.path.join(directory, f"{prefix}vtype_values.txt")

    freqs = np.loadtxt(freq_path)
    velocities = np.loadtxt(vel_path)
    modes = np.loadtxt(mode_path)
    vtypes = np.loadtxt(vtype_path)

    return freqs, velocities, modes, vtypes

def plot_dispersion_comparison(save_fig=False, outname="dispersion_comparison.png"):
    """Plot both Raylee and DISBA dispersion curves for comparison."""
    # Load both datasets
    freqs_raylee, vel_raylee, modes_raylee, vtypes_raylee = load_dispersion_data(prefix="")
    freqs_disba, vel_disba, modes_disba, vtypes_disba = load_dispersion_data(prefix="disba_")

    plt.figure(figsize=(10, 6))

    # Plot DISBA 
    disba_plotted = False
    for mode in np.unique(modes_disba):
        idx = modes_disba == mode
        label = "DISBA Model" if not disba_plotted else None
        style = "-" 
        plt.plot(freqs_disba[idx], vel_disba[idx], style, color="blue", linewidth=2, label=label)
        disba_plotted = True

    # Plot Raylee 
    raylee_plotted = False
    for mode in np.unique(modes_raylee):
        idx = modes_raylee == mode
        vtype = "Phase" if np.all(vtypes_raylee[idx] == 0) else "Group"
        label = "Raylee Model" if not raylee_plotted else None
        style = "-."
        plt.plot(freqs_raylee[idx], vel_raylee[idx], style, color="red", linewidth=2, label=label)
        raylee_plotted = True


    # Add text annotations for modes
    mid_freq = 0.4
    offset = 30
    if 1 in modes_disba:
        plt.text(mid_freq, np.mean(vel_disba[modes_disba == 1]) + offset, "Mode 1", color="gray")
    if 2 in modes_disba:
        plt.text(mid_freq, np.mean(vel_disba[modes_disba == 2]) + offset, "Mode 2", color="gray")

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Rayleigh-Wave Dispersion: Raylee vs. DISBA")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_fig:
        plt.savefig(outname, dpi=300)
        print(f"Figure saved as: {outname}")
    else:
        plt.show()

if __name__ == "__main__":
    plot_dispersion_comparison()
