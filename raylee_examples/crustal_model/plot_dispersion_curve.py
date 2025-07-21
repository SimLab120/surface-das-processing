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

def plot_dispersion_comparison(mode="comparison", save_fig=False, outname="dispersion_comparison.png"):
    """
    Plot Raylee, DISBA, or both dispersion curves for comparison.

    Parameters:
        mode (str): "raylee" for only Raylee, "comparison" for both.
        save_fig (bool): Whether to save the figure.
        outname (str): Output filename if saving figure.
    """
    if mode not in ["raylee", "comparison"]:
        raise ValueError("mode must be 'raylee' or 'comparison'")

    # Load Raylee data
    freqs_raylee, vel_raylee, modes_raylee, vtypes_raylee = load_dispersion_data(prefix="")

    # Load DISBA data only if comparison
    if mode == "comparison":
        freqs_disba, vel_disba, modes_disba, vtypes_disba = load_dispersion_data(prefix="disba_")

    plt.figure(figsize=(10, 6))

    if mode == "comparison":
        disba_plotted = False
        for mode_id in np.unique(modes_disba):
            idx = modes_disba == mode_id
            label = "DISBA Model" if not disba_plotted else None
            plt.plot(freqs_disba[idx], vel_disba[idx], "-", color="blue", linewidth=2, label=label)
            disba_plotted = True

    raylee_plotted = False
    for mode_id in np.unique(modes_raylee):
        idx = modes_raylee == mode_id
        label = "Raylee Model" if not raylee_plotted else None
        plt.plot(freqs_raylee[idx], vel_raylee[idx], "-.", color="red", linewidth=2, label=label)
        raylee_plotted = True

    # Add text annotations for modes (optional)
    if mode == "comparison":
        mid_freq = 0.4
        offset = 30
        if 1 in modes_disba:
            plt.text(mid_freq, np.mean(vel_disba[modes_disba == 1]) + offset, "Mode 1", color="gray")
        if 2 in modes_disba:
            plt.text(mid_freq, np.mean(vel_disba[modes_disba == 2]) + offset, "Mode 2", color="gray")

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Rayleigh-Wave Dispersion")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_fig:
        plt.savefig(outname, dpi=300)
        print(f"Figure saved as: {outname}")
    else:
        plt.show()

if __name__ == "__main__":
    # Option 1: Only Raylee
    plot_dispersion_comparison(mode="raylee")

    # Option 2: Comparison with DISBA
    # plot_dispersion_comparison(mode="comparison")

