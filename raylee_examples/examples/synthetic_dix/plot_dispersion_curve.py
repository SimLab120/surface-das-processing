# =============================================================================
# plot_modx_dispersion.py
#
# This script compares Rayleigh-wave dispersion curves (phase & group velocity)
# from the modx model (Lysmer FEM method) and DISBA outputs.
#
# Output: Dispersion comparison plot
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

def load_data():
    """Load dispersion data from text files."""
    freqs_modx = np.loadtxt("modx_freqs.txt")
    vp_modx = np.loadtxt("modx_phase_vels.txt")
    vg_modx = np.loadtxt("modx_group_vels.txt")

    freqs_disba = np.loadtxt("disba_freqs.txt")
    vp_disba = np.loadtxt("disba_phase_vels.txt")
    vg_disba = np.loadtxt("disba_group_vels.txt")

    return freqs_modx, vp_modx, vg_modx, freqs_disba, vp_disba, vg_disba

def plot_dispersion_comparison(save_fig=False, outname="modx_disba_dispersion.png"):
    """Plot phase and group velocity dispersion curves for MODX vs DISBA."""
    freqs_modx, vp_modx, vg_modx, freqs_disba, vp_disba, vg_disba = load_data()

    plt.figure(figsize=(10, 6))

    # MODX (Lysmer FEM)
    plt.plot(freqs_modx, vp_modx, 'r.-.', label="MODX Phase", linewidth=2)
    plt.plot(freqs_modx, vg_modx, 'r.-', label="MODX Group", linewidth=2)

    # DISBA
    plt.plot(freqs_disba, vp_disba, 'b-', label="DISBA Phase", linewidth=2)
    plt.plot(freqs_disba, vg_disba, 'b-', label="DISBA Group", linewidth=2)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Rayleigh-Wave Dispersion: MODX vs DISBA (Fundamental Mode)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_fig:
        plt.savefig(outname, dpi=300)
        print(f"Plot saved as {outname}")
    else:
        plt.show()

if __name__ == "__main__":
    plot_dispersion_comparison()
