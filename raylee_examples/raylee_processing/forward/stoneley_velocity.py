"""
Stoneley wave velocity solver using the method from:

Haney, M. M., & Tsai, V. C. (2017). Perturbational and nonperturbational
inversion of Rayleigh-wave velocities. Geophysics, 82(3), F15–F28.
https://doi.org/10.1190/geo2016-0397.1

This function computes the fundamental mode of the guided wave
(Stoneley wave) for a two-layer model: water over a solid half-space.

The velocity is determined by solving an 8th-degree polynomial in
1/v² and selecting the physically meaningful root.

Stoneley waves are relevant for modeling seismic interface waves
in marine and fluid-solid settings.

Author: Ported and extended from MATLAB version (Haney & Tsai, 2017)
"""

import numpy as np

def stoneley_velocity(vp_solid, vs_solid, vp_fluid, rho_fluid, rho_solid):
    """
    Compute the Stoneley wave velocity for a water-over-solid interface.

    Parameters:
        vp_solid  (float): P-wave velocity in solid (a)
        vs_solid  (float): S-wave velocity in solid (b)
        vp_fluid  (float): P-wave velocity in fluid (c)
        rho_fluid (float): Density of fluid (f)
        rho_solid (float): Density of solid (s)

    Returns:
        vst (float): Stoneley wave phase velocity (m/s)
    """

    # Rename variables to match original paper notation
    a = vp_solid
    b = vs_solid
    c = vp_fluid
    f = rho_fluid
    s = rho_solid

    # === Step 1: Construct the 8th-degree polynomial coefficients ===
    # The polynomial is in terms of (1/v²), derived from interface wave theory.

    c16 = 256*b**16 - (512*b**18)/a**2 + (256*b**20)/a**4

    c14 = (-768*b**14 + (1280*b**16)/a**2 - (512*b**18)/a**4
           - (512*b**16)/c**2 + (1024*b**18)/(a**2 * c**2)
           - (512*b**20)/(a**4 * c**2))

    c12 = (832*b**12 - (1024*b**14)/a**2 + (256*b**16)/a**4
           + (256*b**16)/c**4 - (512*b**18)/(a**2 * c**4)
           + (256*b**20)/(a**4 * c**4)
           + (1536*b**14)/c**2 - (2560*b**16)/(a**2 * c**2)
           + (1024*b**18)/(a**4 * c**2)
           - (64*b**12*f**2)/s**2)

    c10 = (-416*b**10 + (288*b**12)/a**2 - (768*b**14)/c**4
           + (1280*b**16)/(a**2 * c**4) - (512*b**18)/(a**4 * c**4)
           - (1664*b**12)/c**2 + (2048*b**14)/(a**2 * c**2)
           - (512*b**16)/(a**4 * c**2)
           + (96*b**10*f**2)/s**2 + (96*b**12*f**2)/(a**2 * s**2)
           + (64*b**12*f**2)/(c**2 * s**2))

    c8 = (112*b**8 - (32*b**10)/a**2 + (832*b**12)/c**4
          - (1024*b**14)/(a**2 * c**4) + (256*b**16)/(a**4 * c**4)
          + (832*b**10)/c**2 - (576*b**12)/(a**2 * c**2)
          - (48*b**8*f**2)/s**2 - (128*b**10*f**2)/(a**2 * s**2)
          - (32*b**12*f**2)/(a**4 * s**2)
          - (96*b**10*f**2)/(c**2 * s**2)
          - (96*b**12*f**2)/(a**2 * c**2 * s**2))

    c6 = (-16*b**6 - (416*b**10)/c**4 + (288*b**12)/(a**2 * c**4)
          - (224*b**8)/c**2 + (64*b**10)/(a**2 * c**2)
          + (16*b**6*f**2)/s**2 + (48*b**8*f**2)/(a**2 * s**2)
          + (32*b**10*f**2)/(a**4 * s**2) + (48*b**8*f**2)/(c**2 * s**2)
          + (128*b**10*f**2)/(a**2 * c**2 * s**2)
          + (32*b**12*f**2)/(a**4 * c**2 * s**2))

    c4 = (b**4 + (112*b**8)/c**4 - (32*b**10)/(a**2 * c**4)
          + (32*b**6)/c**2 + (b**4*f**4)/s**4 - (2*b**4*f**2)/s**2
          - (16*b**6*f**2)/(a**2 * s**2) - (16*b**6*f**2)/(c**2 * s**2)
          - (48*b**8*f**2)/(a**2 * c**2 * s**2)
          - (32*b**10*f**2)/(a**4 * c**2 * s**2))

    c2 = (-(16*b**6)/c**4 - (2*b**4)/c**2
          - (2*b**4*f**4)/(a**2 * s**4) + (2*b**4*f**2)/(a**2 * s**2)
          + (2*b**4*f**2)/(c**2 * s**2)
          + (16*b**6*f**2)/(a**2 * c**2 * s**2))

    c0 = ((b**4)/c**4 + (b**4*f**4)/(a**4 * s**4)
          - (2*b**4*f**2)/(a**2 * c**2 * s**2))

    # === Step 2: Solve polynomial for 1/v² and compute wave speeds ===

    vsi = np.roots([c16, c14, c12, c10, c8, c6, c4, c2, c0])  # roots of polynomial
    vs_all = np.sqrt(1.0 / vsi[np.isreal(vsi)].real)         # convert to wave speed (v)

    # === Step 3: Keep only physical roots (real and slower than fluid velocity) ===
    vs_filtered = vs_all[vs_all <= c]

    if len(vs_filtered) == 0:
        raise ValueError("No valid Stoneley roots found: try different material parameters.")

    # === Step 4: Evaluate the Stoneley dispersion condition to pick the best root ===
    def stoneley_condition(v):
        """
        Stoneley dispersion equation (simplified form).
        The correct root minimizes this residual.
        """
        b_over_v = b / v
        t1 = np.sqrt(b_over_v**2 - (b/c)**2)
        t2 = (1 - 2*b_over_v**2)**2 - 4*b_over_v**2 * np.sqrt(b_over_v**2 - (b/a)**2) * np.sqrt(b_over_v**2 - 1)
        t3 = (f/s) * np.sqrt(b_over_v**2 - (b/a)**2)
        return np.abs(t1 * t2 + t3)

    # Evaluate condition at all candidates and pick the minimum
    vst = vs_filtered[np.argmin([stoneley_condition(v) for v in vs_filtered])]

    return vst
