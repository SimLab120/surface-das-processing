# surface-das-processing

This repository contains a modular Python implementation (adapted from Matlab) of the **Rayleigh-wave forward modeling** and synthetic data generation framework described in:

> Haney, M. M., & Tsai, V. C. (2017). *Perturbational and nonperturbational inversion of Rayleigh-wave velocities*, Geophysics, 82(3), F15-F28. [https://doi.org/10.1190/geo2016-0397.1]

We generate and visualize synthetic Rayleigh-wave dispersion data using the finite element method of Lysmer (1970), and to compare it against reference results from the **DISBA** package.

---

## 📁 Project Structure

```text
.
├── raylee_examples/
│   └── examples/synthetic_ex_1/
│       ├── make_synthetic_ex1.py          # Generates synthetic dispersion data using Raylee
│       ├── plot_dispersion_curve.py       # Plots and compares Raylee vs DISBA dispersion curves
│       ├── frequency_values.txt           # Frequency values used in simulation
│       ├── velocity_values.txt            # Raylee-computed phase/group velocities
│       ├── disba_velocity_values.txt      # DISBA-computed dispersion values
│       └── ...
├── raylee_processing/
│   ├── forward/
│   │   ├── raylee_lysmer.py               # Core FEM solver for Rayleigh wave speeds
│   │   └── stoneley_velocity.py           # Computes Stoneley-wave velocity (used internally)
│   └── __init__.py
├── pyproject.toml
└── README.md
