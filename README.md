# surface-das-processing

This repository contains a modular Python implementation (adapted from Matlab) of the **Rayleigh-wave forward modeling** and synthetic data generation framework described in:

> Haney, M. M., & Tsai, V. C. (2017). *Perturbational and nonperturbational inversion of Rayleigh-wave velocities*, Geophysics, 82(3), F15-F28. [https://doi.org/10.1190/geo2016-0397.1]

We generate and visualize synthetic Rayleigh-wave dispersion data using the finite element method of Lysmer (1970), and to compare it against reference results from the **DISBA** package.

---

## 📦 Installation

## ⚙️ Environment Setup

We strongly recommend using a Python virtual environment:

```bash
# Create and activate a virtual environment
python -m venv das-env
source das-env/bin/activate

git clone https://github.com/SimLab120/surface-das-processing.git
cd surface-das-processing
pip install -e .
```

📁 Project Structure


```
.
surface-das-processing/
│
├── dix_inversion/              # Main inversion package
│   ├── __init__.py
│   ├── linvers.py
│   ├── raylee_sensitivity.py
│   ├── ...
│
├── raylee_examples/            # Synthetic and real data examples
│   ├── dix_model/
│   ├── crustal_model/
│
├── setup.py                    # Installation script
└── README.md                   # This file

```

---

## 📥 Importing the Inversion Module

To access all required functions for Rayleigh wave inversion in your scripts, simply import the main inversion package:
```
import dix_inversion as dix

sensitivity = dix.sensitivity(vp, vs, .....)
```
This will give you direct access to core functionalities including:

raylee_sensitivity — Forward modeling using Lysmer’s method

linvers — Damped least squares inversion solver

compute_dix_kernel, compute_homogeneous_dix_kernels — Non-perturbational kernel computation

compute_xia_interpolation — Interpolation of shear velocity for Dix-based inversion

stoneley_velocity — Optional Stoneley mode calculation

check_nans, update_covariances, compute_misfit — Utility tools for robust inversion



---

# 🔬 Example Usage: Synthetic Inversion Workflows

The `raylee_examples/` directory contains complete workflows for **non-perturbational (Dix-based)** and **perturbational (Tarantola–Valette)** Rayleigh wave inversions.

---

## 📁 1. Dix-based (Non-Perturbational) Inversion  
**Folder**: `raylee_examples/dix_model/`

### 🛠 Step-by-Step

```bash
cd raylee_examples/dix_model
```

**Generate synthetic phase velocities:**

```bash
python make_synthetic_dix_data.py
```

**Generate initial model:**

```bash
python make_initial_model_dix.py
```

**Run the inversion:**

```bash
python raylee_invert.py
```

---

## 📁 2. Perturbational Inversion  
**Folder**: `raylee_examples/crustal_model/`

### 🛠 Step-by-Step

```bash
cd raylee_examples/crustal_model
```

**Generate synthetic phase velocities:**

```bash
python make_synthetic_ex1.py
```

**Generate initial model:**

```bash
python make_initial_model_ex1.py
```

**Run the inversion:**

```bash
python taylee_invert.py
```

---

## 📈 Output

- Velocity-depth plots (`Vs_true`, `Vs_initial`, `Vs_inverted`)
- Phase velocity misfit plots
- Chi-squared and RMS evolution over iterations
- Posterior covariance and resolution matrix (optional)

---

## 📎 Notes

- All input/output files are stored within each example folder.
- Models can be adjusted by editing the corresponding Python scripts.
- For custom studies, you may alter the prior assumptions.



