
# raylee_processing/inversion/linvers.py

import numpy as np
from scipy.sparse import vstack, csc_matrix
from scipy.sparse.linalg import lsqr

def linvers(U_data, U, snsmf_vstot, mcmisr, dcmisr, Nn, vsv, vsg):
    """
    Weighted damped least squares inversion using LSQR.
    Faithful translation of MATLAB linvers.m (Haney & Tsai, 2017).

    Parameters
    ----------
    U_data : ndarray (n_data,)
        Observed Rayleigh-wave velocity data
    U : ndarray (n_data,)
        Modeled velocity from forward simulation
    snsmf_vstot : ndarray (n_data, Nn)
        Sensitivity (Jacobian) matrix of velocities w.r.t. Vs
    mcmisr : ndarray (Nn, Nn)
        Inverse square root of model covariance matrix
    dcmisr : ndarray (n_data, n_data)
        Inverse square root of data covariance matrix
    Nn : int
        Number of unknown model parameters (Vs depth points)
    vsv : ndarray (Nn,)
        Current shear wave velocity model
    vsg : ndarray (Nn,)
        Initial guess model

    Returns
    -------
    dvs : ndarray (Nn,)
        Model update to be added to vsv
    """

    # 1. Transpose Jacobian (G in MATLAB is [Nn x Nf])
    G = snsmf_vstot.T  # Now shape: (Nn, Nf)

    # 2. Model difference (vsv - vsg): shape (Nn,)
    delta_v = vsv - vsg

    # 3. Data misfit (U_data - U + G*(vsv - vsg)) : shape (Nf,)
    duv = (U_data - U) + G @ delta_v  # shape: (Nf,)

    # 4. Scale Jacobian and data residual by inverse sqrt data covariance
    Gs = dcmisr @ G        # shape: (Nf, Nn)
    duvs = dcmisr @ duv    # shape: (Nf,)

    # 5. Stack matrices and RHS for regularized least squares
    A = np.vstack([Gs, mcmisr])   # shape: (Nf + Nn, Nn)
    b = np.concatenate([duvs, np.zeros(Nn)])  # shape: (Nf + Nn,)

    # 6. Convert A to sparse CSC format for efficient LSQR
    A_sparse = csc_matrix(A)

    # 7. Solve using LSQR (MATLAB uses tol=1e-2)
    result = lsqr(A_sparse, b, atol=1e-2, btol=1e-2)
    dvs = result[0]  # solution vector (Nn,)

    return dvs
