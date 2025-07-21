## raylee_lysmer.py
# 
# Python translation of: raylee_lysmer.m
# Based on: Haney, M. M., & Tsai, V. C. (2017), Geophysics, 82(3), F15-F28.
# Uses the finite element method (FEM) of Lysmer (1970) to compute
# Rayleigh/Scholte wave phase and group velocities and mode shapes.

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from .stoneley_velocity import stoneley_velocity

def raylee_lysmer(Nn, vsv, vpv, rhov, f, hv, modn, Nnf, vpfv, rhofv, hfv):
    """
    Compute Rayleigh/Scholte wave phase & group velocity and mode shapes
    using FEM (Lysmer, 1970).
    
    Parameters:
        Nn     : Number of solid elements
        vsv    : Shear velocity in solid (m/s)
        vpv    : Compressional velocity in solid (m/s)
        rhov   : Density in solid (kg/m^3)
        f      : Frequency (Hz)
        hv     : Solid element spacing (m)
        modn   : Mode number (1 = fundamental)
        Nnf    : Number of fluid elements
        vpfv   : Compressional velocity in fluid (m/s)
        rhofv  : Density in fluid (kg/m^3)
        hfv    : Fluid element spacing (m)

    Returns:
        kk   : Wavenumber
        vpk  : Phase velocity
        vgk  : Group velocity
        ev   : Eigenfunction (mode shape in solid)
    """
    # 
    omga = 2 * np.pi * f # Angular frequency
    Nnfo = Nnf + 1 if Nnf > 0 else 0 # Number of fluid elements + 1 for coupling
    ndof = Nnfo + 2 * Nn # Total number of degrees of freedom
    kappafv = rhofv * vpfv**2 # Fluid stiffness coefficients

    # Initialize global matrices
    Ka1 = sp.lil_matrix((ndof, ndof))
    Ka2 = sp.lil_matrix((ndof, ndof))
    Ka3 = sp.lil_matrix((ndof, ndof))
    M   = sp.lil_matrix((ndof, ndof))

    # === Fluid region matrix assembly ===
    if Nnf > 0:
        for ii in range(Nnf):
            # Grab grid interval and material properties
            h = hfv[ii]
            rhof = rhofv[ii]
            kappaf = kappafv[ii]

            # FEM coefficients (Lysmer)
            alph = 1 / (6 * rhof)
            bet = 1 / (6 * rhof)

            # Local elemental matrices (2x2)
            M1 = np.zeros((2, 2))
            L1 = np.zeros((2, 2))
            L3 = np.zeros((2, 2))

            # Mass matrix
            M1[0, 0] = h / (2 * kappaf)
            M1[1, 1] = M1[0, 0]

            # Stiffness matrices
            L1[0, 0] = 2 * alph * h
            L1[0, 1] = alph * h
            L1[1, 0] = alph * h
            L1[1, 1] = L1[0, 0]

            L3[0, 0] = 6 * bet / h
            L3[0, 1] = -6 * bet / h
            L3[1, 0] = -6 * bet / h
            L3[1, 1] = L3[0, 0]

            # Assemble into global matrices
            idx = slice(ii, ii + 2)
            M[idx, idx] += M1
            Ka1[idx, idx] += L1
            Ka3[idx, idx] += L3

        # Double the first entry
        M[0, 0] *= 2
        Ka1[0, 0] *= 2
        Ka3[0, 0] *= 2

    # === Solid region matrix assembly ===
    muv = rhov * vsv**2
    lamdav = rhov * vpv**2 - 2 * muv

    for ii in range(Nn):
        h = hv[ii]
        mu = muv[ii]
        lamda = lamdav[ii]
        rho = rhov[ii]

        # FEM coefficients
        alph = ((2 * mu) + lamda) / 6
        bet = mu / 6
        theta = (mu + lamda) / 4
        psi = (mu - lamda) / 4

        # Local elemental matrices (4x4)
        M1 = np.diag([h*rho/2]*4)
        L1 = np.zeros((4, 4))
        L2 = np.zeros((4, 4))
        L3 = np.zeros((4, 4))

        # L1 matrix (1st order stiffness)
        L1[0, 0] = 2*alph*h; L1[0, 2] = alph*h
        L1[1, 1] = 2*bet*h;  L1[1, 3] = bet*h
        L1[2, 0] = alph*h;   L1[2, 2] = 2*alph*h
        L1[3, 1] = bet*h;    L1[3, 3] = 2*bet*h

        # L2 matrix (coupled stiffness)
        L2[0, 1] = 2*psi;  L2[0, 3] = 2*theta
        L2[1, 0] = 2*psi;  L2[1, 2] = -2*theta
        L2[2, 1] = -2*theta; L2[2, 3] = -2*psi
        L2[3, 0] = 2*theta; L2[3, 2] = -2*psi

        # L3 matrix (3rd order stiffness)
        L3[0, 0] = 6*bet/h;  L3[0, 2] = -6*bet/h
        L3[1, 1] = 6*alph/h; L3[1, 3] = -6*alph/h
        L3[2, 0] = -6*bet/h; L3[2, 2] = 6*bet/h
        L3[3, 1] = -6*alph/h; L3[3, 3] = 6*alph/h

        idx = slice(Nnfo + 2*ii, Nnfo + 2*(ii+1))
        if ii == Nn - 1:
            M[idx, idx] += M1[:2, :2]
            Ka1[idx, idx] += L1[:2, :2]
            Ka2[idx, idx] += L2[:2, :2]
            Ka3[idx, idx] += L3[:2, :2]
        else:
            M[idx.start:idx.stop+2, idx.start:idx.stop+2] += M1
            Ka1[idx.start:idx.stop+2, idx.start:idx.stop+2] += L1
            Ka2[idx.start:idx.stop+2, idx.start:idx.stop+2] += L2
            Ka3[idx.start:idx.stop+2, idx.start:idx.stop+2] += L3

    # === Coupling matrix ===
    Cm = sp.lil_matrix((ndof, ndof))
    if Nnf > 0:
        Cm[Nnfo-1, Nnfo+2] = 1
        Cm[Nnfo+2, Nnfo-1] = 1

    # === Lower bound estimate for eigs ===
    if Nnf > 0:
        min_vs = np.min(vsv)
        min_vp = vpv[np.argmin(vsv)]
        min_vpf = np.min(vpfv)
        rho_f = rhofv[np.argmin(vpfv)]
        rho_s = rhov[np.argmin(vsv)]
        rspd = stoneley_velocity(min_vp, min_vs, min_vpf, rho_f, rho_s)
    else:
        min_vs = np.min(vsv)
        min_vp = vpv[np.argmin(vsv)]
        t1 = 1/(min_vs**6)
        t2 = -8/(min_vs**4)
        t3 = 24/(min_vs**2) - 16/(min_vp**2)
        t4 = -16 * (1 - (min_vs/min_vp)**2)
        roots = np.roots([t1, t2, t3, t4])
        rspd = np.sqrt(np.min(roots[np.isreal(roots)].real))

    # === Solve generalized eigenvalue problem ===
    Z = sp.csr_matrix((ndof, ndof))
    I = sp.identity(ndof)
    A = sp.bmat([[Z, I], [(omga**2*M - Ka3 - omga*Cm), Ka2]]).tocsr()
    B = sp.bmat([[I, Z], [Z, Ka1]]).tocsr()
    eigvals, eigvecs = spla.eigs(A, k=int(modn), M=B, sigma=omga/rspd, which='LM')

    d = eigvals[int(modn)-1]
    x = eigvecs[:, int(modn)-1]

    # === Normalize eigenfunction ===
    fctr = 1 / (x[:ndof].T @ (M @ x[:ndof]) - (x[:ndof].T @ (Cm @ x[:ndof])) / (2*omga))
    evp = np.real(x[:ndof]) * np.sqrt(fctr) * np.sign(x[Nnfo+1])
    ev = evp[Nnfo:]

    # === Extract phase, group velocities and wavenumber ===
    kk = d.real
    vpk = omga / kk
    vgk_num = x[:ndof].T @ ((2*kk*Ka1 - Ka2) @ x[:ndof])
    vgk_den = 2*omga * (x[:ndof].T @ M @ x[:ndof]) - (x[:ndof].T @ Cm @ x[:ndof])
    vgk = np.real(vgk_num / vgk_den)

    return kk, vpk, vgk, ev

# # raylee_lysmer.py
# # Exact MATLAB-to-Python translation of raylee_lysmer.m without simplification

# import numpy as np
# import scipy.sparse as sp
# import scipy.sparse.linalg as spla
# from .stoneley_velocity import stoneley_velocity

# def raylee_lysmer(Nn, vsv, vpv, rhov, f, hv, modn, Nnf, vpfv, rhofv, hfv):
#     omga = 2 * np.pi * f

#     if Nnf > 0:
#         Nnfo = Nnf + 1
#     else:
#         Nnfo = 0

#     kappafv = rhofv * vpfv**2

#     ndof = Nnfo + 2 * Nn
#     Ka1 = sp.lil_matrix((ndof, ndof))
#     Ka2 = sp.lil_matrix((ndof, ndof))
#     Ka3 = sp.lil_matrix((ndof, ndof))
#     M = sp.lil_matrix((ndof, ndof))

#     for ii in range(Nnf):
#         h = hfv[ii]
#         rhof = rhofv[ii]
#         kappaf = kappafv[ii]

#         alph = 1 / (6 * rhof)
#         bet = 1 / (6 * rhof)

#         M1 = sp.lil_matrix((2, 2))
#         L1 = sp.lil_matrix((2, 2))
#         L3 = sp.lil_matrix((2, 2))

#         M1[0, 0] = h / (2 * kappaf)
#         M1[1, 1] = h / (2 * kappaf)

#         L1[0, 0] = 2 * alph * h
#         L1[0, 1] = alph * h
#         L1[1, 0] = alph * h
#         L1[1, 1] = L1[0, 0]

#         L3[0, 0] = 6 * bet / h
#         L3[0, 1] = -6 * bet / h
#         L3[1, 0] = -6 * bet / h
#         L3[1, 1] = L3[0, 0]

#         idx = slice(ii, ii + 2)
#         M[idx, idx] += M1
#         Ka1[idx, idx] += L1
#         Ka3[idx, idx] += L3

#     if Nnf > 0:
#         M[0, 0] *= 2
#         Ka1[0, 0] *= 2
#         Ka3[0, 0] *= 2

#     muv = rhov * vsv**2
#     lamdav = rhov * vpv**2 - 2 * muv

#     for ii in range(Nn):
#         h = hv[ii]
#         mu = muv[ii]
#         lamda = lamdav[ii]
#         rho = rhov[ii]

#         alph = ((2 * mu) + lamda) / 6
#         bet = mu / 6
#         theta = (mu + lamda) / 4
#         psi = (mu - lamda) / 4

#         M1 = sp.lil_matrix((4, 4))
#         L1 = sp.lil_matrix((4, 4))
#         L2 = sp.lil_matrix((4, 4))
#         L3 = sp.lil_matrix((4, 4))

#         for i in range(4):
#             M1[i, i] = h * rho / 2

#         L1[0, 0] = 2 * alph * h; L1[0, 2] = alph * h
#         L1[1, 1] = 2 * bet * h;  L1[1, 3] = bet * h
#         L1[2, 0] = alph * h;     L1[2, 2] = 2 * alph * h
#         L1[3, 1] = bet * h;      L1[3, 3] = 2 * bet * h

#         L2[0, 1] = 2 * psi;   L2[0, 3] = 2 * theta
#         L2[1, 0] = 2 * psi;   L2[1, 2] = -2 * theta
#         L2[2, 1] = -2 * theta; L2[2, 3] = -2 * psi
#         L2[3, 0] = 2 * theta; L2[3, 2] = -2 * psi

#         L3[0, 0] = 6 * bet / h;   L3[0, 2] = -6 * bet / h
#         L3[1, 1] = 6 * alph / h; L3[1, 3] = -6 * alph / h
#         L3[2, 0] = -6 * bet / h; L3[2, 2] = 6 * bet / h
#         L3[3, 1] = -6 * alph / h; L3[3, 3] = 6 * alph / h

#         idx = Nnfo + 2 * ii
#         if ii == Nn - 1:
#             M[idx:idx+2, idx:idx+2] += M1[:2, :2]
#             Ka1[idx:idx+2, idx:idx+2] += L1[:2, :2]
#             Ka2[idx:idx+2, idx:idx+2] += L2[:2, :2]
#             Ka3[idx:idx+2, idx:idx+2] += L3[:2, :2]
#         else:
#             M[idx:idx+4, idx:idx+4] += M1
#             Ka1[idx:idx+4, idx:idx+4] += L1
#             Ka2[idx:idx+4, idx:idx+4] += L2
#             Ka3[idx:idx+4, idx:idx+4] += L3

#     Cm = sp.lil_matrix((ndof, ndof))
#     if Nnf > 0:
#         Cm[Nnfo-1, Nnfo+2] = 1
#         Cm[Nnfo+2, Nnfo-1] = 1

#     if Nnf > 0:
#         min_vs = np.min(vsv)
#         min_vp = vpv[np.argmin(vsv)]
#         min_vpf = np.min(vpfv)
#         rho_f = rhofv[np.argmin(vpfv)]
#         rho_s = rhov[np.argmin(vsv)]
#         rspd = stoneley_velocity(min_vp, min_vs, min_vpf, rho_f, rho_s)
#     else:
#         min_vs = np.min(vsv)
#         min_vp = vpv[np.argmin(vsv)]
#         t1 = 1/(min_vs**6)
#         t2 = -8/(min_vs**4)
#         t3 = 24/(min_vs**2) - 16/(min_vp**2)
#         t4 = -16 * (1 - (min_vs/min_vp)**2)
#         roots = np.roots([t1, t2, t3, t4])
#         rspd = np.sqrt(np.min(roots[np.isreal(roots)].real))

#     Z = sp.csr_matrix((ndof, ndof))
#     I = sp.identity(ndof)
#     A = sp.bmat([[Z, I], [(omga**2*M - Ka3 - omga*Cm), Ka2]]).tocsr()
#     B = sp.bmat([[I, Z], [Z, Ka1]]).tocsr()

#     eigvals, eigvecs = spla.eigs(A, k=modn, M=B, sigma=omga/rspd, which='LM')
#     d = eigvals[modn-1]
#     x = eigvecs[:, modn-1]

#     fctr = 1 / (x[:ndof].T @ (M @ x[:ndof]) - (x[:ndof].T @ (Cm @ x[:ndof])) / (2*omga))
#     evp = np.real(x[:ndof]) * np.sqrt(fctr) * np.sign(x[Nnfo+1])
#     ev = evp[Nnfo:]

#     kk = d.real
#     vpk = omga / kk
#     vgk = ((x[:ndof].T @ ((2 * kk * Ka1 - Ka2) @ x[:ndof])) /
#            (2 * omga * (x[:ndof].T @ M @ x[:ndof]) - x[:ndof].T @ Cm @ x[:ndof])).real

#     return kk, vpk, vgk, ev
