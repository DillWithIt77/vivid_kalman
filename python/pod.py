"""
pod.py

Proper Orthogonal Decomposition (Eq. 29-34) and MATLAB (-v7.3 / HDF5)
snapshot loading.
"""

import numpy as np
import h5py


def load_state_snapshots(mat_path):
    """
    Load a state_snapshots.mat file written by export_snapshots.m
    (saved with -v7.3, i.e. HDF5). Returns (X, tsave) where X has shape
    (n_tsave, N, N) [MATLAB stores column-major, so we transpose axes
    accordingly] and tsave is (n_tsave,).
    """
    with h5py.File(mat_path, "r") as f:
        X = np.array(f["X"])          # h5py gives (n_tsave, N, N) already
                                        # reversed vs MATLAB's (N,N,n_tsave)
        tsave = np.array(f["tsave"]).ravel()
    return X, tsave


def compute_pod_basis(X_train, q):
    """
    Eq. 29-34: given a stack of training snapshots X_train (n_state, N, N),
    flatten, SVD, and keep the first q modes.

    Returns
    -------
    L_Xq : (N*N, q) POD basis (paper's L_{X,q})
    energy_ratio : float, gamma_x from Eq. 34 for this q
    singular_values : (min(n_state, N*N),) array, for choosing q
    """
    n_state = X_train.shape[0]
    N = X_train.shape[1]
    Xflat = X_train.reshape(n_state, N * N).T  # (N*N, n_state), Eq. 29's X

    # SVD instead of forming the covariance matrix directly (Eq. 30 note)
    U, S, _ = np.linalg.svd(Xflat, full_matrices=False)
    L_Xq = U[:, :q]

    energy_ratio = np.sum(S[:q] ** 2) / np.sum(S ** 2)
    return L_Xq, energy_ratio, S


def project(x_flat, L_Xq):
    """Eq. 32: x_hat = L_Xq^T @ x."""
    return L_Xq.T @ x_flat


def reconstruct(x_hat, L_Xq):
    """Eq. 33: x_r = L_Xq @ x_hat."""
    return L_Xq @ x_hat
