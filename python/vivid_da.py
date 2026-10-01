"""
vivid_da.py

Variational DA objective and minimization:
  - conventional 3D-Var (Eq. 1)               -> `da_3dvar`
  - VIVID (Eq. 18: background + inverse-op + observation term) -> `da_vivid`
  - POD/ROM variant (Eq. 37)                   -> `da_vivid_rom`

Rather than the paper's Algorithm 1/2 (manual linearization of H_t and
an approximate Hessian, Eq. 9-11), we exploit that everything here
(the QG observation operator H_t, and the VCNN) is differentiable in
PyTorch, so we get *exact* gradients via autograd and hand them to
SciPy's L-BFGS-B. This is mathematically the same 3D-Var / VIVID
objective, just minimized with exact instead of approximate gradients.

Observation operator H_t: for our QG setup we take the identity
sampling operator (Eq. 12): H_t(x)_k = x[i_t,k, j_t,k]. Swap in
`nonlinear_observation_field` from sensors.py inside `_evaluate_obs`
if you want to reproduce the paper's Eq. 42 operator instead.
"""

import numpy as np
import torch
from scipy.optimize import minimize
from sensors import nonlinear_observation_field_torch


# ---------------------------------------------------------------- metrics --
def r_rmse(x_est, x_true):
    """Relative RMSE, as used in the paper's Table 4 / Fig. 11."""
    num = np.linalg.norm(x_est - x_true)
    den = np.linalg.norm(x_true)
    return num / den


def ssim(x_est, x_true, data_range=None):
    """Structural Similarity Index (wraps skimage if available)."""
    try:
        from skimage.metrics import structural_similarity as sk_ssim
    except ImportError as e:
        raise ImportError("pip install scikit-image for SSIM") from e
    if data_range is None:
        data_range = x_true.max() - x_true.min()
    return sk_ssim(x_est, x_true, data_range=data_range)


# -------------------------------------------------------- observation op --
def _sample_at_sensors(x, sensor_coords):
    """x: (N, N) torch tensor. Returns (k*,) tensor sampled at sensor_coords."""
    rows = sensor_coords[:, 0]
    cols = sensor_coords[:, 1]
    return x[rows, cols]


# ------------------------------------------------------------- objectives --
def bg_term_autograd(x, x_b, Sk_tensor):
    """
    0.5 * ||x - x_b||^2_{B^-1} computed fully inside autograd, using the
    Fourier-diagonal representation of a stationary covariance:
    B^-1 acts as division by Sk in Fourier space. Sk_tensor: (N,N) real
    tensor, the covariance's power spectrum (see StationaryCovariance.Sk).
    """
    diff = x - x_b
    Fdiff = torch.fft.fft2(diff)
    inv = Fdiff / (Sk_tensor + 1e-8 * Sk_tensor.max())
    quad = torch.real(torch.fft.ifft2(inv)) * diff
    return 0.5 * torch.sum(quad)


def obs_term(x, y_obs, sensor_coords, r_obs_std, nonlinear=False, r1=3, r2=1.5):
    """0.5 * ||y - H(x)||^2_{R^-1}, R = r_obs_std^2 * I (Eq. 1's R_t).

    H is either the identity sampling operator (Eq. 12, nonlinear=False)
    or the paper's local weighted-sum-of-squares operator (Eq. 42-44,
    nonlinear=True) -- must match whichever generated y_obs in
    sensors.build_tessellated_observation, or the DA cost function is
    being fit against observations that don't correspond to its own H.
    """
    if nonlinear:
        field = nonlinear_observation_field_torch(x, r1=r1, r2=r2)
    else:
        field = x
    Hx = _sample_at_sensors(field, sensor_coords)
    resid = y_obs - Hx
    return 0.5 * torch.sum(resid**2) / (r_obs_std**2 + 1e-12)


def inv_op_term(x, x_v, Sk_tensor_P):
    """0.5 * ||x - x_v||^2_{P^-1}, same Fourier-diagonal trick as B.
    Only valid when P_t is modeled as stationary (has a .Sk spectrum)."""
    return bg_term_autograd(x, x_v, Sk_tensor_P)


def inv_op_term_dense(x, x_v, P_inv_tensor):
    """
    0.5 * ||x - x_v||^2_{P^-1} using a dense, precomputed P^-1 -- for when
    P_t is the true empirical (Gaspari-Cohn localized) covariance from
    `vcnn.estimate_P` (Eq. 19-21), rather than a stationary approximation.
    P_inv_tensor: (N*N, N*N) torch tensor, the regularized pseudo-inverse
    of P_t_localized, precomputed once outside the optimization loop.
    """
    N = x.shape[0]
    diff = (x - x_v).reshape(-1)
    return 0.5 * diff @ (P_inv_tensor @ diff)


# --------------------------------------------------------------- solvers --
def _run_lbfgs(objective_fn, x0, N, maxiter=200):
    """
    objective_fn: callable(x_torch) -> scalar torch loss.
    x0: (N,N) numpy initial guess.
    Returns (x_opt (N,N) numpy, n_iter, history of J).
    """
    x_flat0 = x0.ravel().astype(np.float64)
    history = {"J": []}

    def fun_and_grad(x_flat):
        x_t = torch.tensor(x_flat.reshape(N, N), dtype=torch.float64, requires_grad=True)
        loss = objective_fn(x_t)
        loss.backward()
        history["J"].append(loss.item())
        grad = x_t.grad.detach().numpy().ravel().astype(np.float64)
        return loss.item(), grad

    res = minimize(fun_and_grad, x_flat0, jac=True, method="L-BFGS-B",
                    options={"maxiter": maxiter, "gtol": 1e-6})
    return res.x.reshape(N, N), res.nit, history["J"]


def da_3dvar(x_b, y_obs, sensor_coords, B_cov, r_obs_std, s_b, maxiter=200, nonlinear=False):
    """Conventional 3D-Var, Eq. 1. Returns (x_analysis, n_iter, J_history).

    s_b: background error std used to generate x_b (Eq. 41). B_cov.Sk is
    the UNIT-VARIANCE correlation spectrum -- the true B_t = s_b^2 * C,
    so it must be scaled by s_b**2 here, or the background term is
    weighted ~1/s_b**2 times too weakly relative to J_p/J_o.
    """
    N = x_b.shape[0]
    Sk_B = torch.tensor(B_cov.Sk, dtype=torch.float64) * s_b**2
    x_b_t = torch.tensor(x_b, dtype=torch.float64)
    y_t = torch.tensor(y_obs, dtype=torch.float64)

    def objective(x_t):
        J_b = bg_term_autograd(x_t, x_b_t, Sk_B)
        J_o = obs_term(x_t, y_t, sensor_coords, r_obs_std, nonlinear=nonlinear)
        return J_b + J_o

    return _run_lbfgs(objective, x_b.copy(), N, maxiter=maxiter)


def _dense_P_inv(P_t_localized, rcond=1e-2):
    """
    Build the regularized inverse of a dense empirical covariance matrix
    (from vcnn.estimate_P) once, for reuse across all DA solves in a run.

    Uses a pseudo-inverse rather than a plain inverse: with n_val
    validation samples over N*N pixels, P_t_localized is rank-deficient
    (rank <= n_val - 1) whenever n_val < N*N, which is the common case --
    a plain np.linalg.inv would fail or blow up numerically. `rcond`
    controls how aggressively small/noisy singular values are treated as
    zero (standard Tikhonov-style regularization for this situation).

    Returns a (N*N, N*N) numpy array; convert to a torch tensor once
    (dtype float64) before passing into da_vivid repeatedly, to avoid
    rebuilding it per snapshot.
    """
    P_inv = np.linalg.pinv(P_t_localized, rcond=rcond)
    return P_inv


def da_vivid(x_b, x_v, y_obs, sensor_coords, B_cov, P_cov, r_obs_std, s_b, maxiter=200, nonlinear=False):
    """
    VIVID, Eq. 18: background + inverse-operator + observation terms.
    x_v: VCNN(Y_tilde) prediction, same shape as x_b (Section 3.2).
    ...
    s_b: background error std used to generate x_b (Eq. 41) -- see
    da_3dvar's docstring for why Sk_B must be scaled by s_b**2.
    """
    N = x_b.shape[0]
    Sk_B = torch.tensor(B_cov.Sk, dtype=torch.float64) * s_b**2
    x_b_t = torch.tensor(x_b, dtype=torch.float64)
    x_v_t = torch.tensor(x_v, dtype=torch.float64)
    y_t = torch.tensor(y_obs, dtype=torch.float64)

    use_dense_P = isinstance(P_cov, torch.Tensor)
    if use_dense_P:
        P_inv_t = P_cov  # already a (N*N, N*N) regularized inverse, precomputed once
    else:
        Sk_P = torch.tensor(P_cov.Sk, dtype=torch.float64)

    def objective(x_t):
        J_b = bg_term_autograd(x_t, x_b_t, Sk_B)
        if use_dense_P:
            J_p = inv_op_term_dense(x_t, x_v_t, P_inv_t)
        else:
            J_p = inv_op_term(x_t, x_v_t, Sk_P)
        J_o = obs_term(x_t, y_t, sensor_coords, r_obs_std, nonlinear=nonlinear)
        return J_b + J_p + J_o

    x0 = 0.5 * (x_b + x_v)  # informed initialization
    return _run_lbfgs(objective, x0, N, maxiter=maxiter)


def da_vivid_rom(x_b_hat, x_v_hat, y_obs, sensor_coords, POD_basis,
                  B_hat_cov, P_hat_cov, r_obs_std, maxiter=200):
    """
    Eq. 37: reduced-space VIVID. x_b_hat, x_v_hat: (q,) reduced vectors.
    POD_basis: (N*N, q) array, L_{X,q} from Eq. 32-33 (decompress with
    x = POD_basis @ x_hat before applying H).
    B_hat_cov / P_hat_cov: here treated as diagonal covariances in the
    reduced space (q is usually small enough for a dense q x q matrix,
    but a diagonal approx is often adequate and cheap -- swap in a
    dense np.ndarray + explicit inverse if you have an empirical B_hat).
    """
    N = int(np.sqrt(POD_basis.shape[0]))
    q = POD_basis.shape[1]
    basis_t = torch.tensor(POD_basis, dtype=torch.float64)
    x_b_hat_t = torch.tensor(x_b_hat, dtype=torch.float64)
    x_v_hat_t = torch.tensor(x_v_hat, dtype=torch.float64)
    y_t = torch.tensor(y_obs, dtype=torch.float64)
    Bh_diag = torch.tensor(np.diag(B_hat_cov), dtype=torch.float64) if B_hat_cov.ndim == 2 \
        else torch.tensor(B_hat_cov, dtype=torch.float64)
    Ph_diag = torch.tensor(np.diag(P_hat_cov), dtype=torch.float64) if P_hat_cov.ndim == 2 \
        else torch.tensor(P_hat_cov, dtype=torch.float64)

    history = {"J": []}

    def fun_and_grad(x_hat_flat):
        x_hat_t = torch.tensor(x_hat_flat, dtype=torch.float64, requires_grad=True)
        J_b = 0.5 * torch.sum((x_hat_t - x_b_hat_t) ** 2 / (Bh_diag + 1e-12))
        J_p = 0.5 * torch.sum((x_hat_t - x_v_hat_t) ** 2 / (Ph_diag + 1e-12))
        x_full = (basis_t @ x_hat_t).reshape(N, N)
        Hx = _sample_at_sensors(x_full, sensor_coords)
        J_o = 0.5 * torch.sum((y_t - Hx) ** 2) / (r_obs_std**2 + 1e-12)
        loss = J_b + J_p + J_o
        loss.backward()
        history["J"].append(loss.item())
        return loss.item(), x_hat_t.grad.detach().numpy()

    x0 = 0.5 * (x_b_hat + x_v_hat)
    res = minimize(fun_and_grad, x0, jac=True, method="L-BFGS-B",
                    options={"maxiter": maxiter, "gtol": 1e-6})
    x_analysis = (POD_basis @ res.x).reshape(N, N)
    return x_analysis, res.x, res.nit, history["J"]