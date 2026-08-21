"""
sensors.py

Sensor placement and Voronoi-tessellation of sparse observations,
implementing Section 3.1 (Eq. 12-16) and Appendix A.2 of:

    Cheng, Liu, Guo, Arcucci (2024), "Efficient deep data assimilation
    with sparse observations and time-varying sensors", JCP 496:112581.

Two pieces:
  1. place_sensors_on_grid  -- structured grid + random jitter placement
     (Appendix A.2, Fig. 18): a sensor is placed within a random offset
     of each point of a coarse n_grid x n_grid lattice.
  2. voronoi_tessellate      -- fills the full observation field by
     nearest-sensor assignment (Eq. 13-16, Fig. 1's "Voronoi tessellation"
     panel).

For our QG setup, the observation field Y_t is taken to equal the state
field itself (Nx = Ny = Mx = My, as in the paper's shallow-water case),
i.e. sensors sample qp directly. If you want the paper's nonlinear local
sum-of-squares observation operator (Eq. 42-44) instead, see
`nonlinear_observation_field` at the bottom -- it's optional and off by
default.
"""

import numpy as np
from scipy.spatial import cKDTree


def place_sensors_on_grid(N, n_grid, r_s=3, rng=None):
    """
    Randomly place sensors near a structured n_grid x n_grid lattice
    over an N x N domain, matching Appendix A.2 / Fig. 18.

    Parameters
    ----------
    N : int            domain size (state field is N x N)
    n_grid : int       number of lattice points per side (paper used
                        10x10 for training, 6x6..12x12 for test)
    r_s : float        jitter radius (paper used r_s = 3, in pixels)
    rng : np.random.Generator or None

    Returns
    -------
    coords : (k*, 2) int array of (row, col) sensor positions, 0-indexed,
             clipped to [0, N-1].
    """
    if rng is None:
        rng = np.random.default_rng()

    centers = np.linspace(0, N - 1, n_grid)
    ii, jj = np.meshgrid(centers, centers, indexing="ij")
    centers = np.stack([ii.ravel(), jj.ravel()], axis=1)

    # random offset within a disk of radius r_s around each lattice point
    theta = rng.uniform(0, 2 * np.pi, size=centers.shape[0])
    r = r_s * np.sqrt(rng.uniform(0, 1, size=centers.shape[0]))
    offset = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)

    coords = centers + offset
    coords = np.clip(np.round(coords), 0, N - 1).astype(int)
    return coords


def voronoi_tessellate(N, sensor_coords, sensor_values):
    """
    Build the tessellated observation field Y_tilde in R^{N x N}
    (Eq. 13-16): every grid point takes the value of its nearest sensor.

    Parameters
    ----------
    N : int
    sensor_coords : (k*, 2) int array, (row, col) sensor positions
    sensor_values : (k*,) array, observed values at those positions

    Returns
    -------
    Y_tilde : (N, N) array
    """
    grid_rows, grid_cols = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    grid_pts = np.stack([grid_rows.ravel(), grid_cols.ravel()], axis=1)

    tree = cKDTree(sensor_coords)
    _, nearest_idx = tree.query(grid_pts, k=1)

    Y_tilde = sensor_values[nearest_idx].reshape(N, N)
    return Y_tilde


def sample_observations(X, sensor_coords, obs_noise_std=0.0, rng=None):
    """
    Sample the state field X at sensor locations, optionally adding
    relative Gaussian observation noise (Section 4.2.3: noise given as
    a fraction of the true observation value).

    Returns y (k*,) the observed values.
    """
    if rng is None:
        rng = np.random.default_rng()
    y = X[sensor_coords[:, 0], sensor_coords[:, 1]].copy()
    if obs_noise_std > 0:
        y = y + obs_noise_std * np.abs(y) * rng.standard_normal(y.shape)
    return y


def build_tessellated_observation(X, n_grid, r_s=3, obs_noise_std=0.0, nonlinear=False, rng=None):
    """
    Convenience wrapper: place sensors, sample X, tessellate.
    Returns (Y_tilde, sensor_coords, y) so the same sensor set / values
    can be reused for the DA observation term H_t(x).
    """
    N = X.shape[0]
    ###have the option for nonlinear observation field
    field_to_sample = nonlinear_observation_field(X) if nonlinear else X

    coords = place_sensors_on_grid(N, n_grid, r_s=r_s, rng=rng)
    y = sample_observations(field_to_sample, coords, obs_noise_std=obs_noise_std, rng=rng)
    Y_tilde = voronoi_tessellate(N, coords, y)
    return Y_tilde, coords, y


import torch

# Cache of precomputed kernel FFTs, keyed by (N, radius, dtype, device).
# The kernel doesn't depend on the field X, only on the grid size and
# radius, so redoing its FFT on every call (e.g. every L-BFGS iteration
# inside vivid_da.py's DA loop) would be wasted work.
_KERNEL_FFT_CACHE = {}


def _circular_disk_kernel(N, radius, device=None, dtype=torch.float64):
    """
    (N, N) real kernel, 1 inside `radius` of index (0,0) using PERIODIC
    (torus) distance, 0 outside. Centering the disk at index 0 (rather
    than the middle of the array) is what makes the FFT convolution
    below a true circular convolution: torch.fft.fft2 / ifft2 are
    inherently periodic, so as long as the kernel encodes wraparound
    distance, so does the convolution result -- correctly matching a
    doubly-periodic QG domain, unlike scipy's fftconvolve (zero-padded,
    i.e. implicitly non-periodic boundaries).
    """
    idx = torch.arange(N, device=device, dtype=dtype)
    d = torch.minimum(idx, N - idx)  # periodic distance to 0, per axis
    dy, dx = torch.meshgrid(d, d, indexing="ij")
    dist = torch.sqrt(dy**2 + dx**2)
    return (dist <= radius).to(dtype)


def _get_kernel_fft(N, radius, device, dtype):
    key = (N, radius, dtype, device)
    if key not in _KERNEL_FFT_CACHE:
        kernel = _circular_disk_kernel(N, radius, device=device, dtype=dtype)
        _KERNEL_FFT_CACHE[key] = torch.fft.fft2(kernel)
    return _KERNEL_FFT_CACHE[key]


def _circular_local_sum(field, kernel_fft):
    """
    Periodic 2D convolution of `field` (real torch tensor, may require
    grad) with a precomputed kernel FFT, via the convolution theorem.
    Differentiable w.r.t. `field` -- kernel_fft is treated as constant.
    """
    field_fft = torch.fft.fft2(field.to(kernel_fft.real.dtype))
    return torch.fft.ifft2(field_fft * kernel_fft).real


def nonlinear_observation_field_torch(X, beta_field=None, r1=3, r2=1.5):
    """
    Paper's Eq. 42-44 nonlinear local weighted-sum-of-squares observation
    field, adapted to be (a) doubly PERIODIC, matching the QG domain
    (the paper's own shallow-water setup was non-periodic, so this is a
    deliberate change, not a literal reproduction), and (b) torch-based
    and autograd-differentiable, so it can be used both for offline data
    generation (via the numpy wrapper below) AND directly inside
    vivid_da.py's DA objective (gradients need to flow through it there).

    X : (N, N) torch tensor, real, may require_grad
    beta_field : (N, N) torch tensor or None. None -> ones, matching the
        official reference implementation (voronoi_preprocessing.py),
        which hardcodes a constant `+= 1` for the inner neighborhood --
        NOT a spatially-varying field, despite the paper's beta_{ix,jx}
        notation suggesting otherwise (see chat discussion / repo).
    r1, r2 : outer / inner neighborhood radii. Defaults match the
        reference code's actual values (2.9, 1.5), not the paper's
        rounded r<=3 in Eq. 43.
    """
    N = X.shape[0]
    dtype = X.dtype if X.dtype in (torch.float32, torch.float64) else torch.float64
    device = X.device

    if beta_field is None:
        beta_field = torch.ones((N, N), dtype=dtype, device=device)

    k1_fft = _get_kernel_fft(N, r1, device, dtype)
    k2_fft = _get_kernel_fft(N, r2, device, dtype)

    X2 = X ** 2
    term1 = 0.5 * _circular_local_sum(X2, k1_fft)
    term2 = _circular_local_sum(beta_field * X2, k2_fft)
    return term1 + term2


def nonlinear_observation_field(X, beta_field=None, r1=3, r2=1.5):
    """
    Numpy-facing convenience wrapper around
    `nonlinear_observation_field_torch`, for callers (e.g.
    build_tessellated_observation below) that work with plain numpy
    arrays during offline data generation and don't need gradients.
    For the DA loop (vivid_da.py), call the torch version directly on
    the live tensor instead of round-tripping through numpy here, so
    the autograd graph stays intact.
    """
    X_t = torch.from_numpy(np.asarray(X, dtype=np.float64))
    beta_t = None if beta_field is None else torch.from_numpy(np.asarray(beta_field, dtype=np.float64))
    with torch.no_grad():
        out = nonlinear_observation_field_torch(X_t, beta_field=beta_t, r1=r1, r2=r2)
    return out.numpy()