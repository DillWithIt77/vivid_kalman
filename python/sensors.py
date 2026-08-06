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


def nonlinear_observation_field(X, beta_field=None):
    """
    Optional: paper's Eq. 42-44 nonlinear local weighted-sum-of-squares
    observation field. Not used by default (we observe the state
    directly, as noted above), but provided for parity with the paper.
    rho1 = disk radius 3, rho2 = disk radius 1.5 (in pixels).
    """
    from scipy.ndimage import uniform_filter
    N = X.shape[0]
    if beta_field is None:
        beta_field = np.ones((N, N))

    def disk_mask(radius):
        r = int(np.ceil(radius))
        yy, xx = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), indexing="ij")
        return (np.sqrt(xx**2 + yy**2) <= radius).astype(float)

    def local_sum(field, radius):
        mask = disk_mask(radius)
        from scipy.signal import fftconvolve
        return fftconvolve(field, mask, mode="same")

    X2 = X**2
    term1 = 0.5 * local_sum(X2, 3.0)
    term2 = local_sum(beta_field * X2, 1.5)
    return term1 + term2
