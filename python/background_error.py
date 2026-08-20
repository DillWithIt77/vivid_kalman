"""
background_error.py

Spatially-correlated background error, implementing:
  - Eq. 40-41: Matern(3/2) covariance noise added to the true state to
    build the background x_b,t
  - Eq. 20-21: Gaspari-Cohn localization of an empirically-estimated
    covariance matrix (used for P_t, the VCNN inverse-operator error
    covariance)

Design note
-----------
Your QG domain is doubly-periodic (N x N, spectral solver), so instead
of building the dense N^2 x N^2 covariance matrix B_t explicitly (as the
paper does for their 50x50 grid -- already 2500x2500), we represent
stationary covariance operators by their power spectrum and apply them
via FFT. This is both faster and exact on a periodic domain, and it
lets B^{-1} be applied as a simple Fourier-space division instead of a
dense solve. Use `StationaryCovariance` for B_t (and P_t if you choose
to model it as stationary/isotropic rather than empirically estimating
it per Eq. 19).
"""

import numpy as np


def gaspari_cohn(rho):
    """Gaspari-Cohn localization function, Eq. 20. rho = r / L (>=0)."""
    rho = np.asarray(rho, dtype=float)
    G = np.zeros_like(rho)

    m1 = rho < 1
    r = rho[m1]
    G[m1] = 1 - (5/3)*r**2 + (5/8)*r**3 + (1/2)*r**4 - (1/4)*r**5

    m2 = (rho >= 1) & (rho < 2)
    r = rho[m2]
    G[m2] = (4 - 5*r + (5/3)*r**2 + (5/8)*r**3 - (1/2)*r**4
             + (1/12)*r**5 - (2/3)/r)

    return G  # rho >= 2 stays 0


def matern32(r, L):
    """Matern kernel of order 3/2, Eq. 40: phi(r) = (1 + r/L) exp(-r/L)."""
    r = np.asarray(r, dtype=float)
    return (1 + r / L) * np.exp(-r / L)


class StationaryCovariance:
    """
    Isotropic, stationary (periodic) covariance operator on an N x N
    doubly-periodic grid, defined by its 1D radial correlation function
    phi(r) (e.g. matern32) and correlation length L.

    Represents C implicitly through its spectral (Fourier) diagonal,
    obtained as the FFT of the correlation kernel evaluated on the
    periodic grid (circulant embedding). Provides:
      - sample(std)      : draw a correlated random field ~ N(0, std^2 C)
      - apply(x)          : C @ x  (flattened or 2D x)
      - inv_apply(x, eps) : C^{-1} @ x  (regularized division in Fourier space)
    """

    def __init__(self, N, L, kernel=matern32, gaspari_cohn_localize=False):
        self.N = N
        self.L = L

        ###computes the distance r from paper (see Eq. 20)
        ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        # periodic (wrap-around) distance from (0,0)
        di = np.minimum(ii, N - ii)
        dj = np.minimum(jj, N - jj)
        r = np.sqrt(di**2 + dj**2)

        ###builds a baseline kernal parametered by L (forces smoothness, so not computing raw covariance matrix entries)
        kernel_grid = kernel(r, L)
        if gaspari_cohn_localize:
            ###tapers using the Gaspari-Cohn function
            kernel_grid = kernel_grid * gaspari_cohn(r / L)

        # power spectrum = FFT of the (real, symmetric) correlation kernel;
        # clip tiny negative numerical artifacts from imperfect symmetry
        Sk = np.real(np.fft.fft2(kernel_grid))
        self.Sk = np.clip(Sk, 0, None)

    def sample(self, std=1.0, rng=None, n_samples=1):
        """Draw n_samples correlated fields of shape (N, N) [(n,N,N) if >1]."""
        if rng is None:
            rng = np.random.default_rng()
        N = self.N
        out = np.empty((n_samples, N, N))
        amp = np.sqrt(np.clip(self.Sk, 0, None) / (N * N))
        for s in range(n_samples):
            white = rng.standard_normal((N, N))
            field = np.real(np.fft.ifft2(np.fft.fft2(white) * amp)) * (N * N) ** 0.5
            out[s] = std * field / field.std()  # normalize to unit std, then scale
        return out[0] if n_samples == 1 else out

    def apply(self, x):
        """C @ x for x of shape (N, N)."""
        return np.real(np.fft.ifft2(np.fft.fft2(x) * self.Sk))

    def inv_apply(self, x, eps=1e-8):
        """
        C^{-1} @ x, regularized: divide by (Sk + eps * max(Sk)).
        Note: at wavenumbers where Sk ~ 0 (fine-scale/high-frequency
        content the covariance kernel doesn't represent), this necessarily
        attenuates rather than recovers that content -- this is expected
        behavior for any localized/smooth covariance operator, not a
        numerical bug. Roundtrip apply->inv_apply is near-exact for
        smooth/low-frequency fields and lossy for white-noise-like input.
        """
        denom = self.Sk + eps * self.Sk.max()
        return np.real(np.fft.ifft2(np.fft.fft2(x) / denom))


def make_background(X_true, s_b, L, rng=None):
    """
    Eq. 41: x_b,t = x_t + s_b * eps_b,t,  eps_b,t ~ N(0, B(phi(L,r))).

    Parameters
    ----------
    X_true : (N, N) true state field
    s_b    : background error standard deviation (paper's s_b, e.g. 0.005-0.03)
    L      : Matern correlation length (paper fixes L = 5)

    Returns
    -------
    X_b : (N, N) background field
    cov : StationaryCovariance  (reuse this as B_t in the DA objective)
    """
    N = X_true.shape[0]
    cov = StationaryCovariance(N, L, kernel=matern32)
    noise = cov.sample(std=1.0, rng=rng)
    X_b = X_true + s_b * noise
    return X_b, cov
