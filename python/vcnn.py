"""
vcnn.py

Voronoi-tessellation CNN (VCNN) inverse operator: maps the tessellated
observation field Y_tilde_t -> state field X_t (Table 1), and the
reduced-order variant CNN_ROM: Y_tilde_t -> reduced latent vector
x_hat_t (Table 2).

Both take a single-channel (N, N) tessellated observation as input.
Since our QG domain is square and Nx=Ny=Mx=My, no cropping/resizing
layer is needed (paper's optional Cropping2D is for non-square cases).
"""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


class VCNN(nn.Module):
    """Table 1: 7 conv layers, 48 channels, ReLU, maps Y_tilde -> X."""

    def __init__(self, channels=48, n_layers=6, kernel_size=8):
        super().__init__()
        # padding='same' (not a fixed int) is required here: kernel_size=8
        # is even, so symmetric integer padding (kernel//2) would grow the
        # spatial size by 1 pixel per layer instead of preserving it.
        layers = []
        in_ch = 1
        for _ in range(n_layers):
            layers += [
                nn.Conv2d(in_ch, channels, kernel_size, padding="same"),
                nn.ReLU(),
            ]
            in_ch = channels
        self.body = nn.Sequential(*layers)
        # final 1x1-ish conv back to a single-channel state field
        self.head = nn.Conv2d(channels, 1, kernel_size=1)
        self.act = nn.ReLU()  # matches paper's final ReLU (Table 1)
        # NOTE: if your state field X can be negative (PV/vorticity
        # typically is), consider dropping the final ReLU -- the paper
        # used it because their velocity field/setup made it appropriate.
        # We keep it available but off by default; see `final_relu`.
        self.final_relu = False

    def forward(self, y_tilde):
        # y_tilde: (B, 1, N, N)
        h = self.body(y_tilde)
        out = self.head(h)
        if self.final_relu:
            out = self.act(out)
        return out  # (B, 1, N, N)


class VCNN_ROM(nn.Module):
    """Table 2: convolutional encoder with maxpooling, Y_tilde -> x_hat (dim q)."""

    def __init__(self, N, q, channels=16):
        super().__init__()
        # padding='same' avoids the even-kernel size-growth issue (see VCNN)
        self.conv1 = nn.Conv2d(1, channels, 8, padding="same")
        self.conv2 = nn.Conv2d(channels, channels, 8, padding="same")
        self.pool1 = nn.MaxPool2d(2, ceil_mode=True)
        self.conv3 = nn.Conv2d(channels, channels, 4, padding="same")
        self.pool2 = nn.MaxPool2d(2, ceil_mode=True)
        self.conv4 = nn.Conv2d(channels, channels, 4, padding="same")
        self.act = nn.ReLU()

        with torch.no_grad():
            dummy = torch.zeros(1, 1, N, N)
            flat_dim = self._forward_conv(dummy).numel()
        self.fc = nn.Linear(flat_dim, q)

    def _forward_conv(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.pool1(x)
        x = self.act(self.conv3(x))
        x = self.pool2(x)
        x = self.act(self.conv4(x))
        return x

    def forward(self, y_tilde):
        x = self._forward_conv(y_tilde)
        x = x.flatten(1)
        return self.fc(x)  # (B, q)


def train_vcnn(model, train_loader, val_loader=None, n_epochs=50, lr=1e-3,
               device="cpu", save_path=None):
    """
    Standard supervised training loop with validation monitoring and best-weight saving.
    """
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    history = {"train_loss": [], "val_loss": []}
    
    for epoch in range(n_epochs):
        model.train()
        running = 0.0
        n = 0
        for Yt, target in train_loader:
            Yt, target = Yt.to(device), target.to(device)
            opt.zero_grad()
            pred = model(Yt)
            loss = loss_fn(pred, target)
            loss.backward()
            opt.step()
            running += loss.item() * Yt.shape[0]
            n += Yt.shape[0]
        train_loss = running / n
        history["train_loss"].append(train_loss)

        val_loss = None
        if val_loader is not None:
            model.eval()
            running, n = 0.0, 0
            with torch.no_grad():
                for Yt, target in val_loader:
                    Yt, target = Yt.to(device), target.to(device)
                    pred = model(Yt)
                    loss = loss_fn(pred, target)
                    running += loss.item() * Yt.shape[0]
                    n += Yt.shape[0]
            val_loss = running / n
            history["val_loss"].append(val_loss)

            # Save the model if it achieved the lowest validation loss so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_path is not None:
                    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), save_path)

        msg = f"epoch {epoch+1}/{n_epochs}  train_loss={train_loss:.6g}"
        if val_loss is not None:
            msg += f"  val_loss={val_loss:.6g}"
            if val_loss == best_val_loss:
                msg += " (saved best)"
        print(msg)

    return history


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

def inv_op_residuals(model, val_loader, device="cpu"):
    """
    Eq. 19: empirically estimate P_t (error covariance of the learned
    inverse operator) on an independent validation set, as the sample
    covariance of (X - x_v) flattened residuals. Returns the mean
    residual field-squared map (diagonal proxy) AND the full residual
    stack, so you can localize via Gaspari-Cohn (background_error.py)
    or build a StationaryCovariance from the residuals' empirical
    correlation length.
    """
    model.eval()
    residuals = []
    with torch.no_grad():
        for Yt, target in val_loader:
            Yt, target = Yt.to(device), target.to(device)
            pred = model(Yt)
            residuals.append((target - pred).cpu().numpy())
    import numpy as np
    residuals = np.concatenate(residuals, axis=0)  # (n_val, 1, N, N)
    return residuals


def estimate_P(residuals, L):
    """
    Empirically estimates P_t from validation residuals (Eq. 19) 
    and applies Gaspari-Cohn localization (Eq. 20-21).

    Parameters
    ----------
    residuals : (n_val, N, N) array of spatial residuals (target - pred)
    L         : float, Gaspari-Cohn correlation scale length

    Returns
    -------
    P_t : (N*N, N*N) localized empirical error covariance matrix
    """
    n_val, N, _ = residuals.shape
    n_pixels = N * N
    
    # 1. Flatten the spatial dimensions to 1D vectors per sample
    res_flat = residuals.reshape(n_val, n_pixels)
    
    # 2. Compute the raw sample covariance matrix P_t (Eq. 19)
    # rowvar=False treats columns as variables (grid pixels) and rows as samples
    P_t = np.cov(res_flat, rowvar=False)
    
    # 3. Create 2D grid coordinates for every flattened index
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    coords = np.stack([ii.ravel(), jj.ravel()], axis=-1)  # Shape: (N*N, 2)
    
    # 4. Compute pairwise Euclidean distances d(a, b) between all pixels
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    
    # 5. Calculate normalized distance rho = r / L and evaluate Gaspari-Cohn (Eq. 20)
    rho = distances / L
    G = gaspari_cohn(rho)  # Requires the Gaspari-Cohn function definition
    
    # 6. Element-wise localization of the covariance matrix (Eq. 21)
    P_t_localized = P_t * G
    
    return P_t_localized

