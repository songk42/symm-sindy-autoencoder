"""
Synthetic "coordinate problem" dataset.

Underlying dynamics: a continuous-time symmetric double-well Duffing oscillator

    z1' = z2
    z2' = z1 - z1^3 - delta * z2

Every right-hand-side term is odd, so the system is equivariant under the linear
Z2 inversion  (z1, z2) -> (-z1, -z2).

The latent state z is then pushed through a fixed, smooth, nonlinear map
psi : R^2 -> R^u (a random tanh MLP) to produce the observations x. psi does not
commute with any linear action on x, so the Z2 symmetry is *linear in z but
nonlinear in the observed coordinates* -- exactly the coordinate problem the
autoencoder is meant to untangle.

Output dict matches the Lorenz generator: keys 'x', 'dx', 'dz' (plus 'z'),
flattened to (n_ics * timesteps, dim).
"""

import numpy as np
import torch
from scipy.integrate import odeint


def _duffing_rhs(state, t, delta):
    z1, z2 = state
    return [z2, z1 - z1 ** 3 - delta * z2]


def _simulate(z0, t, delta):
    z = odeint(_duffing_rhs, z0, t, args=(delta,))
    dz = np.stack([_duffing_rhs(z[i], t[i], delta) for i in range(t.size)], axis=0)
    return z, dz


class _PsiMLP(torch.nn.Module):
    """Fixed random tanh MLP used as the unknown nonlinear observation map."""

    def __init__(self, z_dim, u_dim, hidden=64, seed=0, scale=1.0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.l1 = torch.nn.Linear(z_dim, hidden)
        self.l2 = torch.nn.Linear(hidden, hidden)
        self.l3 = torch.nn.Linear(hidden, u_dim)
        for lin, fan_in in [(self.l1, z_dim), (self.l2, hidden), (self.l3, hidden)]:
            w = torch.empty_like(lin.weight)
            torch.nn.init.normal_(w, std=scale / np.sqrt(fan_in), generator=g)
            lin.weight.data.copy_(w)
            b = torch.empty_like(lin.bias)
            torch.nn.init.normal_(b, std=0.1, generator=g)
            lin.bias.data.copy_(b)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, z):
        h = torch.tanh(self.l1(z))
        h = torch.tanh(self.l2(h))
        return self.l3(h)


# the symmetric double-well Duffing latent is intrinsically 2D
LATENT_DIM = 2


def get_duffing_data(n_ics, timesteps=250, u_dim=128, z_dim=LATENT_DIM, delta=0.1,
                     dt=0.02, noise_strength=0.0, psi_seed=0, ic_seed=None,
                     distort=True, psi_scale=1.0):
    """
    Arguments:
        n_ics          - number of random initial conditions / trajectories
        timesteps      - steps per trajectory
        u_dim          - observed dimension (output of psi)
        z_dim          - latent dimension; must be 2 (the Duffing state), kept
                         only so a caller can pass args.z_dim through
        delta          - damping coefficient
        dt             - integration step
        noise_strength - additive Gaussian noise on x / dx
        psi_seed       - seed for the fixed observation map
        distort        - if False, psi is the identity padded to u_dim (no
                         coordinate problem; useful as an ablation)
    """
    if z_dim != LATENT_DIM:
        raise ValueError(f"the synth Duffing system is {LATENT_DIM}D; got z_dim={z_dim}. "
                         f"Run with -Z {LATENT_DIM}.")
    rng = np.random.RandomState(ic_seed)
    t = np.arange(0, timesteps * dt, dt)[:timesteps]

    # spread initial conditions across both wells and the saddle region
    ics = rng.uniform(low=[-2.2, -2.2], high=[2.2, 2.2], size=(n_ics, 2))

    z = np.zeros((n_ics, timesteps, LATENT_DIM))
    dz = np.zeros_like(z)
    for i in range(n_ics):
        z[i], dz[i] = _simulate(ics[i], t, delta)

    z_flat = torch.tensor(z.reshape(-1, LATENT_DIM), dtype=torch.float32)
    dz_flat = torch.tensor(dz.reshape(-1, LATENT_DIM), dtype=torch.float32)

    if distort:
        psi = _PsiMLP(LATENT_DIM, u_dim, seed=psi_seed, scale=psi_scale)
        # x = psi(z),  dx/dt = J_psi(z) @ dz/dt  (forward-mode jacobian-vector product)
        x_flat, dx_flat = torch.autograd.functional.jvp(psi, z_flat, dz_flat)
        x_flat = x_flat.detach().numpy()
        dx_flat = dx_flat.detach().numpy()
    else:
        x_flat = np.zeros((z_flat.shape[0], u_dim), dtype=np.float32)
        dx_flat = np.zeros_like(x_flat)
        x_flat[:, :LATENT_DIM] = z_flat.numpy()
        dx_flat[:, :LATENT_DIM] = dz_flat.numpy()

    if noise_strength:
        x_flat = x_flat + noise_strength * rng.randn(*x_flat.shape)
        dx_flat = dx_flat + noise_strength * rng.randn(*dx_flat.shape)

    return {
        'x': x_flat.astype(np.float32),
        'dx': dx_flat.astype(np.float32),
        'dz': dz.reshape(-1, LATENT_DIM).astype(np.float32),
        'z': z.reshape(-1, LATENT_DIM).astype(np.float32),
    }
