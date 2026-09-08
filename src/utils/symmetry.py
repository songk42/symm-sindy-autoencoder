"""
Learned discrete-symmetry generators for the latent SINDy dynamics (PDF Phase C,
"unconstrained generator discovery").

We do NOT prescribe a group. We learn one (or a few) generator matrices
T_k in SO(d) directly, parameterized through the matrix exponential of a skew
matrix (covers the whole connected group, including -I, with no Cayley
singularity), and score how well the latent dynamics

    zdot = Theta(z) Xi,   Xi in R^{L x d}

are equivariant under them:  f(T z) = T f(z).

Because Theta is a polynomial library, Theta(T z) is itself a linear image of
Theta(z):
    Theta(z) C(T) = Theta(T z)                       (identity in z)
with C(T) an L x L matrix (a representation: C(T1 T2) = C(T1) C(T2)). Plugging
into the equivariance condition and using that Theta(z) spans R^L gives the
finite intertwining constraint

    C(T) Xi = Xi T^T .

`induced_library_matrix` builds C(T) exactly and differentiably by solving the
small linear system Theta(Z_s) C = Theta(Z_s T^T) on a fixed generic sample set
Z_s (Vandermonde-style, invertible for generic points).

Sine library terms break the polynomial-closure identity, so Phase C requires
use_sine=False.
"""

import numpy as np
import torch

from src.utils.model_utils import library_size, sindy_library


def skew(A):
    """Skew-symmetric part: so(d) element."""
    return A - A.transpose(-1, -2)


def so_matrix(A):
    """Map an unconstrained d x d parameter to T in SO(d) via T = expm(A - A^T)."""
    return torch.matrix_exp(skew(A))


def pi_rotation_params(d, dtype=torch.float32):
    """Unconstrained parameter A with so_matrix(A) = rotation by pi in the
    (0, 1) plane = diag(-1, -1, 1, ..., 1).

    d = 2 -> -I (the Z2 inversion);  d = 3 -> the Lorenz (x, y) -> (-x, -y) action.
    This is the canonical order-2 element to *seed* or *freeze* a generator at.
    """
    import math
    A = torch.zeros(d, d, dtype=dtype)
    A[0, 1] = -math.pi / 2
    A[1, 0] = math.pi / 2
    return A


# --------------------------------------------------------------------------- #
# induced action on the polynomial library
# --------------------------------------------------------------------------- #

class InducedLibraryRep:
    """Callable T -> C(T) (L x L), differentiable in T.

    Holds the fixed sample points and the pinv of Theta(Z_s) so each call is one
    matmul + one solve on an L x L system.
    """

    def __init__(self, d, poly_order, include_constant=True, use_sine=False,
                 n_samples=None, seed=0, device="cpu", dtype=torch.float64):
        if use_sine:
            raise ValueError("Phase C needs a polynomial library (use_sine=False)")
        self.d = d
        self.poly_order = poly_order
        self.include_constant = include_constant
        self.L = library_size(d, poly_order, use_sine=False,
                              include_constant=include_constant)
        S = n_samples or (2 * self.L + 8)
        g = torch.Generator(device="cpu").manual_seed(seed)
        Z = torch.randn(S, d, generator=g, dtype=dtype)
        self.Z = Z.to(device)
        B = self._theta(self.Z)                       # (S, L)
        self.Bpinv = torch.linalg.pinv(B)             # (L, S)

    def _theta(self, Z):
        return sindy_library(Z, self.poly_order, Z.device,
                             include_sine=False,
                             include_constant=self.include_constant)

    def to(self, device=None, dtype=None):
        self.Z = self.Z.to(device=device, dtype=dtype)
        self.Bpinv = self.Bpinv.to(device=device, dtype=dtype)
        return self

    def __call__(self, T):
        """T: (d, d) -> C(T): (L, L) with Theta(z) C(T) = Theta(T z)."""
        T = T.to(self.Z.dtype)
        A = self._theta(self.Z @ T.transpose(-1, -2))     # (S, L), differentiable in T
        return self.Bpinv.to(T.dtype) @ A                 # (L, L)


def equivariance_residual(Xi, T, induced_rep):
    """|| C(T) Xi - Xi T^T ||_F^2 / ||Xi||_F^2  -- relative intertwining error.

    Xi: (L, d) SINDy coefficients.  T: (d, d).  induced_rep: InducedLibraryRep.
    """
    C = induced_rep(T).to(Xi.dtype)
    lhs = C @ Xi
    rhs = Xi @ T.transpose(-1, -2).to(Xi.dtype)
    return ((lhs - rhs) ** 2).sum() / (Xi.pow(2).sum() + 1e-12)


def finite_order_residual(T, order):
    """|| T^order - I ||_F^2 / d  -- how close T is to having the given order."""
    d = T.shape[-1]
    Tp = torch.linalg.matrix_power(T, order)
    return ((Tp - torch.eye(d, device=T.device, dtype=T.dtype)) ** 2).sum() / d


def nontriviality_penalty(T, eps=1e-3):
    """Grows as T -> I; bounded. Keeps the learned generator away from identity."""
    d = T.shape[-1]
    dist2 = ((T - torch.eye(d, device=T.device, dtype=T.dtype)) ** 2).sum()
    return 1.0 / (dist2 + eps)
