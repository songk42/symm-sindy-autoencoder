"""
Koopman spectral diagnostic for discrete symmetries (PDF Phase 3, the
coordinate-free route).

The idea: the Koopman operator K advances observables along the flow,
    (K g)(y) = g(F(y)).
A state-space symmetry S (F(S y) = S F(y)) induces a pullback operator
    (U_S g)(y) = g(S y),
and because F and S commute, so do K and U_S:  U_S K = K U_S.

Given trajectory data we estimate finite matrices in a dictionary basis
(Extended DMD). The primary test is a *separating* one: fit the Koopman
operator with and without the constraint that it commute with U_S, and compare
one-step accuracy on the same held pairs.

  * symmetry_cost  (fit_with_S - fit_free) / fit_free
                   ~0   => imposing S is free       => S is a real symmetry
                   >>0  => imposing S costs accuracy => S is not a symmetry
  * commutator     || U_S K - K U_S || / || K ||    (secondary, on the free K)
  * closure        || U_S^p - I ||                  (S has finite order p)
  * sectors        fraction of Koopman eigenfunctions with phi(S y) = w phi(y),
                   w^p = 1  (informational -- noisy under eigenvalue degeneracy)

The constrained fit is obtained by augmenting the pair set with (S y_t, S y_{t+1});
a least-squares operator fit to an exactly S-symmetric pair set commutes with
U_S by construction, so scoring it back on the raw pairs isolates whether the
*dynamics* (not the sample cloud) actually respect S.

Everything here is plain numpy so it runs on a login node without a GPU.
Discrete-time EDMD is used (consecutive samples within a trajectory). The
dataset also carries exact derivatives, so a generator variant is available --
see `generator_edmd`.
"""

import numpy as np


# ----------------------------------------------------------------------------
# dictionary
# ----------------------------------------------------------------------------

def _monomial_exponents(n, degree, include_constant=True):
    """Exponent tuples for every monomial in n vars up to total `degree`,
    ordered (constant, degree 1, degree 2 with i<=j, ...) -- same convention as
    src.utils.model_utils.sindy_library."""
    exps = []
    if include_constant:
        exps.append((0,) * n)
    from itertools import combinations_with_replacement
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(n), d):
            e = [0] * n
            for idx in combo:
                e[idx] += 1
            exps.append(tuple(e))
    return exps


def poly_features(Y, degree, include_constant=True):
    """(m, n) -> (m, N) polynomial dictionary. Returns (Psi, exps)."""
    Y = np.asarray(Y, dtype=np.float64)
    m, n = Y.shape
    exps = _monomial_exponents(n, degree, include_constant)
    Psi = np.empty((m, len(exps)), dtype=np.float64)
    for j, e in enumerate(exps):
        col = np.ones(m)
        for k, p in enumerate(e):
            if p:
                col = col * Y[:, k] ** p
        Psi[:, j] = col
    return Psi, exps


# ----------------------------------------------------------------------------
# EDMD
# ----------------------------------------------------------------------------

def _pairs(Y, timesteps):
    """Split a flattened (n_ics*timesteps, n) array into consecutive-in-time
    snapshot pairs, without crossing trajectory boundaries."""
    n = Y.shape[1]
    Y = Y.reshape(-1, timesteps, n)
    Xf = Y[:, :-1, :].reshape(-1, n)
    Xn = Y[:, 1:, :].reshape(-1, n)
    return Xf, Xn


def symmetrize_pairs(Xf, Xn, S):
    """Augment snapshot pairs with their image under S: (S y_t, S y_{t+1}).

    This makes the empirical measure exactly S-invariant, so any residual
    non-commutation of the EDMD operator with U_S reflects the *dynamics* not
    the finite-sample asymmetry of the data cloud. It is the right control when
    testing whether F is equivariant under S.
    """
    S = np.asarray(S, dtype=np.float64)
    return np.vstack([Xf, Xf @ S.T]), np.vstack([Xn, Xn @ S.T])


def edmd(Yf, Yn, degree=3, include_constant=True, rcond=1e-10):
    """Extended DMD. Yf, Yn are (m, n) arrays of snapshot pairs y_t, y_{t+1}.

    Returns dict with:
        K      (N, N)   Koopman matrix acting on the right:  Psi(y_{t+1}) ~ Psi(y_t) K
        evals  (N,)     eigenvalues of K, sorted by |lambda| desc
        xi     (N, N)   right eigenvectors (columns), K xi = xi diag(evals)
        exps            monomial exponent tuples
        fit             relative one-step prediction residual in dictionary space
    """
    PsiX, exps = poly_features(Yf, degree, include_constant)
    PsiY, _ = poly_features(Yn, degree, include_constant)
    K, *_ = np.linalg.lstsq(PsiX, PsiY, rcond=rcond)     # (N, N)
    resid = np.linalg.norm(PsiX @ K - PsiY) / (np.linalg.norm(PsiY) + 1e-30)
    evals, xi = np.linalg.eig(K)
    order = np.argsort(-np.abs(evals))
    return {
        "K": K,
        "evals": evals[order],
        "xi": xi[:, order],
        "exps": exps,
        "degree": degree,
        "include_constant": include_constant,
        "fit": float(resid),
        "PsiX": PsiX,
    }


def generator_edmd(Y, Ydot, degree=3, include_constant=True, rcond=1e-10):
    """Generator EDMD from exact derivatives:  d/dt Psi(y) = Psi(y) L.

    Y, Ydot are (m, n).  Eigenvalues of L are continuous rates
    (lambda_discrete = exp(lambda_L * dt)).
    """
    Psi, exps = poly_features(Y, degree, include_constant)
    n = Y.shape[1]
    m, N = Psi.shape
    # d/dt Psi_j(y) = sum_k (d Psi_j / d y_k) * ydot_k
    Psidot = np.zeros((m, N))
    for j, e in enumerate(exps):
        for k, p in enumerate(e):
            if p == 0:
                continue
            term = p * Y[:, k] ** (p - 1)
            for kk, pp in enumerate(e):
                if kk != k and pp:
                    term = term * Y[:, kk] ** pp
            Psidot[:, j] += term * Ydot[:, k]
    L, *_ = np.linalg.lstsq(Psi, Psidot, rcond=rcond)
    resid = np.linalg.norm(Psi @ L - Psidot) / (np.linalg.norm(Psidot) + 1e-30)
    evals, xi = np.linalg.eig(L)
    order = np.argsort(-evals.real)
    return {"L": L, "evals": evals[order], "xi": xi[:, order],
            "exps": exps, "fit": float(resid), "Psi": Psi}


# ----------------------------------------------------------------------------
# symmetry test
# ----------------------------------------------------------------------------

def symmetry_report(Yf, Yn, S, order=2, degree=3, include_constant=True,
                    rcond=1e-10):
    """Test whether the dynamics behind snapshot pairs (Yf, Yn) are equivariant
    under a candidate linear action S (n x n) of finite `order`.

    Primary discriminator -- the separating test:
        fit_free       one-step EDMD residual, unconstrained
        fit_sym        one-step EDMD residual on the SAME raw pairs, using the
                       operator that is forced to commute with U_S (obtained by
                       fitting on the S-augmented pair set)
        symmetry_cost  (fit_sym - fit_free) / fit_free
                       ~0   => imposing S is free  => S is a real symmetry
                       >>0  => imposing S costs accuracy => S is not a symmetry

    Secondary signals:
        commutator     || P K_free - K_free P || / || K_free ||   (on the free K)
        closure        || P^order - I ||                          (S has that order)
        pullback_fit   how exactly Psi(S y) lies in span(Psi)     (~0 for poly S)
        sector_frac    fraction of leading K_free eigenfunctions that split as
                       phi(S y) = w phi(y),  w^order = 1
    """
    S = np.asarray(S, dtype=np.float64)
    inc = include_constant

    PsiX, exps = poly_features(Yf, degree, inc)
    PsiY, _ = poly_features(Yn, degree, inc)
    normY = np.linalg.norm(PsiY) + 1e-30

    # unconstrained
    K_free, *_ = np.linalg.lstsq(PsiX, PsiY, rcond=rcond)
    fit_free = np.linalg.norm(PsiX @ K_free - PsiY) / normY

    # symmetry-constrained: fit on augmented pairs, score on the raw pairs
    af, an = symmetrize_pairs(Yf, Yn, S)
    PsiAX, _ = poly_features(af, degree, inc)
    PsiAY, _ = poly_features(an, degree, inc)
    K_sym, *_ = np.linalg.lstsq(PsiAX, PsiAY, rcond=rcond)
    fit_sym = np.linalg.norm(PsiX @ K_sym - PsiY) / normY

    symmetry_cost = (fit_sym - fit_free) / (fit_free + 1e-30)

    # pullback matrix P (U_S in the dictionary basis)
    PsiSX, _ = poly_features(Yf @ S.T, degree, inc)
    P, *_ = np.linalg.lstsq(PsiX, PsiSX, rcond=rcond)
    pullback_fit = np.linalg.norm(PsiX @ P - PsiSX) / (np.linalg.norm(PsiSX) + 1e-30)
    closure = np.linalg.norm(np.linalg.matrix_power(P, order) - np.eye(P.shape[0]))
    commutator = (np.linalg.norm(P @ K_free - K_free @ P) /
                  (np.linalg.norm(K_free) + 1e-30))

    # per-eigenfunction symmetry-sector test on the free operator
    evals, xi = np.linalg.eig(K_free)
    order_idx = np.argsort(-np.abs(evals))
    evals, xi = evals[order_idx], xi[:, order_idx]
    roots = np.exp(2j * np.pi * np.arange(order) / order)
    rows, n_clean = [], 0
    n_check = min(12, xi.shape[1])
    for j in range(n_check):
        a = PsiX @ xi[:, j]
        b = PsiSX @ xi[:, j]
        denom = np.vdot(a, a)
        if abs(denom) < 1e-20:
            continue
        w = np.vdot(a, b) / denom
        rel = np.linalg.norm(b - w * a) / (np.linalg.norm(b) + 1e-30)
        nearest = roots[np.argmin(np.abs(roots - w))]
        clean = rel < 0.05 and abs(w - nearest) < 0.1
        n_clean += clean
        rows.append((float(abs(evals[j])), float(np.angle(evals[j])),
                     complex(w), float(rel), bool(clean)))

    return {
        "fit_free": float(fit_free),
        "fit_sym": float(fit_sym),
        "symmetry_cost": float(symmetry_cost),
        "commutator": float(commutator),
        "closure": float(closure),
        "pullback_fit": float(pullback_fit),
        "sector_frac": n_clean / max(1, n_check),
        "sector_table": rows,
        "evals_free": evals,
        "P_evals": np.linalg.eigvals(P),
    }


# ----------------------------------------------------------------------------
# top-level driver
# ----------------------------------------------------------------------------

def _pca(Y, k):
    Yc = Y - Y.mean(0, keepdims=True)
    U, s, Vt = np.linalg.svd(Yc, full_matrices=False)
    return Yc @ Vt[:k].T, s


def koopman_diagnostic(data, timesteps, candidate_S, order=2, degree=3,
                       pca_dim=6, verbose=True):
    """Run the Koopman symmetry diagnostic on a dataset dict with keys
    'z' (canonical latent), 'x' (observations) and optionally 'dz'.

    `candidate_S` is the linear group action in canonical z-coordinates
    (e.g. -np.eye(2) for the Duffing Z2, or diag(-1,-1,1) for Lorenz).

    Reports on:
      (A) canonical z            -- validates the diagnostic (should pass)
      (B) raw observations x     -- PCA-reduced; spectrum only (S unknown there)
      (A) vs (B) eigenvalues     -- spectral-invariance check
    """
    out = {}

    # datasets store either (n_ics*timesteps, dim) or (n_ics, timesteps, dim)
    z = np.asarray(data["z"], dtype=np.float64).reshape(-1, np.shape(data["z"])[-1])
    x = np.asarray(data["x"], dtype=np.float64).reshape(-1, np.shape(data["x"])[-1])

    # (A) canonical coordinates: separating test for the candidate action
    zf, zn = _pairs(z, timesteps)
    resz = edmd(zf, zn, degree=degree)
    symz = symmetry_report(zf, zn, candidate_S, order=order, degree=degree)
    out["z"] = {"edmd": resz, "sym": symz}

    # (B) observations: spectrum only (S is unknown in these coordinates)
    xr, svals = _pca(x, min(pca_dim, x.shape[1]))
    xf, xn = _pairs(xr, timesteps)
    resx = edmd(xf, xn, degree=min(degree, 2))   # keep dict small in >2 dims
    out["x"] = {"edmd": resx, "pca_singular_values": svals[:pca_dim]}

    if verbose:
        _print(out, order)
    return out


def _print(out, order):
    resz, symz = out["z"]["edmd"], out["z"]["sym"]
    print("=" * 70)
    print("KOOPMAN SYMMETRY DIAGNOSTIC")
    print("=" * 70)
    print(f"\n(A) canonical coordinates z")
    print(f"    leading |lambda|:            " +
          ", ".join(f"{abs(l):.3f}" for l in resz["evals"][:6]))
    print(f"    separating test")
    print(f"      fit  unconstrained        = {symz['fit_free']:.3e}")
    print(f"      fit  with S imposed       = {symz['fit_sym']:.3e}")
    print(f"      symmetry cost (rel.)      = {symz['symmetry_cost']:+.3e}"
          f"   (~0 => S is a real symmetry)")
    print(f"    secondary")
    print(f"      commutator ||PK-KP||/||K||= {symz['commutator']:.2e}")
    print(f"      order-{order} closure ||P^{order}-I||  = {symz['closure']:.2e}")
    print(f"      pullback fit              = {symz['pullback_fit']:.2e}")
    print(f"      clean symmetry sectors    = "
          f"{symz['sector_frac']*100:.0f}% of leading eigenfunctions (informational)")
    cost = symz["symmetry_cost"]
    verdict = ("SYMMETRY PRESENT" if cost < 0.02 else
               "WEAK / AMBIGUOUS" if cost < 0.1 else "NO CLEAN SYMMETRY")
    print(f"    --> {verdict}  (symmetry cost {cost:+.2e})")

    resx = out["x"]["edmd"]
    print(f"\n(B) raw observations x (PCA-reduced)   [EDMD fit resid = {resx['fit']:.2e}]")
    print(f"    PCA singular values: " +
          ", ".join(f"{s:.2e}" for s in out["x"]["pca_singular_values"]))
    print(f"    leading |lambda|:    " +
          ", ".join(f"{abs(l):.3f}" for l in resx["evals"][:6]))
    # spectral-invariance check: do the dominant rates match those from z?
    lz = np.sort_complex(resz["evals"][:4])
    lx = np.sort_complex(resx["evals"][:4])
    print(f"    spectral-invariance |lambda_z - lambda_x| (top 4): " +
          ", ".join(f"{abs(a-b):.2e}" for a, b in zip(lz, lx)))
    print("=" * 70)
