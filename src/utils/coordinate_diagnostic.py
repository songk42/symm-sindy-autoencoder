"""
Phase A diagnostics for the coordinate problem.

Given a trained encoder E: R^u -> R^d and observation samples, quantify whether
E is a well-conditioned coordinate map (a smooth immersion onto its image, no
dimensional collapse, bounded distortion) and whether it has linearized the
target discrete symmetry.

    jacobian_rank_stats   -- singular values of dz/dx; full-rank fraction, worst
                             conditioning. rank < d  =>  dimensional collapse
    bilipschitz_stats     -- spread of ||E(x_i) - E(x_j)|| / ||x_i - x_j|| over
                             sample pairs; ratio hi/lo is the bi-Lipschitz const
    latent_symmetry_check -- Koopman separating test (see src.utils.koopman) run
                             on z = E(x): does the candidate linear action
                             commute with the dynamics in the learned coords?
"""

import numpy as np
import torch

from src.utils.koopman import symmetry_report, _pairs


@torch.no_grad()
def encode_all(net, X, device=None, batch=4096):
    device = device or next(net.parameters()).device
    net.eval()
    Z = []
    for i in range(0, len(X), batch):
        xb = torch.as_tensor(X[i:i + batch], dtype=torch.float32, device=device)
        Z.append(net.encode(xb).cpu().numpy())
    return np.concatenate(Z, 0)


def encoder_jacobian(net, X, n_samples=256, seed=0, device=None):
    """(n_samples, d, u) stack of dz/dx Jacobians at random sample rows."""
    device = device or next(net.parameters()).device
    net.eval()
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    enc = net.encode
    Js = []
    for i in idx:
        x = torch.as_tensor(X[i], dtype=torch.float32, device=device)
        J = torch.autograd.functional.jacobian(enc, x, vectorize=True)
        Js.append(J.detach().cpu().numpy())
    return np.stack(Js, 0)


def jacobian_rank_stats(J, tol=1e-6):
    """J: (m, d, u).  Reports per-point singular values of dz/dx."""
    d = J.shape[1]
    sv = np.linalg.svd(J, compute_uv=False)          # (m, d)
    smin, smax = sv[:, -1], sv[:, 0]
    ranks = (sv > tol * smax[:, None]).sum(1)
    return {
        "d": int(d),
        "sv_mean": sv.mean(0).tolist(),
        "min_singular_value": float(smin.min()),
        "worst_condition": float((smax / np.maximum(smin, 1e-30)).max()),
        "median_condition": float(np.median(smax / np.maximum(smin, 1e-30))),
        "full_rank_frac": float((ranks == d).mean()),
    }


def bilipschitz_stats(X, Z, n_pairs=20000, seed=0):
    rng = np.random.RandomState(seed)
    i = rng.randint(0, len(X), n_pairs)
    j = rng.randint(0, len(X), n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    dx = np.linalg.norm(X[i] - X[j], axis=1)
    dz = np.linalg.norm(Z[i] - Z[j], axis=1)
    m = dx > 1e-9
    r = dz[m] / dx[m]
    lo, hi = np.percentile(r, [1, 99])          # robust to outliers
    return {
        "ratio_p1": float(lo),
        "ratio_p99": float(hi),
        "bilipschitz_const": float(hi / max(lo, 1e-30)),
        "ratio_median": float(np.median(r)),
    }


def latent_symmetry_check(Z, timesteps, S, order=2, degree=3):
    zf, zn = _pairs(np.asarray(Z, dtype=np.float64), timesteps)
    return symmetry_report(zf, zn, S, order=order, degree=degree)


def run_diagnostic(net, X, timesteps, S, order=2, degree=3, verbose=True):
    Z = encode_all(net, X)
    J = encoder_jacobian(net, X)
    out = {
        "recon": None,
        "jacobian": jacobian_rank_stats(J),
        "bilipschitz": bilipschitz_stats(X, Z),
        "latent_symmetry": latent_symmetry_check(Z, timesteps, S, order, degree),
    }
    with torch.no_grad():
        xb = torch.as_tensor(X[:4096], dtype=torch.float32,
                             device=next(net.parameters()).device)
        out["recon"] = float(((net.decoder(net.encode(xb)) - xb) ** 2).mean())

    # Phase C: if the model learned generators, report them and score the
    # learned T (not just the prescribed S) with the same separating test
    gens = getattr(net, "symmetry_generators", lambda: [])()
    if gens:
        learned = []
        for T in gens:
            Tn = T.detach().cpu().numpy().astype(np.float64)
            rep = latent_symmetry_check(Z, timesteps, Tn, order, degree)
            learned.append({
                "T": Tn,
                "det": float(np.linalg.det(Tn)),
                "order_residual": float(np.linalg.norm(
                    np.linalg.matrix_power(Tn, order) - np.eye(Tn.shape[0]))),
                "dist_to_I": float(np.linalg.norm(Tn - np.eye(Tn.shape[0]))),
                "dist_to_S": float(np.linalg.norm(Tn - np.asarray(S))),
                "symmetry_cost": rep["symmetry_cost"],
                "fit_free": rep["fit_free"], "fit_sym": rep["fit_sym"],
            })
        out["learned_generators"] = learned

    if verbose:
        _print(out)
    return out


def _print(out):
    j, b, s = out["jacobian"], out["bilipschitz"], out["latent_symmetry"]
    print("=" * 70)
    print("COORDINATE-MAP DIAGNOSTIC (Phase A)")
    print("=" * 70)
    print(f"\nreconstruction MSE (val sample)   = {out['recon']:.3e}")
    print(f"\nencoder Jacobian dz/dx  (d = {j['d']})")
    print(f"  mean singular values            = "
          + ", ".join(f"{v:.3e}" for v in j['sv_mean']))
    print(f"  min singular value (any point)  = {j['min_singular_value']:.3e}"
          f"   (near 0 => dimensional collapse)")
    print(f"  full-rank fraction              = {j['full_rank_frac']*100:.1f}%")
    print(f"  condition number  median/worst  = {j['median_condition']:.2e} / {j['worst_condition']:.2e}")
    print(f"\nbi-Lipschitz behaviour of E  (||dz|| / ||dx|| over sample pairs)")
    print(f"  1st / 99th pct ratio            = {b['ratio_p1']:.3e} / {b['ratio_p99']:.3e}")
    print(f"  bi-Lipschitz constant (p99/p1)  = {b['bilipschitz_const']:.2f}"
          f"   (closer to 1 => better conditioned)")
    print(f"\nlatent symmetry (Koopman separating test on z = E(x))")
    print(f"  prescribed S: fit free / with S = {s['fit_free']:.3e} / {s['fit_sym']:.3e}")
    print(f"  prescribed S: symmetry cost     = {s['symmetry_cost']:+.3e}")
    print(f"  order closure ||P^p - I||       = {s['closure']:.2e}")

    def _verdict(cost):
        return ("SYMMETRY LINEARIZED" if cost < 0.02 else
                "WEAK / AMBIGUOUS" if cost < 0.1 else
                "SYMMETRY NOT LINEARIZED")

    for i, g in enumerate(out.get("learned_generators", [])):
        print(f"\nlearned generator T[{i}]  (det {g['det']:+.3f}, "
              f"||T^p - I|| {g['order_residual']:.2e}, "
              f"||T - I|| {g['dist_to_I']:.3f}, ||T - S|| {g['dist_to_S']:.3f})")
        with np.printoptions(precision=3, suppress=True):
            for row in g["T"]:
                print("      ", row)
        print(f"  learned T: fit free / with T   = {g['fit_free']:.3e} / {g['fit_sym']:.3e}")
        print(f"  learned T: symmetry cost       = {g['symmetry_cost']:+.3e}"
              f"   --> {_verdict(g['symmetry_cost'])}")

    print(f"\n  --> (prescribed S) {_verdict(s['symmetry_cost'])}")
    print("=" * 70)
