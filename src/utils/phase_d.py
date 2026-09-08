"""
Phase D -- held-out separating evaluation + equation recovery.

Everything here runs on a *test* split that was never touched during training or
the Phase A / C diagnostics (those used val). Three questions:

  1. cost of the constraint      does forcing the discrete symmetry to act
                                 linearly degrade recon / dx / dz vs an
                                 unconstrained model of the same architecture?
                                 (`evaluate_model`, compared to a baseline)

  2. separating test             on z = E(x_test), the discovered action must
                                 pass the Koopman separating test AND every
                                 wrong linear action of the same size must fail
                                 (`separating_battery`).  A flexible encoder can
                                 manufacture *one* apparent symmetry; it cannot
                                 make an arbitrary basket of them all hold.

  3. equation recovery           fit the linear gauge A taking z = E(x) onto the
                                 true latent coordinates, read the dynamics back
                                 in that de-gauged frame, and compare the
                                 recovered polynomial to the generating ODE
                                 (`linear_gauge_fit`, `recover_equation`).
"""

import numpy as np
import torch

from src.utils.koopman import (poly_features, _pairs, symmetry_report,
                               _monomial_exponents)


# --------------------------------------------------------------------------- #
# 1. coordinate-map quality on held-out data
# --------------------------------------------------------------------------- #

@torch.no_grad()
def _encode(net, X, device, batch=8192):
    net.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.as_tensor(X[i:i + batch], dtype=torch.float32, device=device)
        out.append(net.encode(xb).cpu().numpy())
    return np.concatenate(out, 0)


def evaluate_model(net, data, device, batch=8192):
    """Raw (unweighted) recon / dz / dx MSE on a flat dataset dict.

    Mirrors SINDyAE.forward: dz is the encoded velocity J_E(x) xdot, dz_pred is
    Theta(z) Xi_masked, dx_pred is the decoder-propagated dz_pred.
    """
    net.eval()
    X = np.asarray(data["x"], dtype=np.float32)
    dX = np.asarray(data["dx"], dtype=np.float32)
    n = len(X)
    se_recon = se_dz = se_dx = 0.0
    cnt_u = cnt_z = 0
    from src.utils.model_utils import sindy_library
    for i in range(0, n, batch):
        xb = torch.as_tensor(X[i:i + batch], dtype=torch.float32, device=device)
        dxb = torch.as_tensor(dX[i:i + batch], dtype=torch.float32, device=device)
        z = net.encode(xb)
        x_recon = net.decoder(z)
        theta = sindy_library(z, net.poly_order, device, net.use_sine,
                              net.include_constant)
        dz = net.get_derivative(xb, dxb, net.encoder)
        xi = net.effective_xi() if hasattr(net, "effective_xi") \
            else net.sindy_coefficients * net.threshold_mask
        dz_pred = torch.matmul(theta, xi)
        dx_pred = net.get_derivative(z, dz_pred, net.decoder)
        se_recon += ((x_recon - xb) ** 2).sum().item()
        se_dz += ((dz_pred - dz) ** 2).sum().item()
        se_dx += ((dx_pred - dxb) ** 2).sum().item()
        cnt_u += xb.numel()
        cnt_z += dz.numel()
    return {
        "recon_mse": se_recon / cnt_u,
        "dz_mse": se_dz / cnt_z,
        "dx_mse": se_dx / cnt_u,
        "n": n,
    }


# --------------------------------------------------------------------------- #
# 2. separating battery -- discovered action vs wrong ones
# --------------------------------------------------------------------------- #

def candidate_actions(d):
    """A basket of finite linear actions on R^d to run the separating test on.

    Every entry is (name, matrix, nominal order, is_trivial).  `is_trivial`
    flags identity, which passes the separating test by construction (augmenting
    with (I y_t, I y_{t+1}) just duplicates the pairs) -- kept as a sanity anchor,
    not evidence.
    """
    I = np.eye(d)
    out = [("identity  (+I)", I.copy(), 1, True)]

    if d == 2:
        out += [
            ("inversion (-I)       [discovered]", -I.copy(), 2, False),
            ("flip z1   diag(-1,1)", np.diag([-1.0, 1.0]), 2, False),
            ("flip z2   diag(1,-1)", np.diag([1.0, -1.0]), 2, False),
            ("swap      [[0,1],[1,0]]", np.array([[0.0, 1.0], [1.0, 0.0]]), 2, False),
            ("rot 90deg [[0,-1],[1,0]]", np.array([[0.0, -1.0], [1.0, 0.0]]), 4, False),
            ("shear     [[1,.5],[0,1]]", np.array([[1.0, 0.5], [0.0, 1.0]]), 2, False),
        ]
    elif d == 3:
        out += [
            ("Lorenz Z2 diag(-1,-1,1) [discovered]", np.diag([-1.0, -1.0, 1.0]), 2, False),
            ("full inversion -I", -I.copy(), 2, False),
            ("flip z   diag(1,1,-1)", np.diag([1.0, 1.0, -1.0]), 2, False),
            ("flip x   diag(-1,1,1)", np.diag([-1.0, 1.0, 1.0]), 2, False),
            ("swap xy  P(0,1)", np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]]), 2, False),
            ("rot90 xy", np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]), 4, False),
        ]
    else:
        out += [("inversion (-I)  [discovered]", -I.copy(), 2, False)]
    return out


def _verdict(cost):
    return ("LINEARIZED" if cost < 0.02 else
            "WEAK" if cost < 0.10 else "NOT A SYMMETRY")


def separating_battery(Z, timesteps, d, degree=3):
    zf, zn = _pairs(np.asarray(Z, dtype=np.float64), timesteps)
    rows = []
    for name, S, order, trivial in candidate_actions(d):
        rep = symmetry_report(zf, zn, S, order=order, degree=degree)
        rows.append({
            "name": name, "trivial": trivial,
            "symmetry_cost": rep["symmetry_cost"],
            "fit_free": rep["fit_free"], "fit_sym": rep["fit_sym"],
            "closure": rep["closure"],
            "verdict": _verdict(rep["symmetry_cost"]),
        })
    return rows


def battery_pass(rows, margin=3.0, wrong_floor=0.05):
    """Phase D passes the separating test iff the discovered action is LINEARIZED
    and every non-trivial wrong action is clearly worse."""
    disc = next(r for r in rows if "discovered" in r["name"])
    wrong = [r for r in rows if not r["trivial"] and "discovered" not in r["name"]]
    disc_ok = disc["symmetry_cost"] < 0.02
    sep_ok = all(w["symmetry_cost"] > max(wrong_floor, margin * disc["symmetry_cost"])
                 for w in wrong)
    return {
        "pass": bool(disc_ok and sep_ok),
        "discovered_cost": disc["symmetry_cost"],
        "min_wrong_cost": min((w["symmetry_cost"] for w in wrong), default=float("nan")),
        "disc_ok": bool(disc_ok), "sep_ok": bool(sep_ok),
    }


# --------------------------------------------------------------------------- #
# 3. equation recovery
# --------------------------------------------------------------------------- #

def linear_gauge_fit(Zhat, Ztrue):
    """Least-squares affine map  Ztrue ~ Zhat @ Alin^T + b.

    Returns Alin (d,d), b (d,), per-component R^2, and the de-gauged coordinates
    Zdeg = Zhat @ Alin^T + b.  High R^2 => E is the true chart up to a linear
    gauge (the coordinate problem is solved); low R^2 => E linearized *a*
    symmetry but not onto the generating coordinates.
    """
    Zhat = np.asarray(Zhat, np.float64)
    Ztrue = np.asarray(Ztrue, np.float64)
    A = np.hstack([Zhat, np.ones((len(Zhat), 1))])           # (N, d+1)
    M, *_ = np.linalg.lstsq(A, Ztrue, rcond=None)            # (d+1, d)
    pred = A @ M
    ss_res = ((Ztrue - pred) ** 2).sum(0)
    ss_tot = ((Ztrue - Ztrue.mean(0)) ** 2).sum(0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-30)
    d = Zhat.shape[1]
    Alin, b = M[:d].T, M[d]
    return {"Alin": Alin, "b": b, "r2": r2, "Zdeg": pred}


def lib_names(d, degree, include_constant=True):
    """Human-readable monomial names in the poly_features ordering."""
    v = [f"z{i+1}" for i in range(d)]
    out = []
    for e in _monomial_exponents(d, degree, include_constant):
        if sum(e) == 0:
            out.append("1")
        else:
            out.append(" ".join(f"{v[k]}^{p}" if p > 1 else v[k]
                                for k, p in enumerate(e) if p))
    return out


def true_duffing_coeffs(delta=0.1):
    """Generating ODE  z1' = z2 ;  z2' = z1 - z1^3 - delta z2  in poly_features
    ordering (10 x 2)."""
    C = np.zeros((10, 2))
    C[2, 0] = 1.0                       # z1' = z2
    C[1, 1] = 1.0                       # z2' = +z1
    C[6, 1] = -1.0                      # z2' = ... - z1^3
    C[2, 1] = -delta                    # z2' = ... - delta z2
    return C


def induced_library_matrix(S, degree, include_constant=True, seed=0):
    """C(S) with  Theta(z) C(S) = Theta(S z),  solved on generic sample points
    (numpy analogue of src.utils.symmetry.InducedLibraryRep)."""
    S = np.asarray(S, np.float64)
    d = S.shape[0]
    L = len(_monomial_exponents(d, degree, include_constant))
    rng = np.random.RandomState(seed)
    Z = rng.randn(2 * L + 8, d)
    B, _ = poly_features(Z, degree, include_constant)
    A, _ = poly_features(Z @ S.T, degree, include_constant)
    C, *_ = np.linalg.lstsq(B, A, rcond=None)
    return C


def equivariance_residual_np(Xi, S, degree, include_constant=True):
    """|| C(S) Xi - Xi S^T ||_F / || Xi ||_F  -- relative intertwining error of
    the discovered dynamics under S.  ~0 => S acts linearly on the latent
    vector field (for S = -I this is the 'Xi is purely odd' check; for
    mixed-parity S it correctly allows the even terms in the invariant rows)."""
    Xi = np.asarray(Xi, np.float64)
    C = induced_library_matrix(S, degree, include_constant)
    lhs = C @ Xi
    rhs = Xi @ np.asarray(S, np.float64).T
    return float(np.linalg.norm(lhs - rhs) / (np.linalg.norm(Xi) + 1e-30))


def recover_equation(Zdeg, dZtrue, degree=3, threshold=0.05, true_coeffs=None):
    """Fit  dZtrue ~ Theta(Zdeg) C  by least squares and hard-threshold.  If
    `true_coeffs` (in poly_features ordering) is given, also report the max
    coefficient error against it."""
    Zdeg = np.asarray(Zdeg, np.float64)
    dZtrue = np.asarray(dZtrue, np.float64)
    Psi, _ = poly_features(Zdeg, degree, include_constant=True)
    C, *_ = np.linalg.lstsq(Psi, dZtrue, rcond=None)
    resid = np.linalg.norm(Psi @ C - dZtrue) / (np.linalg.norm(dZtrue) + 1e-30)
    Ct = C.copy()
    Ct[np.abs(Ct) < threshold] = 0.0
    resid_t = np.linalg.norm(Psi @ Ct - dZtrue) / (np.linalg.norm(dZtrue) + 1e-30)
    d = Zdeg.shape[1]
    coeff_err = (float(np.abs(Ct - np.asarray(true_coeffs, np.float64)).max())
                 if true_coeffs is not None else None)
    return {
        "coeffs": C, "coeffs_thresholded": Ct, "coeffs_true": true_coeffs,
        "names": lib_names(d, degree), "fit_resid": float(resid),
        "fit_resid_thresholded": float(resid_t), "coeff_err": coeff_err,
    }


def parity_structure(Xi):
    """max |even-degree coeff| / max |odd-degree coeff| for a (10, d) Xi in the
    poly_order-3 2D library.  ~0 => the discovered dynamics are purely odd
    (the frozen -I is exactly linearized on Xi).  Returns None for d != 2."""
    Xi = np.asarray(Xi, np.float64)
    if Xi.shape[0] != 10:
        return None
    even = [0, 3, 4, 5]           # 1, z1^2, z1 z2, z2^2
    odd = [1, 2, 6, 7, 8, 9]      # z1, z2, z1^3, z1^2 z2, z1 z2^2, z2^3
    num = np.abs(Xi[even]).max()
    den = np.abs(Xi[odd]).max() + 1e-30
    return float(num / den)


def _eq_str(C, names, start):
    terms = [f"{C[i]:+.3f}·{names[i]}" for i in range(len(C)) if C[i] != 0]
    return start + (" ".join(terms) if terms else "0")


def own_chart_equation(Xi, degree=3):
    """The governing equation SINDy actually discovered, in the model's own
    latent chart z_hat = E(x) -- i.e. the masked Xi read symbolically.  This is
    reported separately from the de-gauged-to-truth comparison because the
    encoder chart need not be linearly related to the generating coordinates."""
    Xi = np.asarray(Xi, np.float64)
    d = Xi.shape[1]
    names = lib_names(d, degree)
    if len(names) != Xi.shape[0]:
        names = [f"t{k}" for k in range(Xi.shape[0])]
    nnz = int((np.abs(Xi) > 0).sum())
    return {"names": names, "nnz": nnz, "Xi": Xi}


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #

def run_phase_d(net, test_data, timesteps, degree=3, baseline_net=None,
                true_coeffs=None, verbose=True):
    device = next(net.parameters()).device
    d = net.z_dim
    S_disc = next(M for name, M, *_ in candidate_actions(d) if "discovered" in name)

    ev = evaluate_model(net, test_data, device)
    base_ev = evaluate_model(baseline_net, test_data,
                             next(baseline_net.parameters()).device) \
        if baseline_net is not None else None

    Zhat = _encode(net, np.asarray(test_data["x"], np.float32), device)
    rows = separating_battery(Zhat, timesteps, d, degree=degree)
    verdict = battery_pass(rows)

    xi_t = net.effective_xi() if hasattr(net, "effective_xi") \
        else net.threshold_mask * net.sindy_coefficients
    Xi = xi_t.detach().cpu().numpy()
    parity = parity_structure(Xi)
    equiv_resid = equivariance_residual_np(Xi, S_disc, degree)
    own = own_chart_equation(Xi, degree=degree)

    # sections needing a ground-truth latent (z/dz) -- absent for datasets with
    # no known reduced-order model (e.g. cylinder). Everything above this line
    # (cost of the constraint, the separating test, the own-chart equation) is
    # fully self-referential and does not need ground truth.
    has_truth = ("z" in test_data) and ("dz" in test_data)
    gauge = rec = None
    push_mse = push_rel = None
    if has_truth:
        gauge = linear_gauge_fit(Zhat, np.asarray(test_data["z"], np.float64))
        rec = recover_equation(gauge["Zdeg"], np.asarray(test_data["dz"], np.float64),
                               degree=degree, true_coeffs=true_coeffs)

        # the *discovered* model's dynamics, pushed to true coords by the linear gauge
        Psi_hat, _ = poly_features(Zhat.astype(np.float64), degree, include_constant=True)
        dz_hat = Psi_hat @ Xi
        dz_true_pred = dz_hat @ gauge["Alin"].T
        dz_true_np = np.asarray(test_data["dz"], np.float64)
        push_mse = float(((dz_true_pred - dz_true_np) ** 2).mean())
        # relative version (fraction of ||dz_true|| unexplained) so it reads on
        # the same 0..1+ scale as (1 - R^2) and fit_resid -- "how close to raw data"
        push_rel = float(np.linalg.norm(dz_true_pred - dz_true_np)
                         / (np.linalg.norm(dz_true_np) + 1e-30))

    out = {
        "eval": ev, "baseline_eval": base_ev,
        "battery": rows, "battery_verdict": verdict,
        "gauge": gauge, "recovery": rec, "parity_ratio": parity,
        "equiv_residual": equiv_resid, "own_chart": own, "Xi": Xi,
        "pushforward_dz_mse": push_mse, "pushforward_dz_rel": push_rel, "d": d,
        "has_truth": has_truth,
    }
    if verbose:
        _print(out)
    return out


def _print(out):
    ev, base = out["eval"], out["baseline_eval"]
    print("=" * 72)
    print("PHASE D  --  held-out separating evaluation + equation recovery")
    print("=" * 72)

    print(f"\n[1] coordinate-map quality on the TEST split  (n = {ev['n']})")
    hdr = f"      {'':22s}{'recon':>12s}{'dz':>12s}{'dx':>12s}"
    print(hdr)
    print(f"      {'constrained (this)':22s}{ev['recon_mse']:>12.3e}"
          f"{ev['dz_mse']:>12.3e}{ev['dx_mse']:>12.3e}")
    if base is not None:
        print(f"      {'unconstrained (base)':22s}{base['recon_mse']:>12.3e}"
              f"{base['dz_mse']:>12.3e}{base['dx_mse']:>12.3e}")
        r = lambda a, b: a / max(b, 1e-30)
        print(f"      {'ratio constr / base':22s}{r(ev['recon_mse'], base['recon_mse']):>12.2f}"
              f"{r(ev['dz_mse'], base['dz_mse']):>12.2f}{r(ev['dx_mse'], base['dx_mse']):>12.2f}")
        print("      (ratio ~1 => the linear-symmetry constraint is free; "
              ">>1 => it costs fit)")

    print(f"\n[2] separating test on z = E(x_test)   "
          f"[verdict bands: <0.02 linearized, <0.10 weak]")
    print(f"      {'action':38s}{'fit_free':>10s}{'fit_S':>10s}{'cost':>11s}   verdict")
    for r in out["battery"]:
        tag = "  (trivial)" if r["trivial"] else ""
        print(f"      {r['name']:38s}{r['fit_free']:>10.3e}{r['fit_sym']:>10.3e}"
              f"{r['symmetry_cost']:>+11.3e}   {r['verdict']}{tag}")
    v = out["battery_verdict"]
    print(f"\n      discovered cost {v['discovered_cost']:+.3e} | "
          f"min wrong cost {v['min_wrong_cost']:+.3e}")
    print(f"      discovered linearized: {v['disc_ok']}   "
          f"wrong actions all rejected: {v['sep_ok']}")
    print(f"      --> SEPARATING TEST {'PASS' if v['pass'] else 'FAIL'}")

    g, rec, own = out["gauge"], out["recovery"], out["own_chart"]
    d = out["d"]
    rn = [f"z{i+1}' = " for i in range(d)]
    print(f"\n[3] equation recovery")

    print(f"\n   (a) governing equation SINDy discovered, in the model's own "
          f"latent chart z = E(x)   [{own['nnz']} nonzero terms]:")
    Xi, names = own["Xi"], own["names"]
    for i in range(d):
        print("        " + _eq_str(Xi[:, i], names, rn[i]))
    print(f"        intertwining residual ||C(S)Xi - Xi S^T|| / ||Xi||  = "
          f"{out['equiv_residual']:.2e}   (~0 => S acts linearly on the discovered field)")
    if out["parity_ratio"] is not None:
        print(f"        (d=2 parity  max|even| / max|odd|  = {out['parity_ratio']:.2e})")

    if not out["has_truth"]:
        print(f"\n   (b)/(c) skipped -- no ground-truth latent (z/dz) for this "
              f"dataset (e.g. cylinder has no known reduced-order model). The "
              f"separating test above is fully self-referential and does not "
              f"need one.")
        print(f"\n[4] closeness to the raw (true) data")
        v = out["battery_verdict"]
        print(f"      symmetry discovery (separating test) : "
              f"{'PASS' if v['pass'] else 'FAIL'}")
        print(f"      position/dynamics closeness          : N/A (no ground truth)")
        print("=" * 72)
        return

    print(f"\n   (b) is E the true chart up to a LINEAR gauge?  "
          f"fit z_true ~ A z_hat + b:")
    print(f"        R^2 per component = "
          + ", ".join(f"{v:.4f}" for v in g["r2"])
          + "   (->1 => coordinate problem solved; <1 => residual nonlinear distortion)")
    with np.printoptions(precision=3, suppress=True):
        print(f"        A = {g['Alin'].tolist()}   b = {g['b'].tolist()}")

    print(f"\n   (c) dynamics re-fit in the de-gauged frame"
          + ("  vs the generating ODE" if rec["coeff_err"] is not None else "")
          + f"   (only meaningful if (b) R^2 ~ 1):")
    Ct, tnames = rec["coeffs_thresholded"], rec["names"]
    for i in range(d):
        print("        " + _eq_str(Ct[:, i], tnames, rn[i]))
    if rec["coeffs_true"] is not None:
        Ctr = np.asarray(rec["coeffs_true"], np.float64)
        for i in range(d):
            print("        true:  " + _eq_str(Ctr[:, i], tnames, rn[i]))
    ce = rec["coeff_err"]
    print(f"        fit residual (rel) = {rec['fit_resid_thresholded']:.3e}   "
          + (f"max|coeff - true| = {ce:.3e}   " if ce is not None else "")
          + f"pushforward dz MSE = {out['pushforward_dz_mse']:.3e}"
          + f"  (rel = {out['pushforward_dz_rel']:.3e})")

    # ---- summary: how close is this to the raw (true) data? --------------- #
    # two independent closeness numbers, both on a 0 (perfect) .. 1+ (useless)
    # scale: position (is z_hat = E(x) the true chart up to a linear gauge?)
    # and dynamics (does the recovered ODE, re-fit in that gauge, reproduce the
    # true dz?). They are reported separately on purpose -- a chart can be
    # ~linearly correct in position while its *derivative* structure is not
    # polynomial in the true coordinates (residual chart error is amplified by
    # differentiation), so position closeness alone overstates recovery.
    v = out["battery_verdict"]
    disc = "PASS" if v["pass"] else "FAIL"
    r2min = float(min(g["r2"]))
    pos_err = max(0.0, 1.0 - r2min)
    dyn_err = rec["fit_resid_thresholded"]
    print(f"\n[4] closeness to the raw (true) data")
    print(f"      symmetry discovery (separating test)        : {disc}")
    print(f"      position error  (1 - min R^2, z_hat vs z_true, linear gauge) : {pos_err:.1%}")
    print(f"      dynamics error  (rel STLSQ fit resid in that gauge)          : {dyn_err:.1%}")
    print(f"      pushforward dz error (rel, model's own dz pushed to truth)   : {out['pushforward_dz_rel']:.1%}")
    if pos_err < 0.02 and dyn_err < 0.05:
        coord = "STRONG -- chart is the generator's, up to linear gauge"
    elif pos_err < 0.2:
        coord = "PARTIAL -- chart ~linear to truth in position, but dynamics not recovered"
    else:
        coord = "WEAK -- chart not linearly related to the true coordinates"
    print(f"      coordinate recovery                          : {coord}")
    print("=" * 72)
