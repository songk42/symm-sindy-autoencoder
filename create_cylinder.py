"""
Generate the cylinder-wake 'coordinate problem' dataset: sparse pressure-probe
observations of the real (Firedrake-simulated) Navier-Stokes vortex-shedding
limit cycle behind a 2D cylinder at Re=100.

Physical symmetry: the flow equations + BCs are invariant under the mirror
reflection y -> -y (u -> u, v -> -v, p -> p unchanged since pressure is a true
scalar). This is a symmetry of the *vector field*, not of any one trajectory --
a point on the limit cycle maps under reflection to ANOTHER point on the same
limit cycle (a half-shedding-period away). That is exactly the kind of
commuting-vector-field symmetry F(Sz) = S F(z) that Phase B/C/D already test
(via the EDMD separating test on the *discovered* z = E(x), not by requiring
S to be a pointwise trajectory symmetry) -- no changes to the pipeline needed.

Unlike synth/lorenz_distort there is no known ground-truth latent z or
governing ODE here, so this dataset carries ONLY (x, dx) -- no 'z'/'dz' keys.
SINDyAE training only ever needs (x, dx) (dz is computed on the fly via
get_derivative); phase_d_eval.py's position-R^2 / equation-recovery sections
need 'z'/'dz' and so do not apply here -- use coordinate_diagnostic.py's
`latent_symmetry_check` instead, which is entirely self-referential (EDMD on
z = E(x) alone, no ground truth required).

MUST run inside the Firedrake environment, not the symm-sindy-autoencoder
.venv:
    ~/scratch/hydrogym/hg-firedrake python3 create_cylinder.py --role train --seed 42 [--tf 5 --smoke]
For the real (long) run, submit via sbatch:
    sbatch ~/scratch/hydrogym/firedrake-job.sbatch create_cylinder.py --role train --seed 42

One call = one independent CFD trajectory (its own random perturbation seed),
written to data/cylinder/_raw/{role}_seed{seed}.npy. Generate SEVERAL
independent trajectories per role (esp. >=2 for train) and run
combine_cylinder.py afterward to concatenate them into
data/cylinder/{train,val,test}.npy. This -- whole independent trajectories
assigned to a split, not time-slices of one trajectory -- is the fix for the
first cylinder_v1 attempt: a single short correlated trajectory gave a
held-out EDMD fit so good (near machine precision) that the separating-test's
relative-cost metric became numerically degenerate (denominator ~0), making
every candidate action -- right or wrong -- look catastrophically bad. Run
multiple sbatch jobs (different --seed, same or different --role) in
parallel; each is independent.

Pipeline (per trajectory):
  1. Newton-solve the unstable steady state (Re ramped 40->60->80->100).
  2. Perturb (using --seed) and integrate forward; vortex shedding grows into
     a limit cycle. Log pressure at `n_probes` points evenly spaced around
     the cylinder (mirror-symmetric about y=0 by construction) every step.
  3. Discard the transient (`t < t_discard`), central-difference the
     remaining pressure series for dx/dt, and slice into overlapping windows
     of length `timesteps` with stride `stride`.
  4. Save ALL of this trajectory's windows to
     data/cylinder/_raw/{role}_seed{seed}.npy (x, dx dict) -- combine_cylinder.py
     merges same-role files across seeds into the final split.
"""

import argparse
import os

import numpy as np
import psutil

import hydrogym.firedrake as hgym


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--re", type=float, default=100.0)
    ap.add_argument("--mesh", default="medium")
    ap.add_argument("--n_probes", type=int, default=20)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--tf", type=float, default=150.0, help="total integration time")
    ap.add_argument("--t_discard", type=float, default=50.0,
                    help="transient to discard before the flow saturates onto the limit cycle")
    ap.add_argument("--timesteps", type=int, default=250, help="window length (matches synth/lorenz TS)")
    ap.add_argument("--stride", type=int, default=25, help="hop between consecutive windows")
    ap.add_argument("--seed", type=int, default=42,
                    help="perturbation RNG seed -- defines an independent trajectory")
    ap.add_argument("--role", choices=("train", "val", "test"), default=None,
                    help="which split this trajectory belongs to (required unless --smoke)")
    ap.add_argument("--out_dir", default="data/cylinder")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny sanity run (short tf, coarse mesh acceptable) -- does NOT write a usable dataset")
    args = ap.parse_args()
    if not args.smoke and args.role is None:
        ap.error("--role is required (train/val/test) unless --smoke")
    return args


def main():
    args = parse_args()
    if args.smoke:
        args.tf = min(args.tf, 6.0)
        args.t_discard = 0.0

    raw_dir = f"{args.out_dir}/_raw"
    work_dir = f"{args.out_dir}/_sim"
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)
    tag = f"{args.role or 'smoke'}_seed{args.seed}"
    data_file = f"{work_dir}/pressure_{tag}.dat"

    R = 0.5
    probes = [(R * np.cos(th), R * np.sin(th))
              for th in np.linspace(0, 2 * np.pi, args.n_probes, endpoint=False)]

    flow = hgym.Cylinder(Re=args.re, mesh=args.mesh, velocity_order=2,
                         observation_type="pressure_probes", probes=probes,
                         use_HF_data_manager=False)

    # ---- Stage 1: steady state (Newton, Re-ramped for convergence) --------
    hgym.print("=" * 70)
    hgym.print("Stage 1: solving for the steady-state base flow")
    hgym.print("=" * 70)
    Re_init = [v for v in (40.0, 60.0, 80.0) if v < args.re] + [args.re]
    for Re_val in Re_init:
        flow.Re.assign(Re_val)
        hgym.print(f"  steady solve at Re={Re_val}")
        solver = hgym.NewtonSolver(flow, stabilization="none",
                                   solver_parameters={"snes_monitor": None})
        solver.solve()
    flow.qB = flow.q.copy(deepcopy=True)

    # ---- Stage 2: perturb + integrate, logging probes every step ----------
    hgym.print("\n" + "=" * 70)
    hgym.print(f"Stage 2: transient integration, tf={args.tf}, dt={args.dt}")
    hgym.print("=" * 70)

    import firedrake as fd
    rng = fd.RandomGenerator(fd.PCG64(seed=args.seed))
    flow.q += rng.normal(flow.mixed_space, 0.0, 1e-3)

    def log_postprocess(flow):
        mem = psutil.virtual_memory().percent
        p = flow.get_observations()
        return *p, mem

    log = hgym.io.LogCallback(
        postprocess=log_postprocess, nvals=args.n_probes + 1, interval=1,
        print_fmt="t: {0:0.2f}", filename=data_file,
    )
    hgym.integrate(flow, t_span=(0, args.tf), dt=args.dt, callbacks=[log],
                   method="BDF", stabilization="none")

    if args.smoke:
        hgym.print("\n[--smoke] ran end-to-end OK; not writing a dataset.")
        return

    # ---- Post-process: discard transient, FD derivative, window, split ----
    raw = np.loadtxt(data_file)
    t, p = raw[:, 0], raw[:, 1:1 + args.n_probes]
    keep = t >= args.t_discard
    t, p = t[keep], p[keep]
    hgym.print(f"\npost-transient series: {p.shape[0]} steps x {p.shape[1]} probes "
               f"(t in [{t[0]:.1f}, {t[-1]:.1f}])")

    dp = np.gradient(p, args.dt, axis=0)  # central differences, edge-forward/backward at ends

    TS, stride = args.timesteps, args.stride
    starts = list(range(0, len(p) - TS + 1, stride))
    if len(starts) < 3:
        raise SystemExit(f"only {len(starts)} windows of length {TS} fit in "
                         f"{len(p)} post-transient steps; lower --timesteps/--t_discard "
                         f"or raise --tf")
    X = np.stack([p[s:s + TS] for s in starts], 0).astype(np.float32)
    dX = np.stack([dp[s:s + TS] for s in starts], 0).astype(np.float32)

    out_path = f"{raw_dir}/{tag}.npy"
    np.save(out_path, {"x": X, "dx": dX})
    hgym.print(f"\nsaved {len(starts)} windows (this trajectory, role={args.role}, "
               f"seed={args.seed}) to {out_path}  x {X.shape}  dx {dX.shape}")
    hgym.print("run combine_cylinder.py once all seeds/roles are done to build "
              "the final train/val/test split.")


if __name__ == "__main__":
    main()
