"""Formal comparison of where the prior enters a drift-diffusion process, with held-out likelihood and model recovery.

Model (per session, both colours jointly; upper boundary = favoured side):
  drift  v = v_c (one free drift per |coherence| level 13/35/100, signed toward the favoured side) + v0 + dv * pos
  start  w = z0 + dz * pos   (relative start, 0..1);   boundary a, non-decision time t0; 2% uniform contaminant over [0, 5] s.
  pos = 1 for the prior colour, 0 for the equal colour.
Variants: none (dv = dz = 0), z (dz free), v (dv free), zv (both).
This is the plain Wiener model (no leak/urgency/across-trial variability), fitted by exact likelihood (Navarro & Fuss 2009),
so it supports likelihood-based comparison; it is a simplification of the leaky/urgency model in scripts/ddm.
usage: python -m scripts.audit.a13_ddm_compare [fit|recover]"""
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import optimize

from scripts.audit.common import TWO_PRIOR, trials
from src.data_io import catalog

LAPSE, TMAX = 0.02, 5.0


def wiener_lower(t, v, a, w):
    """Density of hitting the lower boundary at time t (vectorised)."""
    u = t / a ** 2
    out = np.empty_like(u)
    small = u < 1.0
    if small.any():
        us, ws = u[small], w[small]
        k = np.arange(-6, 7)[:, None]
        s = ((ws + 2 * k) * np.exp(-((ws + 2 * k) ** 2) / (2 * us))).sum(0)
        out[small] = s / np.sqrt(2 * np.pi * us ** 3)
    if (~small).any():
        ul, wl = u[~small], w[~small]
        k = np.arange(1, 16)[:, None]
        out[~small] = np.pi * (k * np.exp(-(k ** 2) * np.pi ** 2 * ul / 2) * np.sin(k * np.pi * wl)).sum(0)
    return np.maximum(out, 1e-300) / a ** 2 * np.exp(-v * a * w - v ** 2 * t / 2)


PARAMS = {"none": [], "z": ["dz"], "v": ["dv"], "zv": ["dz", "dv"]}
BASE = ["a", "t0", "z0", "v13", "v35", "v100", "v0"]
BOUNDS = {"a": (0.3, 4.0), "t0": (0.05, 0.9), "z0": (0.15, 0.85), "v13": (-1, 6), "v35": (-1, 10), "v100": (0, 20), "v0": (-3, 3), "dz": (-0.3, 0.3), "dv": (-3, 3)}


def unpack(theta, names):
    p = dict(zip(names, theta))
    p.setdefault("dz", 0.0); p.setdefault("dv", 0.0)
    return p


def loglik(theta, names, D):
    p = unpack(theta, names)
    rt, up, sc, pos, lev = D["rt"], D["up"], D["sc"], D["pos"], D["lev"]
    vlev = np.select([lev == 13, lev == 35, lev == 100], [p["v13"], p["v35"], p["v100"]], 0.0)
    v = vlev * sc + p["v0"] + p["dv"] * pos
    w = np.clip(p["z0"] + p["dz"] * pos, 0.02, 0.98)
    t = rt - p["t0"]
    dens = np.zeros_like(rt)
    ok = t > 0
    # upper boundary: flip drift and start
    vv = np.where(up, -v, v); ww = np.where(up, 1 - w, w)
    dens[ok] = wiener_lower(t[ok], vv[ok], p["a"], ww[ok])
    return np.log((1 - LAPSE) * dens + LAPSE / (2 * TMAX))


def fit(names, D, starts=4, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    bnds = [BOUNDS[n] for n in names]
    for s in range(starts):
        x0 = np.array([rng.uniform(lo + 0.2 * (hi - lo), hi - 0.2 * (hi - lo)) for lo, hi in bnds])
        if s == 0:
            x0 = np.array([{"a": 1.5, "t0": 0.3, "z0": 0.5, "v13": 0.5, "v35": 2.0, "v100": 5.0, "v0": 0.0, "dz": 0.0, "dv": 0.0}[n] for n in names])
        r = optimize.minimize(lambda x: -loglik(x, names, D).sum(), x0, method="L-BFGS-B", bounds=bnds)
        if best is None or r.fun < best.fun:
            best = r
    return best


def session_data(d):
    return {"rt": d.reaction_time.to_numpy() / 1000.0, "up": d.choice.to_numpy() == 1, "sc": np.where(d.target == 1, 1.0, -1.0) * (d.coherence.to_numpy() > 0),
            "pos": (d.color == 1).to_numpy().astype(float), "lev": d.coherence.to_numpy()}


def subset(D, m):
    return {k: v[m] for k, v in D.items()}


def run_session(args):
    sid, d = args
    d = d[(d.reaction_time / 1000 < TMAX)]
    D = session_data(d)
    n = len(d)
    rng = np.random.default_rng(abs(hash(sid)) % 2 ** 32)
    folds = rng.permutation(n) % 5
    out = {"session_id": sid, "n": n}
    for mname, extra in PARAMS.items():
        names = BASE + extra
        full = fit(names, D)
        out[f"ll_{mname}"] = -full.fun
        out[f"k_{mname}"] = len(names)
        p = unpack(full.x, names)
        for kk in ("dz", "dv", "z0", "v0", "a", "t0"):
            out[f"{mname}_{kk}"] = p[kk]
        cv = 0.0
        for f in range(5):
            tr, te = folds != f, folds == f
            r = fit(names, subset(D, tr), starts=2, seed=f)
            cv += loglik(r.x, names, subset(D, te)).sum()
        out[f"cv_{mname}"] = cv
    return out


def simulate(p, D, rng, dt=0.001):
    """Euler simulation of the Wiener process for the session's trial list under parameters p (dict)."""
    lev, sc, pos = D["lev"], D["sc"], D["pos"]
    vlev = np.select([lev == 13, lev == 35, lev == 100], [p["v13"], p["v35"], p["v100"]], 0.0)
    v = vlev * sc + p["v0"] + p["dv"] * pos
    x = (p["z0"] + p["dz"] * pos) * p["a"]
    n = len(v); rt = np.full(n, np.nan); up = np.zeros(n, bool); alive = np.ones(n, bool); t = 0.0
    while alive.any() and t < TMAX - p["t0"]:
        idx = np.where(alive)[0]
        x[idx] += v[idx] * dt + np.sqrt(dt) * rng.standard_normal(len(idx))
        t += dt
        hit_u = alive & (x >= p["a"]); hit_l = alive & (x <= 0)
        rt[hit_u | hit_l] = t + p["t0"]; up[hit_u] = True; alive &= ~(hit_u | hit_l)
    keep = ~np.isnan(rt)
    return {"rt": rt[keep], "up": up[keep], "sc": sc[keep], "pos": pos[keep], "lev": lev[keep]}


def recover_session(args):
    sid, d, row, scen, seed = args
    rng = np.random.default_rng(seed)
    d = d[(d.reaction_time / 1000 < TMAX)]
    D = session_data(d)
    p = {k: row[f"zv_{k}"] for k in ("a", "t0", "z0", "v0")}
    for k in ("v13", "v35", "v100"):
        p[k] = row[f"zv_{k}"] if f"zv_{k}" in row else None
    # take drift levels from the full 'zv' refit stored in row; dz/dv set by scenario
    p["dz"], p["dv"] = {"none": (0, 0), "z": (row["true_dz"], 0), "v": (0, row["true_dv"]), "zv": (row["true_dz"], row["true_dv"])}[scen]
    S = simulate(p, D, rng)
    res = {"session_id": sid, "scenario": scen, "seed": seed, "true_dz": p["dz"], "true_dv": p["dv"]}
    for mname, extra in PARAMS.items():
        r = fit(BASE + extra, S, starts=2, seed=seed)
        res[f"ll_{mname}"] = -r.fun; res[f"k_{mname}"] = len(BASE + extra)
        q = unpack(r.x, BASE + extra); res[f"{mname}_dz"], res[f"{mname}_dv"] = q["dz"], q["dv"]
    return res


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "fit"
    t = trials()
    t = t[~t.session_id.isin(TWO_PRIOR)]
    groups = list(t.groupby("session_id"))
    if mode == "fit":
        with Pool(24) as pool:
            res = pool.map(run_session, groups)
        catalog.save_table(pd.DataFrame(res), "audit/a13_ddm_compare.csv", index=False)
    else:
        fitted = pd.read_csv("outputs/audit/a13_ddm_compare.csv").set_index("session_id")
        # refit zv on each session to get drift levels for the generator
        jobs = []
        rng = np.random.default_rng(7)
        for sid, d in groups:
            D = session_data(d[(d.reaction_time / 1000 < TMAX)])
            r = fit(BASE + ["dz", "dv"], D, starts=2)
            row = {f"zv_{k}": v for k, v in unpack(r.x, BASE + ["dz", "dv"]).items()}
            # generating effect sizes: the population-typical prior effect of each kind (median |estimate| across sessions)
            row["true_dz"], row["true_dv"] = 0.04, 0.25
            for scen in ("none", "z", "v", "zv"):
                jobs.append((sid, d, row, scen, int(rng.integers(1e9))))
        with Pool(24) as pool:
            res = pool.map(recover_session, jobs)
        catalog.save_table(pd.DataFrame(res), "audit/a13_ddm_recovery.csv", index=False)
