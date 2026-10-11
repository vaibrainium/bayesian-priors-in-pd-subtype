"""DDM parameter recovery with the reported fitting pipeline (review comment 4).

For a stratified sample of sessions, simulate a synthetic dataset from each session-colour's fitted
parameters (final model, ``leak-1_tc-1``) on that dataset's own coherence sequence, then refit it with
``ddm_fitting.fit_model``: identical parameterisation, bounds, likelihood and optimiser settings.

Two arms per sampled session:

* ``fitted`` -- colour 0 and colour 1 each simulated from their own fitted parameters, so the true
  colour difference (unequal minus equal) is the one the real fits report.
* ``null`` -- colour 1's coherence sequence simulated from colour 0's parameters, so every true colour
  difference is zero. Paired with the ``fitted`` colour-0 refit it gives the spread of recovered
  colour differences when nothing differs.

Simulation noise is drawn independently of the fitting's common random numbers (seed 42), so the
refit does not see the noise that generated the data.

    python3 scripts/ddm/ddm_recovery.py --make-jobs --n-sessions 20
    NUMBA_NUM_THREADS=4 python3 scripts/ddm/ddm_recovery.py --worker   # start as many as cores allow

Workers claim jobs with an exclusive lock file, so more can be started at any time. Results go to
``outputs/ddm_recovery``; the data directory is only read.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from config import dir_config, main_config
from src.ddm.utils import build_stimulus, load_model, prepare_data

OUT_DIR = ROOT / "outputs" / "ddm_recovery"
SHARDS = OUT_DIR / "shards"
JOBS = OUT_DIR / "jobs.json"
GROUPS = ["HC", "tremor_off", "tremor_on", "bradykinetic_off", "bradykinetic_on"]


def make_jobs(n_sessions, seed):
    processed_dir = Path(dir_config.data.processed)
    metadata = pd.read_csv(processed_dir / "processed_metadata_all_data_accu_60.csv")
    md = metadata.dropna(subset=["session_id"]).drop_duplicates("session_id")
    md["group"] = np.where(md["is_pd"] == 0, "HC", md["trem_vs_brady_type"].astype(str) + "_" + md["treatment"].astype(str).str.lower())
    md = md[md["group"].isin(GROUPS)]

    rng = np.random.default_rng(seed)
    per_group = n_sessions // len(GROUPS)
    chosen = []
    for group in GROUPS:
        ids = np.sort(md.loc[md["group"] == group, "session_id"].to_numpy())
        chosen += [(s, group) for s in rng.choice(ids, size=min(per_group, len(ids)), replace=False)]

    jobs = []
    for session_id, group in chosen:
        for arm, data_colour, param_colour in [("fitted", 0, 0), ("fitted", 1, 1), ("null", 1, 0)]:
            jobs.append({"job": len(jobs), "session_id": session_id, "group": group, "arm": arm, "data_colour": data_colour, "param_colour": param_colour})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    JOBS.write_text(json.dumps({"seed": seed, "model": main_config.DDM_MODELING.final_model, "jobs": jobs}, indent=1))
    print(f"{len(jobs)} jobs over {len(chosen)} sessions -> {JOBS}")


def claim(job_id):
    try:
        fd = os.open(SHARDS / f"{job_id}.claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"{os.getpid()}\n".encode())
    os.close(fd)
    return True


def run_job(job, model_name):
    import logging

    logging.disable(logging.INFO)
    from ddm_fitting import fit_model, make_params, DDMModel, LIKELIHOOD_PARAMS

    processed_dir = Path(dir_config.data.processed)
    behavior = pd.read_csv(processed_dir / "ddm" / "behavior_data.csv")
    fitted = load_model(processed_dir / "ddm" / model_name / f"{job['session_id']}_color_{job['param_colour']}.pkl")
    true = {k: float(v) for k, v in fitted["results"]["parameters"].items()}

    # same design as the real dataset: its coherence sequence and its stimulus length
    real = prepare_data(behavior, job["session_id"], job["data_colour"])
    stimulus = build_stimulus(real)
    free, fixed = make_params(enable_leak=True, enable_time_constant=True)
    simulator = DDMModel(fixed_params=fixed, free_params=free, likelihood_params=LIKELIHOOD_PARAMS)
    sim = simulator.simulate(stimulus, true, n_reps=1, seed=10_000 + job["job"])
    sim = pd.DataFrame({"rt": np.asarray(sim["rt"], float), "choice": np.asarray(sim["choice"], float), "signed_coherence": real["signed_coherence"].to_numpy()})
    n_sim = len(sim)
    sim = sim[np.isfinite(sim["rt"]) & np.isfinite(sim["choice"])].reset_index(drop=True)
    sim["choice"] = sim["choice"].astype(int)

    start = time.time()
    _, result = fit_model(sim, build_stimulus(sim), enable_leak=True, enable_time_constant=True)
    recovered = {k: float(v) for k, v in result["parameters"].items()}
    return {
        **job,
        "true": true,
        "recovered": recovered,
        "nll": float(result["likelihood"]),
        "success": bool(result["success"]),
        "n_trials": n_sim,
        "n_valid": len(sim),
        "sim_p_choice1": float(sim["choice"].mean()),
        "real_p_choice1": float(real["choice"].mean()),
        "sim_median_rt": float(sim["rt"].median()),
        "real_median_rt": float(real["rt"].median()),
        "fit_seconds": time.time() - start,
    }


def worker():
    spec = json.loads(JOBS.read_text())
    for job in spec["jobs"]:
        done = SHARDS / f"{job['job']}.json"
        if done.exists() or not claim(job["job"]):
            continue
        print(f"[{os.getpid()}] job {job['job']}: {job['session_id']} {job['arm']} colour {job['data_colour']}", flush=True)
        result = run_job(job, spec["model"])
        tmp = done.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=1))
        tmp.rename(done)
        print(f"[{os.getpid()}] job {job['job']} done in {result['fit_seconds'] / 60:.1f} min", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--make-jobs", action="store_true")
    parser.add_argument("--n-sessions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    processed_dir = Path(dir_config.data.processed)
    assert not str(OUT_DIR.resolve()).startswith(str(processed_dir.parents[0])), "never write to the data dir"
    SHARDS.mkdir(parents=True, exist_ok=True)
    if args.make_jobs:
        make_jobs(args.n_sessions, args.seed)
    if args.worker:
        worker()
