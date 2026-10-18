"""Run-level audit of the Harvard PD raw files (raw/harvard_data/pd/<patient>_<med>/ and hc/<control>/, each with {practice,trial}/YYMMDD_HHMMSS.mat).

Each session was run as several blocks (one practice file plus several trial files) that 0.00 concatenates. For every
file: date/time from the name, trial count, per-colour target split (% toward right), the prior implied by the file,
100%-coherence accuracy, and the codes present in each event column. Writes outputs/audit/a22_harvard_runs.csv."""
import glob
import os
import re

import hdf5storage
import numpy as np
import pandas as pd
import scipy.io

from src.data_io import catalog
from src.utils.sessions import decode_events

ROOT = "/mnt/pd-data/raw/harvard_data"


def load(path):
    try:
        return hdf5storage.loadmat(path)
    except Exception:
        return scipy.io.loadmat(path)


rows = []
SESSION_DIRS = sorted(glob.glob(f"{ROOT}/pd/P*_*")) + sorted(glob.glob(f"{ROOT}/hc/HC*"))
for sess_dir in SESSION_DIRS:
    if not os.path.isdir(sess_dir):
        continue
    name = os.path.basename(sess_dir)
    subj, med = name.split("_") if "_" in name else (name, "hc")
    for kind in ("practice", "trial"):
        for f in sorted(glob.glob(f"{sess_dir}/{kind}/*.mat")):
            stamp = re.search(r"(\d{6})_(\d{6})", os.path.basename(f))
            d = load(f)
            ev, tm = np.asarray(d["event"]), np.asarray(d["time"])
            rec = {"subject": subj, "med": med.upper(), "kind": kind, "file": os.path.basename(f),
                   "datetime": pd.to_datetime(stamp.group(1) + stamp.group(2), format="%y%m%d%H%M%S") if stamp else pd.NaT,
                   "n_rows": ev.shape[0], "n_cols": ev.shape[1]}
            if ev.shape[1] == 10:
                t = decode_events(ev.astype(float), tm)
                shown = t.dropna(subset=["color", "target"])
                for col in ("green", "red"):
                    s = shown[shown.color == col]
                    rec[f"n_{col}"] = len(s)
                    rec[f"right_{col}_pct"] = 100 * (s.target == "right").mean() if len(s) else np.nan
                easy = t[(t.coherence == 100) & t.is_valid]
                rec["acc100"] = 100 * easy.outcome.mean() if len(easy) else np.nan
                rec["valid_pct"] = 100 * t.is_valid.mean()
                # condition implied by this file alone
                dev = {c: abs(rec.get(f"right_{c}_pct", np.nan) - 50) for c in ("green", "red")}
                biased = [c for c, v in dev.items() if v > 15]
                rec["implied"] = "none" if not biased else " + ".join(
                    f"{c[0]}{'r' if rec[f'right_{c}_pct'] > 50 else 'l'}" for c in biased)
            for c in range(ev.shape[1]):
                vals = np.unique(ev[:, c])
                rec[f"col{c + 1}"] = ",".join(map(str, vals[:6])) + ("…" if len(vals) > 6 else "")
            rows.append(rec)
runs = pd.DataFrame(rows)
catalog.save_table(runs, "audit/a22_harvard_runs.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300); pd.set_option("display.max_columns", 40)
show = ["subject", "med", "kind", "file", "datetime", "n_rows", "n_green", "right_green_pct", "n_red", "right_red_pct", "implied", "acc100", "valid_pct"]
print(runs[show].round(1).to_string(index=False))

# one row per session, in date order: the condition each trial block implies
blocks = runs[runs.kind == "trial"].sort_values("datetime")
sessions = blocks.groupby(["subject", "med"]).agg(
    first_block=("datetime", "min"), n_blocks=("file", "size"),
    block_conditions=("implied", lambda x: " | ".join(x.fillna("aborted"))),
    two_prior_blocks=("implied", lambda x: int((x == "gr + rl").sum())),
).reset_index().sort_values("first_block")
catalog.save_table(sessions, "audit/a22_harvard_sessions.csv", index=False)
print()
print(sessions.to_string(index=False))
