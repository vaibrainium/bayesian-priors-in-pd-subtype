"""Trial-level hierarchical logistic models (same priors/sampler settings as src/utils/mcmc_cache.sample_model),
optionally with a person-level colour-direction preference (k*col) and side preference (s_d), and a tremor moderator.

usage: python -m scripts.audit.a10_hier_models <model>   with model in M0, M1, M1_excl2p, M0_tremor, M1_tremor, M1_same20
"""
import json
import sys

import numpy as np
import pandas as pd

from scripts.audit.common import TWO_PRIOR, session_table
from src.data_io import catalog
from src.utils.mcmc_cache import FIT_SETTINGS, ess_bulk, prepare_trials, rhat

name = sys.argv[1]
meta = session_table()
pdm = meta[meta.group == "pd"]
trials = catalog.load_trials(max_equal_prior_bias=None)
subjects = pdm.subject_id.unique()
if "excl2p" in name:
    bad = {sid.split("_")[0] for sid in TWO_PRIOR if "_" in sid}
    subjects = [s for s in subjects if s not in bad]
if ("tremor" in name or "same20" in name) and "all40" not in name:
    sub20 = pdm[pdm.trem_vs_brady_type.isin(["tremor", "bradykinetic"])]
    subjects = sub20.subject_id.unique()
df = prepare_trials(trials, subjects)
kmap = pdm.set_index("session_id").prior_condition
df["k"] = np.where(df.session_id.map(kmap).isin(["gr", "rl"]), 1.0, -1.0)
df["sd"] = np.where(df.session_id.map(kmap).str[1] == "r", 0.5, -0.5)
df["kcol"] = df.k * df.col

terms = ["intercept", "coh", "col", "med", "col:med", "prev_rc", "prev_rc:med"]
cols = [np.ones(len(df)), df.coh, df.col, df.med, df.col * df.med, df.prev_rc, df.prev_rc * df.med]
if name.startswith("M1"):
    terms += ["kcol", "side"]
    cols += [df.kcol, df.sd]
X = np.column_stack(cols).astype("float64")
subj = df.subject_id.unique()
sidx = df.subject_id.map({s: i for i, s in enumerate(subj)}).to_numpy()
y = df.choice.to_numpy().astype("int8")
mod = None
if "tremor" in name:
    typed = pdm[(pdm.med == "off") & pdm.trem_vs_brady_type.isin(["tremor", "bradykinetic"])].set_index("subject_id").tremor_score
    z = ((typed - typed.mean()) / typed.std()).reindex(subj).fillna(0.0)  # all40 variant: unscored patients at the mean
    mod = (z.to_numpy()[sidx] * (df.col * df.med).to_numpy()).astype("float64")

import pymc as pm

with pm.Model():
    beta = pm.Normal("beta", 0, 2.5, shape=X.shape[1])
    sd_u = pm.HalfNormal("sd_u", 1.0, shape=X.shape[1])
    z_u = pm.Normal("z_u", 0, 1, shape=(len(subj), X.shape[1]))
    eta = pm.math.sum(X * (beta + (z_u * sd_u)[sidx]), axis=1)
    if mod is not None:
        gamma = pm.Normal("gamma", 0, 2.5)
        eta = eta + gamma * mod
    pm.Bernoulli("obs", logit_p=eta, observed=y)
    idata = pm.sample(FIT_SETTINGS["draws"], tune=FIT_SETTINGS["tune"], chains=4, cores=4, target_accept=0.9, random_seed=0, progressbar=False)

post = idata.posterior
b = post["beta"].values
sdu = post["sd_u"].values
zu = post["z_u"].values
summ = {"model": name, "n_subjects": len(subj), "n_trials": int(len(df)), "divergences": int(idata.sample_stats["diverging"].values.sum()), "terms": terms}
for i, t in enumerate(terms):
    d = b[:, :, i].ravel()
    summ[t] = {"mean": float(d.mean()), "2.5%": float(np.percentile(d, 2.5)), "97.5%": float(np.percentile(d, 97.5)), "P(>0)": float((d > 0).mean()), "rhat": rhat(b[:, :, i]), "ess": ess_bulk(b[:, :, i])}
    s = sdu[:, :, i].ravel()
    summ["sd_u " + t] = {"mean": float(s.mean()), "2.5%": float(np.percentile(s, 2.5)), "97.5%": float(np.percentile(s, 97.5)), "rhat": rhat(sdu[:, :, i]), "ess": ess_bulk(sdu[:, :, i])}
if mod is not None:
    g = post["gamma"].values
    summ["gamma tremor x col:med"] = {"mean": float(g.mean()), "2.5%": float(np.percentile(g, 2.5)), "97.5%": float(np.percentile(g, 97.5)), "P(>0)": float((g > 0).mean()), "rhat": rhat(g), "ess": ess_bulk(g)}
# all rhat / ess incl. subject-level
all_r = [rhat(b[:, :, i]) for i in range(len(terms))] + [rhat(sdu[:, :, i]) for i in range(len(terms))]
summ["max_rhat_pop"] = float(max(all_r))
# participant-level col:med effects
j = terms.index("col:med")
part = b[:, :, j][..., None] + zu[:, :, :, j] * sdu[:, :, j][..., None]
part = part.reshape(-1, len(subj))
pe = pd.DataFrame({"subject_id": subj, "mean": part.mean(0), "sd": part.std(0), "2.5%": np.percentile(part, 2.5, 0), "97.5%": np.percentile(part, 97.5, 0), "P(>0)": (part > 0).mean(0)})
catalog.save_table(pe, f"audit/fits/{name}_participant_colmed.csv", index=False)
catalog.save_arrays(f"audit/fits/{name}_draws.npz", beta=b, sd_u=sdu, part=part[::4], subjects=np.asarray(subj, dtype=str))
catalog.save_text(json.dumps(summ, indent=2), f"audit/fits/{name}.json")
print(json.dumps({k: v for k, v in summ.items() if k in ("col:med", "sd_u col:med", "kcol", "sd_u kcol", "gamma tremor x col:med", "divergences", "max_rhat_pop")}, indent=1))
