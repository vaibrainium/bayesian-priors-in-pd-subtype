"""Tremor specificity: rest vs action items, overall severity, OFF performance; uses unshrunk two-stage changes."""
import numpy as np
import pandas as pd
from scipy import stats

from src.data_io import catalog

m = catalog.load_metadata(prior_design_tolerance=None)
d = pd.read_csv("outputs/audit/a11_patient_changes.csv").set_index("subject_id")
d20 = d[d.subtype.isin(["tremor", "bradykinetic"])].copy()
off = m[m.treatment == "OFF"].set_index("subject_id")
rest = [c for c in off.columns if c.startswith("317") or c.startswith("318")]
action = [c for c in off.columns if c.startswith("315") or c.startswith("316")]
d20["rest_tremor"] = off.loc[d20.index, rest].astype(float).sum(axis=1)
d20["action_tremor"] = off.loc[d20.index, action].astype(float).sum(axis=1)
s = pd.read_csv("outputs/audit/a02_session_features.csv")
so = s[s.med == "off"].set_index("subject_id")
for v in ["acc100", "b_coh", "median_rt"]:
    d20[f"{v}_off"] = so.loc[d20.index, v]
rows = []
for v in ["tremor_off", "rest_tremor", "action_tremor", "brady_off", "updrs_off", "age", "acc100_off", "b_coh_off", "median_rt_off", "b_off"]:
    x = d20[v].astype(float)
    ok = x.notna()
    r = stats.spearmanr(x[ok], d20.change_2stage[ok])
    rows.append({"predictor (OFF)": v, "n": int(ok.sum()), "rho with two-stage change": r.statistic, "p": r.pvalue,
                 "rho with tremor": stats.spearmanr(x[ok], d20.tremor_off[ok]).statistic})
res = pd.DataFrame(rows)
catalog.save_table(res, "audit/a19_tremor_specificity.csv", index=False)
print(res.round(3).to_string())
# partial Spearman of tremor controlling for each covariate
def partial(x, y, z):
    rx = stats.rankdata(x); ry = stats.rankdata(y); rz = stats.rankdata(z)
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz); ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return stats.pearsonr(ex, ey)
for z in ["brady_off", "updrs_off", "age", "acc100_off", "b_off"]:
    ok = d20[z].notna()
    r = partial(d20.tremor_off[ok], d20.change_2stage[ok], d20[z][ok].astype(float))
    print(f"partial rho(tremor, change | {z}) = {r.statistic:+.3f} p={r.pvalue:.3f} n={ok.sum()}")
r = partial(d20.action_tremor, d20.change_2stage, d20.rest_tremor); print(f"partial rho(action, change | rest) = {r.statistic:+.3f} p={r.pvalue:.3f}")
