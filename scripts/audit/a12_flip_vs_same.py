"""Randomised comparison: between-patient SD of the OFF->ON change when the colour-direction sign k is unchanged vs flipped.
Mapping was assigned by the experimenters, so the two groups differ only in whether a stable colour-direction preference P
projects onto the change (change = d_learned + (k_on - k_off) * P)."""
import numpy as np
import pandas as pd
from scipy import stats

from scripts.audit.a03_heterogeneity import q_profile_ci, reml_tau2
from src.data_io import catalog

d = pd.read_csv("outputs/audit/a09_change_by_k.csv").set_index("subject_id")
s = pd.read_csv("outputs/audit/a09_session_k.csv")
tp = s[s.two_prior].subject_id.unique()
rows = []
for lab, sub in [("all 40", d), ("excluding two-prior patients", d.drop(index=[x for x in tp if x in d.index]))]:
    for g, m in [("sign unchanged", sub.dk == 0), ("sign flipped", sub.dk != 0)]:
        y, v = sub[m].change.to_numpy(), sub[m].v.to_numpy()
        t2, mu, se = reml_tau2(y, v)
        lo, hi = q_profile_ci(y, v)
        rows.append({"subset": lab, "group": g, "n": int(m.sum()), "mean change": mu, "SE": se, "tau": np.sqrt(t2), "tau 95% lo": lo, "tau 95% hi": hi})
    # permutation test of tau difference (shuffle group labels)
    rng = np.random.default_rng(1)
    y, v, g = sub.change.to_numpy(), sub.v.to_numpy(), (sub.dk != 0).to_numpy()
    def stat(gg):
        return np.sqrt(reml_tau2(y[gg], v[gg])[0]) - np.sqrt(reml_tau2(y[~gg], v[~gg])[0])
    obs = stat(g)
    perm = np.array([stat(rng.permutation(g)) for _ in range(5000)])
    rows.append({"subset": lab, "group": "difference flipped - unchanged (perm. p)", "n": len(sub), "tau": obs, "SE": np.mean(perm >= obs)})
    # implied SDs under change = d_learned + 2P for flipped
    t_same = rows[-3]["tau"]; t_flip = rows[-2]["tau"]
    rows.append({"subset": lab, "group": "implied SD of colour-direction preference P = sqrt((tau_flip^2 - tau_same^2)/4)", "tau": np.sqrt(max(t_flip ** 2 - t_same ** 2, 0) / 4)})
r = pd.DataFrame(rows)
catalog.save_table(r, "audit/a12_flip_vs_same.csv", index=False)
pd.set_option("display.width", 250)
print(r.round(3).to_string())
# Is flipping balanced across sites and subtypes?
meta = s[s.group == "pd"].drop_duplicates("subject_id").set_index("subject_id")
x = d.join(meta[["experiment_site", "trem_vs_brady_type"]])
print(pd.crosstab(x.experiment_site, x.dk != 0)); print(pd.crosstab(x.trem_vs_brady_type.fillna("not subtyped"), x.dk != 0))
