"""Prior use at each coherence level, per session (two-stage), and OFF->ON change per level."""
import numpy as np
import pandas as pd

from scripts.audit.a03_heterogeneity import reml_tau2
from scripts.audit.common import TWO_PRIOR, logit, trials
from src.data_io import catalog

t = trials(); t = t[~t.session_id.isin(TWO_PRIOR)]
rows = []
for sid, d in t.groupby("session_id"):
    for c in (0, 13, 35):
        z = d[d.coherence == c]
        X = pd.DataFrame({"const": 1.0, "col": z.col}); 
        if c > 0:
            X["sgn"] = np.sign(z.signed_coherence)
        p, se = logit(z.choice, X)
        rows.append({"session_id": sid, "subject_id": d.subject_id.iat[0], "group": d.group.iat[0], "med": d.medication.iat[0], "coh": c, "b": p["col"], "se": se["col"]})
e = pd.DataFrame(rows)
e = e[np.isfinite(e.se) & (e.se < 5)]
catalog.save_table(e, "audit/a16_coherence_specific.csv", index=False)
out = []
for c in (0, 13, 35):
    for cond in ["hc", "off", "on"]:
        sub = e[(e.coh == c) & ((e.group == "hc") if cond == "hc" else (e.med == cond))]
        t2, mu, se = reml_tau2(sub.b.to_numpy(), sub.se.to_numpy() ** 2)
        out.append({"coherence": c, "condition": cond, "n": len(sub), "mean prior use": mu, "SE": se, "tau": np.sqrt(t2)})
    w = e[(e.coh == c) & (e.group == "pd")].pivot(index="subject_id", columns="med", values=["b", "se"]).dropna()
    y = (w[("b", "on")] - w[("b", "off")]).to_numpy(); v = (w[("se", "on")] ** 2 + w[("se", "off")] ** 2).to_numpy()
    t2, mu, se = reml_tau2(y, v)
    out.append({"coherence": c, "condition": "ON - OFF", "n": len(y), "mean prior use": mu, "SE": se, "tau": np.sqrt(t2)})
r = pd.DataFrame(out); r["z"] = r["mean prior use"] / r.SE
catalog.save_table(r, "audit/a16_coherence_summary.csv", index=False)
print(r.round(3).to_string())
