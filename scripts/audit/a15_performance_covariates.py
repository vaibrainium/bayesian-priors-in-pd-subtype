"""Does session-level prior use track sensitivity, lapses or speed? Does medication change them?"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from src.data_io import catalog

s = pd.read_csv("outputs/audit/a09_session_k.csv")
s = s[~s.two_prior].copy()
s["log_rt"] = np.log(s.median_rt)
s["cond"] = np.where(s.group == "hc", "hc", s.med)
print("session-level Spearman with prior use (all 92 sessions):")
for v in ["b_coh", "acc100", "acc35", "acc13", "log_rt", "b_prevrc"]:
    r = stats.spearmanr(s[v], s.b_col); print(f"  {v:8s} rho={r.statistic:+.3f} p={r.pvalue:.3f}")
pdx = s[s.group == "pd"]
w = pdx.pivot(index="subject_id", columns="med", values=["b_col", "b_coh", "acc100", "acc13", "log_rt", "b_prevrc"]).dropna()
print("\nmedication effects on performance (paired, n=%d):" % len(w))
rows = []
for v in ["b_col", "b_coh", "acc100", "acc13", "log_rt", "b_prevrc"]:
    d = w[(v, "on")] - w[(v, "off")]
    r = stats.wilcoxon(d)
    rows.append({"measure": v, "mean OFF": w[(v, "off")].mean(), "mean ON": w[(v, "on")].mean(), "mean change": d.mean(), "Wilcoxon p": r.pvalue})
perf = pd.DataFrame(rows); print(perf.round(3).to_string())
print("\nwithin-patient: change in prior use vs change in performance (Spearman):")
dd = pd.DataFrame({v: w[(v, "on")] - w[(v, "off")] for v in ["b_col", "b_coh", "acc100", "acc13", "log_rt", "b_prevrc"]})
for v in ["b_coh", "acc100", "acc13", "log_rt", "b_prevrc"]:
    r = stats.spearmanr(dd[v], dd.b_col); print(f"  d{v:8s} rho={r.statistic:+.3f} p={r.pvalue:.3f}")
catalog.save_table(perf, "audit/a15_medication_performance.csv", index=False)
# prior effect at nonzero coherence: does the colour effect generalise beyond 0%? colour x |coherence| in pooled logistic
from scripts.audit.common import TWO_PRIOR, trials
t = trials(); t = t[~t.session_id.isin(TWO_PRIOR)]
t["cond"] = np.where(t.group == "hc", "hc", t.medication)
print("\ncolour effect by coherence level (session fixed effects, signed-coherence dummies):")
for cond in ["hc", "off", "on"]:
    out = []
    for c in [0, 13, 35]:
        z = t[(t.cond == cond) & (t.coherence == c)]
        f = "choice ~ col + C(session_id)" + (" + np.sign(signed_coherence)" if c > 0 else "")
        r = smf.logit(f, z).fit(disp=0)
        out.append(f"{c}%: {r.params['col']:+.2f} ({r.bse['col']:.2f})")
    print(f"  {cond}: " + "; ".join(out))
