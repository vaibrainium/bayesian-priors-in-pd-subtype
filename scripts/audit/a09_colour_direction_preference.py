"""Hypothesis: each person has a stable colour->direction preference P (bias toward 'right' for green relative to red).
In a session with mapping (colour c, side d), measured prior use b_col = learned + k*P, with k = s_c*s_d
(gr, rl: k=+1; gl, rr: k=-1). If P is stable, k*b_col should correlate across visits even though b_col does not."""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from scripts.audit.a03_heterogeneity import reml_tau2
from src.data_io import catalog

s = pd.read_csv("outputs/audit/a04_session_design.csv")
s["k"] = np.where(s.prior_condition.isin(["gr", "rl"]), 1, -1)
s["P_hat"] = s.b_col * s.k
pdx = s[s.group == "pd"]
w = pdx.pivot(index="subject_id", columns="med", values=["b_col", "P_hat", "k", "se_col", "two_prior", "experiment_site"])
same_k = (w[("k", "off")] == w[("k", "on")])
print("patients with same k across visits:", int(same_k.sum()), "different k:", int((~same_k).sum()))
for lab, m in [("all", np.ones(len(w), bool)), ("same k", same_k.to_numpy()), ("different k", (~same_k).to_numpy())]:
    ww = w[m]
    r1 = stats.pearsonr(ww[("b_col", "off")].astype(float), ww[("b_col", "on")].astype(float))
    r2 = stats.pearsonr(ww[("P_hat", "off")].astype(float), ww[("P_hat", "on")].astype(float))
    print(f"{lab:12s} n={len(ww):2d}  r(b_col OFF,ON)={r1.statistic:+.2f} (p={r1.pvalue:.3f})   r(k*b_col OFF,ON)={r2.statistic:+.2f} (p={r2.pvalue:.3f})")
excl = ~(w[("two_prior", "off")].astype(bool) | w[("two_prior", "on")].astype(bool))
ww = w[excl]
r2 = stats.pearsonr(ww[("P_hat", "off")].astype(float), ww[("P_hat", "on")].astype(float))
print(f"excluding two-prior patients n={len(ww)}: r(k*b_col)={r2.statistic:+.2f} p={r2.pvalue:.4f}")
rs = stats.spearmanr(w[("P_hat", "off")].astype(float), w[("P_hat", "on")].astype(float))
print(f"Spearman r(k*b_col) all 40: {rs.statistic:+.2f} p={rs.pvalue:.4f}")
# disattenuated
for col in ["b_col", "P_hat"]:
    a, b = w[(col, "off")].astype(float), w[(col, "on")].astype(float)
    va, vb = (w[("se_col", "off")].astype(float) ** 2).mean(), (w[("se_col", "on")].astype(float) ** 2).mean()
    print(col, "disattenuated r =", round(np.cov(a, b)[0, 1] / np.sqrt((a.var() - va) * (b.var() - vb)), 3))

# Variance decomposition in a session-level model: b_col_ij = mu_cond + k_ij * P_i + e_ij  (random P_i per patient)
long = pdx.copy()
long["kc"] = long.k.astype(float)
md = smf.mixedlm("b_col ~ C(med)", long, groups=long.subject_id, re_formula="0 + kc").fit(reml=True)
print(md.summary().tables[1])
print("SD of person colour-direction preference P:", round(np.sqrt(md.cov_re.iloc[0, 0]), 3), " residual session SD (incl. sampling):", round(np.sqrt(md.scale), 3))
md0 = smf.mixedlm("b_col ~ C(med)", long, groups=long.subject_id, re_formula="1").fit(reml=True)
print("comparison: random intercept (stable prior-use trait) SD:", round(np.sqrt(md0.cov_re.iloc[0, 0]), 3), "residual SD", round(np.sqrt(md0.scale), 3))
print("ML log-lik: pref model", round(smf.mixedlm("b_col ~ C(med)", long, groups=long.subject_id, re_formula="0 + kc").fit(reml=False).llf, 2),
      " trait model", round(smf.mixedlm("b_col ~ C(med)", long, groups=long.subject_id, re_formula="1").fit(reml=False).llf, 2))

# Medication effect after removing the projected preference: the projection differs between visits only when k changes
# change = learned_on - learned_off + (k_on - k_off) * P.  For patients with different k, (k_on-k_off)=+-2.
d = pd.DataFrame({"change": (w[("b_col", "on")] - w[("b_col", "off")]).astype(float), "dk": (w[("k", "on")] - w[("k", "off")]).astype(float),
                  "v": (w[("se_col", "on")] ** 2 + w[("se_col", "off")] ** 2).astype(float)})
for lab, m in [("same k", d.dk == 0), ("different k", d.dk != 0)]:
    t2, mu, se = reml_tau2(d[m].change.to_numpy(), d[m].v.to_numpy())
    print(f"{lab}: n={m.sum()}, mean change {mu:+.3f} (SE {se:.3f}), between-patient SD of change tau={np.sqrt(t2):.3f}")
catalog.save_table(s, "audit/a09_session_k.csv", index=False)
catalog.save_table(d.reset_index(), "audit/a09_change_by_k.csv", index=False)

# eq sessions: pre-prior pseudo-bias vs same-day biased prior use
eq = pd.read_csv("outputs/audit/a08_eq_sessions.csv")
r = stats.pearsonr(eq.pseudo_b_sameday, eq.biased_b_col)
print(f"UCLA no-prior block (run before the prior) vs same-day prior use: r={r.statistic:.2f} p={r.pvalue:.4f} n={len(eq)}")
