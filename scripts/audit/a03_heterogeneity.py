"""Heterogeneity of OFF->ON change in prior use: meta-analytic tau, parametric null, variance decomposition."""
import numpy as np
import pandas as pd
from scipy import optimize, stats

from scripts.audit.common import logit, trials
from src.data_io import catalog

s = pd.read_csv("outputs/audit/a02_session_features.csv")
pd_ = s[s.group == "pd"]
w = pd_.pivot(index="subject_id", columns="med", values=["b_col", "se_col", "b_col_flex", "se_col_flex", "b_col0", "se_col0", "two_prior", "visit", "experiment_site"])


def reml_tau2(y, v):
    """REML estimate of between-unit variance for y_i ~ N(mu, v_i + tau2)."""
    def nll(t2):
        W = 1 / (v + t2)
        mu = np.sum(W * y) / W.sum()
        return 0.5 * (np.sum(np.log(v + t2)) + np.log(W.sum()) + np.sum(W * (y - mu) ** 2))
    r = optimize.minimize_scalar(nll, bounds=(0, 25), method="bounded")
    t2 = r.x
    W = 1 / (v + t2)
    mu = np.sum(W * y) / W.sum()
    return t2, mu, np.sqrt(1 / W.sum())


def q_profile_ci(y, v, alpha=0.05):
    k = len(y)
    def Q(t2):
        W = 1 / (v + t2)
        mu = np.sum(W * y) / W.sum()
        return np.sum(W * (y - mu) ** 2)
    lo_target, hi_target = stats.chi2.ppf(1 - alpha / 2, k - 1), stats.chi2.ppf(alpha / 2, k - 1)
    f = lambda t2, tgt: Q(t2) - tgt
    lo = 0 if Q(0) < lo_target else optimize.brentq(f, 0, 50, args=(lo_target,))
    hi = 0 if Q(0) < hi_target else optimize.brentq(f, 0, 50, args=(hi_target,))
    return np.sqrt(lo), np.sqrt(hi)


out = []
for label, b, se in [("model-matched (coh linear)", "b_col", "se_col"), ("flexible coherence", "b_col_flex", "se_col_flex"), ("0% coherence only", "b_col0", "se_col0")]:
    for subset_label, keep in [("all 40", np.ones(len(w), bool)), ("excluding two-prior sessions", ~(w[("two_prior", "off")] | w[("two_prior", "on")]).to_numpy())]:
        ww = w[keep]
        ww = ww[np.isfinite(ww[(se, "on")].astype(float)) & np.isfinite(ww[(se, "off")].astype(float))]
        y = (ww[(b, "on")] - ww[(b, "off")]).to_numpy(float)
        v = (ww[(se, "on")] ** 2 + ww[(se, "off")] ** 2).to_numpy(float)
        Wf = 1 / v
        mu_f = np.sum(Wf * y) / Wf.sum()
        Q = np.sum(Wf * (y - mu_f) ** 2)
        k = len(y)
        t2, mu, mu_se = reml_tau2(y, v)
        lo, hi = q_profile_ci(y, v)
        # per-session (not difference) variance decomposition
        off, on = ww[(b, "off")].to_numpy(float), ww[(b, "on")].to_numpy(float)
        voff, von = (ww[(se, "off")] ** 2).to_numpy(float), (ww[(se, "on")] ** 2).to_numpy(float)
        true_var_off = off.var(ddof=1) - voff.mean()
        true_var_on = on.var(ddof=1) - von.mean()
        cov = np.cov(off, on)[0, 1]
        out.append({
            "estimate": label, "subset": subset_label, "k": k,
            "mean change (REML)": mu, "SE": mu_se,
            "observed SD of changes": y.std(ddof=1), "mean sampling SE of change": np.sqrt(v.mean()),
            "Q": Q, "df": k - 1, "p(Q)": stats.chi2.sf(Q, k - 1), "I2": max(0, (Q - (k - 1)) / Q),
            "tau (REML)": np.sqrt(t2), "tau 95% lo": lo, "tau 95% hi": hi,
            "true SD OFF": np.sqrt(max(true_var_off, 0)), "true SD ON": np.sqrt(max(true_var_on, 0)),
            "raw r(OFF,ON)": np.corrcoef(off, on)[0, 1],
            "disattenuated r(OFF,ON)": cov / np.sqrt(max(true_var_off, 1e-9) * max(true_var_on, 1e-9)),
        })
res = pd.DataFrame(out)
catalog.save_table(res, "audit/a03_heterogeneity_meta.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(res.round(3).T)

# controls: between-subject spread of session prior use (single session each)
hc = s[s.group == "hc"]
for lab, sub in [("HC all", hc), ("HC excl HC13", hc[~hc.two_prior])]:
    print(lab, "n", len(sub), "mean b_col", sub.b_col.mean().round(3), "SD", sub.b_col.std().round(3), "mean SE", sub.se_col.mean().round(3),
          "true SD", np.sqrt(max(sub.b_col.var() - (sub.se_col ** 2).mean(), 0)).round(3))
for med in ["off", "on"]:
    sub = pd_[(pd_.med == med) & ~pd_.two_prior]
    print("PD", med, "n", len(sub), "mean", sub.b_col.mean().round(3), "SD", sub.b_col.std().round(3), "true SD", np.sqrt(max(sub.b_col.var() - (sub.se_col ** 2).mean(), 0)).round(3))
# Levene/Brown-Forsythe: HC vs PD OFF vs PD ON spread
print("Brown-Forsythe HC vs PD-OFF vs PD-ON:", stats.levene(hc.b_col, pd_[pd_.med == "off"].b_col, pd_[pd_.med == "on"].b_col, center="median"))
