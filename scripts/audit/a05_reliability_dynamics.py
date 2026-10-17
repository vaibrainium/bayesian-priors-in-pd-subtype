"""Split-half reliability (odd/even vs first/second half), change-score reliability, and within-session time course of prior use."""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from scripts.audit.common import logit, trials
from src.data_io import catalog

t = trials().dropna(subset=["prev_rc"])
s = pd.read_csv("outputs/audit/a04_session_design.csv").set_index("session_id")


def bcol(d):
    X = pd.DataFrame({"const": 1.0, "coh": d.coh, "col": d.col, "prev_rc": d.prev_rc})
    p, se = logit(d.choice, X)
    return p["col"], se["col"]


rows = []
for sid, d in t.groupby("session_id"):
    n = len(d)
    rec = {"session_id": sid}
    idx = np.arange(n)
    for lab, mask in [("odd", idx % 2 == 1), ("even", idx % 2 == 0), ("h1", idx < n / 2), ("h2", idx >= n / 2)]:
        rec[f"b_{lab}"], rec[f"se_{lab}"] = bcol(d[mask])
    for q in range(4):
        m = (idx >= q * n / 4) & (idx < (q + 1) * n / 4)
        rec[f"b_q{q + 1}"], rec[f"se_q{q + 1}"] = bcol(d[m])
    rows.append(rec)
r = pd.DataFrame(rows).set_index("session_id").join(s[["subject_id", "group", "med", "visit", "two_prior", "b_col", "se_col"]])
catalog.save_table(r.reset_index(), "audit/a05_split_estimates.csv", index=False)

def sb(x):  # Spearman-Brown
    return 2 * x / (1 + x)

out = []
for lab, sub in [("all sessions", r), ("excl two-prior", r[~r.two_prior])]:
    oe = np.corrcoef(sub.b_odd, sub.b_even)[0, 1]
    hh = np.corrcoef(sub.b_h1, sub.b_h2)[0, 1]
    out.append({"subset": lab, "n sessions": len(sub), "r odd/even": oe, "SB odd/even": sb(oe), "r 1st/2nd half": hh, "SB 1st/2nd half": sb(hh)})
# change-score reliability: OFF->ON change from odd trials vs from even trials (and halves)
pdr = r[r.group == "pd"].reset_index().pivot(index="subject_id", columns="med")
for lab, a, b in [("change odd vs even", "b_odd", "b_even"), ("change 1st vs 2nd half", "b_h1", "b_h2")]:
    da = pdr[(a, "on")] - pdr[(a, "off")]
    db = pdr[(b, "on")] - pdr[(b, "off")]
    rr = np.corrcoef(da, db)[0, 1]
    out.append({"subset": lab, "n sessions": len(da), "r odd/even" if "odd" in lab else "r 1st/2nd half": rr})
# across-visit correlation for comparison
out.append({"subset": "across visits (OFF vs ON, full sessions)", "n sessions": len(pdr), "r 1st/2nd half": np.corrcoef(pdr[("b_col", "off")], pdr[("b_col", "on")])[0, 1]})
rel = pd.DataFrame(out)
catalog.save_table(rel, "audit/a05_reliability.csv", index=False)
pd.set_option("display.width", 250)
print(rel.round(3).to_string())

# time course: prior use by quarter, by condition (mean of per-session estimates, inverse-variance not used: simple mean +/- SEM)
long = r.reset_index().melt(id_vars=["session_id", "subject_id", "group", "med", "visit", "two_prior"], value_vars=[f"b_q{q}" for q in range(1, 5)], var_name="quarter", value_name="b")
long["q"] = long.quarter.str[-1].astype(int)
long["cond"] = np.where(long.group == "hc", "hc", long.med)
tc = long[~long.two_prior].groupby(["cond", "q"]).b.agg(["mean", "sem", "size"]).round(3)
print(tc.unstack(0))
catalog.save_table(tc.reset_index(), "audit/a05_time_course.csv", index=False)
# slope of prior use over quarters, by condition; mixed model with random intercept+slope per session
long["qc"] = long.q - 2.5
for cond in ["hc", "off", "on"]:
    sub = long[(long.cond == cond) & ~long.two_prior]
    mm = smf.mixedlm("b ~ qc", sub, groups=sub.session_id, re_formula="~qc").fit()
    print(cond, "slope per quarter", round(mm.params["qc"], 3), "SE", round(mm.bse["qc"], 3), "p", round(mm.pvalues["qc"], 4))
sub = long[(long.cond != "hc") & ~long.two_prior].copy()
mm = smf.mixedlm("b ~ qc * C(cond)", sub, groups=sub.subject_id, re_formula="~1", vc_formula={"sess": "0 + C(session_id)"}).fit()
print(pd.DataFrame({"coef": mm.params, "se": mm.bse, "p": mm.pvalues}).round(3))
# first quarter: does medication difference exist from the start?
w = r[(r.group == "pd") & ~r.two_prior].reset_index()
for q in range(1, 5):
    pv = w.pivot(index="subject_id", columns="med", values=f"b_q{q}").dropna()
    dd = pv.on - pv.off
    print(f"Q{q}: mean ON-OFF {dd.mean():.3f} SEM {dd.sem():.3f} n={len(dd)} Wilcoxon p={stats.wilcoxon(dd).pvalue:.3f}")
