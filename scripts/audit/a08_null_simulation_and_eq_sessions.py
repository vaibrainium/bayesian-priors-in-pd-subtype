"""(A) Parametric null: all patients share one medication effect; trial counts, coherence sets, history regressors and
psychometric parameters are each session's own. Re-estimate tau by the same two-stage REML.
(B) Session-state null: sessions draw prior use independently from N(mu_cond, sigma_s^2) with sigma_s from controls.
(C) UCLA no-prior ('eq') sessions: pseudo prior use with colour coded as in the same-day biased session; carry-over on visit 2."""
import numpy as np
import pandas as pd
from scipy import stats

from scripts.audit.a03_heterogeneity import reml_tau2
from scripts.audit.common import logit, session_table, trials
from src.data_io import catalog
from src.utils.sessions import valid_trials

rng = np.random.default_rng(20261016)
t = trials().dropna(subset=["prev_rc"])
s = pd.read_csv("outputs/audit/a04_session_design.csv").set_index("session_id")
pdp = s[s.group == "pd"].reset_index().pivot(index="subject_id", columns="med", values="b_col")
patient_mean = pdp.mean(axis=1)
delta = (pdp.on - pdp.off).mean()

sess = {sid: d for sid, d in t[t.group == "pd"].groupby("session_id")}
def simulate_tau(colfun):
    est = {}
    for sid, d in sess.items():
        r = s.loc[sid]
        b = colfun(r)
        eta = r.b_int + r.b_coh * d.coh.to_numpy() + b * d.col.to_numpy() + r.b_prevrc * d.prev_rc.to_numpy()
        y = rng.random(len(d)) < 1 / (1 + np.exp(-eta))
        X = pd.DataFrame({"const": 1.0, "coh": d.coh.to_numpy(), "col": d.col.to_numpy(), "prev_rc": d.prev_rc.to_numpy()})
        p, se = logit(y.astype(int), X)
        est[sid] = (r.subject_id, r.med, p["col"], se["col"])
    e = pd.DataFrame(est, index=["subject_id", "med", "b", "se"]).T
    w = e.pivot(index="subject_id", columns="med", values=["b", "se"]).astype(float)
    y = (w[("b", "on")] - w[("b", "off")]).to_numpy()
    v = (w[("se", "on")] ** 2 + w[("se", "off")] ** 2).to_numpy()
    return np.sqrt(reml_tau2(y, v)[0]), np.corrcoef(w[("b", "off")], w[("b", "on")])[0, 1]

N = 200
null_hom = [simulate_tau(lambda r: patient_mean[r.subject_id] + (delta / 2 if r.med == "on" else -delta / 2)) for _ in range(N)]
hc = s[(s.group == "hc") & ~s.two_prior]
sigma_s = np.sqrt(hc.b_col.var() - (hc.se_col ** 2).mean())
mu = {"off": s[(s.med == "off")].b_col.mean(), "on": s[(s.med == "on")].b_col.mean()}
null_state = [simulate_tau(lambda r: mu[r.med] + sigma_s * rng.standard_normal()) for _ in range(N)]
obs_tau = 0.880
res = pd.DataFrame({
    "scenario": ["A: homogeneous medication effect, stable patient prior use", "B: no patient trait, independent session states (SD from controls)"],
    "sigma_session_used": [0.0, sigma_s],
    "tau mean": [np.mean([x[0] for x in null_hom]), np.mean([x[0] for x in null_state])],
    "tau 2.5%": [np.percentile([x[0] for x in null_hom], 2.5), np.percentile([x[0] for x in null_state], 2.5)],
    "tau 97.5%": [np.percentile([x[0] for x in null_hom], 97.5), np.percentile([x[0] for x in null_state], 97.5)],
    "P(tau_sim >= observed)": [np.mean([x[0] >= obs_tau for x in null_hom]), np.mean([x[0] >= obs_tau for x in null_state])],
    "r(OFF,ON) mean": [np.mean([x[1] for x in null_hom]), np.mean([x[1] for x in null_state])],
    "r(OFF,ON) 2.5%": [np.percentile([x[1] for x in null_hom], 2.5), np.percentile([x[1] for x in null_state], 2.5)],
    "r(OFF,ON) 97.5%": [np.percentile([x[1] for x in null_hom], 97.5), np.percentile([x[1] for x in null_state], 97.5)],
    "n sims": [N, N],
})
catalog.save_table(res, "audit/a08_null_simulations.csv", index=False)
pd.set_option("display.width", 250)
print(res.round(3).T)

# (C) eq sessions
c = catalog.load_trials(stage="compiled", min_easy_accuracy=None, prior_design_tolerance=None)
c = valid_trials(c, 0.1) if c.reaction_time.max() < 50 else c[(c.is_valid == True) & (c.reaction_time >= 100)]
meta = session_table()
rows = []
for subj in ["CG", "COH", "DP", "FUR", "LBR", "MAR", "MBY", "SMI"]:
    for med in ["off", "on"]:
        biased = meta[(meta.subject_id == subj) & (meta.med == med)]
        if biased.empty:
            continue
        pc, pdir = biased.prior_color.iat[0], biased.prior_dir.iat[0]
        visit = biased.visit.iat[0]
        other = meta[(meta.subject_id == subj) & (meta.med != med)]
        e = c[(c.subject_id == subj) & (c.medication == med) & (c.prior == "eq")]
        if e.empty:
            continue
        def code(d, col_letter, dir_letter):
            fav = "right" if dir_letter == "r" else "left"
            colname = "green" if col_letter == "g" else "red"
            X = pd.DataFrame({"const": 1.0, "coh": np.where(d.target == fav, 1, -1) * d.coherence / 100, "col": np.where(d.color == colname, 0.5, -0.5)})
            y = (d.choice == fav).astype(int)
            p, se = logit(y, X)
            return p["col"], se["col"]
        b_same, se_same = code(e, pc, pdir)  # coded by the same-day prior (learned only AFTER this block)
        rec = {"subject_id": subj, "med": med, "visit": visit, "n_eq_trials": len(e), "pseudo_b_sameday": b_same, "se_sameday": se_same,
               "biased_b_col": s.loc[biased.session_id.iat[0], "b_col"]}
        if visit == 2 and not other.empty:
            b_prev, se_prev = code(e, other.prior_color.iat[0], other.prior_dir.iat[0])  # coded by the previous visit's prior
            rec.update(pseudo_b_prevvisit=b_prev, se_prevvisit=se_prev, prev_visit_b_col=s.loc[other.session_id.iat[0], "b_col"])
        rows.append(rec)
eq = pd.DataFrame(rows)
catalog.save_table(eq, "audit/a08_eq_sessions.csv", index=False)
print(eq.round(3).to_string())
y, v = eq.pseudo_b_sameday.to_numpy(), eq.se_sameday.to_numpy() ** 2
t2, m_, se_m = reml_tau2(y, v)
print(f"eq-session pseudo prior use (same-day coding): mean {m_:.3f} (SE {se_m:.3f}), tau {np.sqrt(t2):.3f}, mean SE {np.sqrt(v.mean()):.3f}, Q p={stats.chi2.sf(np.sum((y - np.average(y, weights=1 / v)) ** 2 / v), len(y) - 1):.3f}")
pv = eq.pivot(index="subject_id", columns="med", values=["pseudo_b_sameday", "se_sameday"]).dropna()
dy = (pv[("pseudo_b_sameday", "on")] - pv[("pseudo_b_sameday", "off")]).to_numpy()
dv = (pv[("se_sameday", "on")] ** 2 + pv[("se_sameday", "off")] ** 2).to_numpy()
t2d, md, sed = reml_tau2(dy, dv)
print(f"eq-session pseudo OFF->ON change: mean {md:.3f} (SE {sed:.3f}), tau {np.sqrt(t2d):.3f}, n={len(dy)}")
v2 = eq.dropna(subset=["pseudo_b_prevvisit"])
if len(v2):
    t2c, mc, sec = reml_tau2(v2.pseudo_b_prevvisit.to_numpy(), v2.se_prevvisit.to_numpy() ** 2)
    print(f"visit-2 eq block coded by visit-1 prior (carry-over): mean {mc:.3f} (SE {sec:.3f}), n={len(v2)}; corr with visit-1 prior use r={np.corrcoef(v2.pseudo_b_prevvisit, v2.prev_visit_b_col)[0, 1]:.2f}")
