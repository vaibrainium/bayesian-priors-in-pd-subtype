"""Per-session prior-use estimates (log-odds, with SE) and performance measures."""
import numpy as np
import pandas as pd

from scripts.audit.common import logit, session_table, trials
from src.data_io import catalog

t = trials()
meta = session_table()
rows = []
for sid, d in t.groupby("session_id"):
    d = d.dropna(subset=["prev_rc"])
    rec = {"session_id": sid}
    # (a) model-matched: choice ~ 1 + coh + col + prev_rc  (as the hierarchical model, per session)
    X = pd.DataFrame({"const": 1.0, "coh": d.coh, "col": d.col, "prev_rc": d.prev_rc})
    p, se = logit(d.choice, X)
    rec.update(b_col=p["col"], se_col=se["col"], b_coh=p["coh"], b_int=p["const"], b_prevrc=p["prev_rc"])
    # (b) flexible: signed coherence as categorical levels
    lv = pd.get_dummies(d.signed_coherence.astype(int), prefix="c", drop_first=False).astype(float)
    lv = lv.drop(columns=["c_0"])
    X2 = pd.concat([pd.DataFrame({"const": 1.0, "col": d.col, "prev_rc": d.prev_rc}), lv], axis=1)
    p2, se2 = logit(d.choice, X2)
    rec.update(b_col_flex=p2["col"], se_col_flex=se2["col"])
    # (c) 0% coherence only: proportion-of-favoured difference and logit difference
    z = d[d.coherence == 0]
    pp, pe = z[z.color == 1].choice.mean(), z[z.color == 0].choice.mean()
    n1, n0 = (z.color == 1).sum(), (z.color == 0).sum()
    rec.update(p0_pos=pp, p0_eq=pe, n0_pos=n1, n0_eq=n0, d0=pp - pe, se_d0=np.sqrt(pp * (1 - pp) / n1 + pe * (1 - pe) / n0))
    Xz = pd.DataFrame({"const": 1.0, "col": z.col})
    pz, sez = logit(z.choice, Xz)
    rec.update(b_col0=pz["col"], se_col0=sez["col"])
    # performance
    rec.update(
        n_trials=len(d),
        acc100=d[d.coherence == 100].outcome.mean(),
        acc35=d[d.coherence == 35].outcome.mean(),
        acc13=d[d.coherence == 13].outcome.mean(),
        lapse=1 - d[d.coherence == 100].outcome.mean(),
        median_rt=d.reaction_time.median(),
        median_rt0=z.reaction_time.median(),
        p_fav_overall=d.choice.mean(),
        prev_choice_bias=np.nan,
    )
    rows.append(rec)
s = pd.DataFrame(rows)
s = meta[["session_id", "subject_id", "group", "med", "visit", "experiment_site", "prior_condition", "prior_color", "prior_dir", "two_prior",
          "tremor_score", "bradykinesia_score", "trem_vs_brady_type", "UPDRS", "UPDRS_improvement", "age", "MoCA", "off_meds_hrs", "years_since_diagnosis",
          "easy_accuracy_pct"]].merge(s, on="session_id")
catalog.save_table(s, "audit/a02_session_features.csv", index=False)
pd.set_option("display.width", 250)
print(s.groupby(["group", "med"])[["b_col", "se_col", "b_col_flex", "b_col0", "d0", "b_coh", "acc100", "median_rt"]].agg(["mean", "std"]).round(3).T)
