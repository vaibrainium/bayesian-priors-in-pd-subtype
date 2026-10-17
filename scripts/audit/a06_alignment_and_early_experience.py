"""(1) Does behaviour track the colour recorded on the same trial (vs neighbouring trials)?
(2) Natural experiment: does the contingency a participant happened to experience early in a session predict that session's prior use?"""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

from scripts.audit.common import logit, trials
from src.data_io import catalog

ta = trials(kind="all")  # full sequence incl. invalid trials, for experienced contingencies
tv = trials().dropna(subset=["prev_rc"])
s = pd.read_csv("outputs/audit/a04_session_design.csv").set_index("session_id")

# (1) colour alignment: prior-use coefficient using colour at lag -1, 0, +1 (full sequence order)
ta = ta.sort_values(["session_filename"], kind="stable")
ta["col_prev"] = ta.groupby("session_filename")["col"].shift(1)
ta["col_next"] = ta.groupby("session_filename")["col"].shift(-1)
va = ta[ta.is_valid.astype(bool)].dropna(subset=["col_prev", "col_next", "choice"])
va = va[va.coherence == 0]
rows = []
for sid, d in va.groupby("session_id"):
    rec = {"session_id": sid}
    for lab in ["col", "col_prev", "col_next"]:
        X = pd.DataFrame({"const": 1.0, "c": d[lab]})
        p, se = logit(d.choice, X)
        rec[lab] = p["c"]
    rows.append(rec)
al = pd.DataFrame(rows)
print("0% coherence colour coefficient, mean over sessions (lag check):", al[["col", "col_prev", "col_next"]].mean().round(3).to_dict())
print("sessions where a neighbouring-trial colour beats the same-trial colour by >0.5:", ((al[["col_prev", "col_next"]].max(axis=1) - al.col) > 0.5).sum())

# (2) early experienced contingency. For each session: in the first N trials of the prior colour (all coherences, valid or not
# — the participant saw the stimulus, feedback only on valid), fraction whose target was the favoured side; same for the equal colour.
rows = []
for sid, d in ta.groupby("session_id"):
    rec = {"session_id": sid}
    for N in (20, 40, 80):
        pos = d[d.color == 1].head(N)
        eq = d[d.color == 0].head(N)
        rec[f"fav_pos_{N}"] = pos.target.mean()
        rec[f"fav_eq_{N}"] = eq.target.mean()
        rec[f"contrast_{N}"] = pos.target.mean() - eq.target.mean()
    # experienced over the whole session
    rec["contrast_all"] = d[d.color == 1].target.mean() - d[d.color == 0].target.mean()
    # 0%-coherence feedback contingency in the first 40 0% trials of each colour (reward of favoured choice)
    rows.append(rec)
e = pd.DataFrame(rows).set_index("session_id").join(s[["subject_id", "group", "med", "two_prior", "b_col", "se_col"]])
e = e[~e.two_prior]
catalog.save_table(e.reset_index(), "audit/a06_early_experience.csv", index=False)
for N in (20, 40, 80):
    r = stats.spearmanr(e[f"contrast_{N}"], e.b_col)
    print(f"first {N} trials/colour: experienced contrast SD {e[f'contrast_{N}'].std():.3f}; Spearman with session prior use rho={r.statistic:.3f} p={r.pvalue:.3f} n={len(e)}")
r = stats.spearmanr(e.contrast_all, e.b_col); print(f"whole session contrast SD {e.contrast_all.std():.3f}; rho={r.statistic:.3f} p={r.pvalue:.3f}")
# prior use measured only AFTER the first 80 trials/colour vs early contrast (avoid using the same trials)
