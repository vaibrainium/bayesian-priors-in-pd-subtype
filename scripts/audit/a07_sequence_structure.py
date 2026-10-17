"""Colour-lag check (done properly: undefined colours are missing) and whether the colour effect survives trial-history controls."""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from scripts.audit.common import TWO_PRIOR, trials
from src.data_io import catalog

ta = trials(kind="all")
ta["colx"] = np.where(ta.color.isna(), np.nan, np.where(ta.color == 1, 0.5, -0.5))
g = ta.groupby("session_filename")
ta["col_m1"], ta["col_p1"] = g["colx"].shift(1), g["colx"].shift(-1)
ta["tgt_m1"] = g["target"].shift(1)
d = ta.dropna(subset=["colx", "col_m1", "col_p1"])
print("corr colour_t,colour_t+1", round(np.corrcoef(d.colx, d.col_p1)[0, 1], 3))
dd = d.dropna(subset=["target", "tgt_m1"])
print("corr target_t, colour_t+1", round(np.corrcoef(dd.target, dd.col_p1)[0, 1], 3), "corr target_t, target_t-1", round(np.corrcoef(dd.target, dd.tgt_m1)[0, 1], 3))

# Trial-history model on valid trials. History variables defined on the previous *valid* trial (what the hierarchical model uses)
tv = trials().copy()
g = tv.groupby("session_filename")
tv["prev_col"] = g["col"].shift(1)
tv["prev_choice_x_col"] = tv["prev_choice"] * tv["prev_col"]
tv["next_col"] = g["col"].shift(-1)
# colour-specific previous outcome: last valid trial of the SAME colour, rewarded choice (signed toward favoured side)
tv["rc_same_col"] = tv.groupby(["session_filename", "color"])["rc"].shift(1)
tv["choice_same_col"] = tv.groupby(["session_filename", "color"])["signed_choice"].shift(1)
z = tv[(tv.coherence == 0) & ~tv.session_id.isin(TWO_PRIOR)].dropna(subset=["prev_col", "next_col", "prev_rc", "rc_same_col"])
z = z.assign(sess=z.session_id)
specs = {
    "colour only": "choice ~ col",
    "colour + prev/next colour": "choice ~ col + prev_col + next_col",
    "colour + previous choice/target": "choice ~ col + prev_choice + prev_target",
    "colour + same-colour history": "choice ~ col + prev_choice + prev_target + rc_same_col + choice_same_col + prev_col + prev_choice_x_col",
}
rows = []
for name, f in specs.items():
    r = smf.logit(f + " + C(sess)", data=z).fit(disp=0, maxiter=200)  # session fixed effects absorb session intercepts
    for term in [t for t in r.params.index if not t.startswith("C(sess)") and t != "Intercept"]:
        rows.append({"model": name, "term": term, "coef": r.params[term], "se": r.bse[term], "p": r.pvalues[term]})
    rows.append({"model": name, "term": "AIC", "coef": r.aic})
res = pd.DataFrame(rows)
catalog.save_table(res, "audit/a07_history_models.csv", index=False)
pd.set_option("display.width", 200)
print(res.round(4).to_string())
