"""Model-free signature of where the prior enters: prior effect (positive minus equal colour, favoured-choice proportion)
as a function of RT quantile, at 0% and 13% coherence. A starting-point bias predicts the largest effect at the fastest RTs,
decaying with RT; a drift-rate bias predicts an effect that persists or grows across RT quantiles."""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from scripts.audit.common import TWO_PRIOR, trials
from src.data_io import catalog

t = trials()
t = t[~t.session_id.isin(TWO_PRIOR)].copy()
t["cond"] = np.where(t.group == "hc", "hc", t.medication)
rows = []
for coh in (0, 13):
    z = t[t.coherence == coh].copy()
    z["rtq"] = z.groupby("session_id").reaction_time.transform(lambda x: pd.qcut(x.rank(method="first"), 4, labels=False))
    g = z.groupby(["cond", "session_id", "rtq", "color"]).choice.mean().unstack("color")
    g["effect"] = g[1] - g[0]
    e = g.reset_index().groupby(["cond", "rtq"]).effect.agg(["mean", "sem", "size"]).reset_index()
    e["coherence"] = coh
    rows.append(e)
    # trend test: logistic choice ~ col * rtq_c with session fixed effects
    z["rtq_c"] = z.rtq - 1.5
    for cond in ["hc", "off", "on"]:
        zz = z[z.cond == cond]
        r = smf.logit("choice ~ col * rtq_c + C(session_id)", data=zz).fit(disp=0)
        print(f"coh {coh:>3} {cond:>3}: prior effect at mean RT {r.params['col']:+.3f} (SE {r.bse['col']:.3f}); change per RT quartile {r.params['col:rtq_c']:+.3f} (SE {r.bse['col:rtq_c']:.3f}, p={r.pvalues['col:rtq_c']:.3f})")
cbf = pd.concat(rows)
catalog.save_table(cbf, "audit/a14_conditional_bias.csv", index=False)
print(cbf.pivot_table(index=["coherence", "rtq"], columns="cond", values="mean").round(3))
