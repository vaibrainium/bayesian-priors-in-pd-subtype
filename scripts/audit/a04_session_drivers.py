"""What explains session-level prior use? Design factors (colour, direction, visit, carry-over), group, medication."""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.data_io import catalog

s = pd.read_csv("outputs/audit/a02_session_features.csv")
# carry-over index for visit-2 sessions: how the visit-1 mapping should bias the visit-2 measured prior use
pdv = s[s.group == "pd"].copy()
prev = pdv[pdv.visit == 1].set_index("subject_id")[["prior_color", "prior_dir"]]
def carry(r):
    if r.group != "pd" or r.visit != 2 or r.subject_id not in prev.index:
        return 0.0
    pc, pdir = prev.loc[r.subject_id]
    same_col, same_dir = pc == r.prior_color, pdir == r.prior_dir
    if same_col and same_dir:
        return 1.0   # identical mapping: carry-over helps
    if same_col and not same_dir:
        return -1.0  # same colour now cues the opposite side
    if not same_col and same_dir:
        return -1.0  # old prior colour is now the equal colour, biased toward the same side -> smaller difference
    return 1.0       # old prior colour (now equal) biased to the opposite side -> larger difference
s["carry"] = s.apply(carry, axis=1)
s["mapping_relation"] = np.select(
    [s.carry == 0, (s.carry == 1) & s.subject_id.isin(prev.index) & (s.prior_color == s.subject_id.map(prev.prior_color)),
     (s.carry == -1) & (s.prior_color == s.subject_id.map(prev.prior_color)), (s.carry == -1), (s.carry == 1)],
    ["visit 1 / control", "identical", "same colour, flipped side", "new colour, same side", "new colour, flipped side"], "?")
s["visit2"] = (s.visit == 2).astype(float)
s["green"] = (s.prior_color == "g").astype(float)
s["right"] = (s.prior_dir == "r").astype(float)
s["cond"] = np.where(s.group == "hc", "hc", s.med)
s["w"] = 1 / (s.se_col ** 2 + 0.59 ** 2)
catalog.save_table(s, "audit/a04_session_design.csv", index=False)
print(s.groupby("mapping_relation").agg(n=("b_col", "size"), mean_b=("b_col", "mean"), sd=("b_col", "std")).round(3))
print(s.groupby(["cond", "prior_color"]).b_col.agg(["size", "mean"]).round(3))
fits = {}
for name, f, data in [
    ("colour+side", "b_col ~ green + right", s),
    ("cond+colour+side", "b_col ~ C(cond, Treatment('off')) + green + right", s),
    ("pd: med+visit2+carry+colour+side", "b_col ~ C(med) + visit2 + carry + green + right", s[s.group == "pd"]),
    ("pd: med+visit2+carry+colour+side+site", "b_col ~ C(med) + visit2 + carry + green + right + C(experiment_site)", s[s.group == "pd"]),
    ("pd excl two-prior", "b_col ~ C(med) + visit2 + carry + green + right", s[(s.group == "pd") & ~s.two_prior]),
]:
    r = smf.wls(f, data=data.dropna(subset=["visit"] if "visit2" in f else []), weights=data.dropna(subset=["visit"] if "visit2" in f else [])["w"]).fit(cov_type="HC1")
    fits[name] = r
    print("\n==", name, "n =", int(r.nobs), "R2 =", round(r.rsquared, 3))
    print(pd.DataFrame({"coef": r.params, "se": r.bse, "p": r.pvalues}).round(3))
