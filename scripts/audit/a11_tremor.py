"""Tremor association: robustness, shrinkage artefact, mapping-flip confound, measurement error, specificity, prediction."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from src.data_io import catalog

s = pd.read_csv("outputs/audit/a09_session_k.csv")
m = catalog.load_metadata(prior_design_tolerance=None)
pdx = s[s.group == "pd"]
w = pdx.pivot(index="subject_id", columns="med", values=["b_col", "se_col", "k", "two_prior", "tremor_score", "bradykinesia_score", "UPDRS", "trem_vs_brady_type", "age", "acc100", "b_coh", "median_rt"])
d = pd.DataFrame({
    "change_2stage": (w[("b_col", "on")] - w[("b_col", "off")]).astype(float),
    "se_change": np.sqrt(w[("se_col", "on")].astype(float) ** 2 + w[("se_col", "off")].astype(float) ** 2),
    "dk": (w[("k", "on")] - w[("k", "off")]).astype(float) / 2,  # -1, 0, +1
    "k_off": w[("k", "off")].astype(float),
    "two_prior": (w[("two_prior", "off")].astype(bool) | w[("two_prior", "on")].astype(bool)),
    "tremor_off": w[("tremor_score", "off")].astype(float),
    "brady_off": w[("bradykinesia_score", "off")].astype(float),
    "updrs_off": w[("UPDRS", "off")].astype(float),
    "subtype": w[("trem_vs_brady_type", "off")],
    "age": w[("age", "off")].astype(float),
    "b_off": w[("b_col", "off")].astype(float),
    "b_on": w[("b_col", "on")].astype(float),
    "dacc": (w[("acc100", "on")] - w[("acc100", "off")]).astype(float),
    "dslope": (w[("b_coh", "on")] - w[("b_coh", "off")]).astype(float),
    "drt": (w[("median_rt", "on")] - w[("median_rt", "off")]).astype(float),
})
rep = pd.read_csv("outputs/participant_medication_effects.csv").set_index("participant_id")
d["reported_effect"] = rep["mean"]
d20 = d[d.subtype.isin(["tremor", "bradykinetic"])].copy()
print("n subtyped:", len(d20), " with two-prior session:", int(d20.two_prior.sum()), list(d20.index[d20.two_prior]))
print("mapping-sign flip (dk != 0):", int((d20.dk != 0).sum()), "of", len(d20))

def sp(x, y):
    r = stats.spearmanr(x, y); return r.statistic, r.pvalue

rows = []
def add(label, x, y, n=None):
    rho, p = sp(x, y)
    rows.append({"analysis": label, "n": len(x), "spearman rho": rho, "p": p})

add("reported effects (subtype-specific fits) vs tremor OFF", d20.tremor_off, d20.reported_effect)
add("two-stage raw change vs tremor OFF", d20.tremor_off, d20.change_2stage)
ok = ~d20.two_prior
add("two-stage change, excluding 5 two-prior patients", d20.tremor_off[ok], d20.change_2stage[ok])
add("reported effects, excluding 5 two-prior patients", d20.tremor_off[ok], d20.reported_effect[ok])
same = d20.dk == 0
add("two-stage change, mapping sign unchanged", d20.tremor_off[same], d20.change_2stage[same])
add("two-stage change, mapping sign flipped", d20.tremor_off[~same], d20.change_2stage[~same])
# colour-direction preference-adjusted change: residualise change on dk * P_hat where P_hat is estimated... use partial on dk*(b_off*k_off)
d20["flip_term"] = d20.dk * d20.k_off * d20.b_off  # projected visit-1 association that flips; crude
add("tremor vs tremor-free? tremor OFF vs mapping flip (dk)", d20.tremor_off, d20.dk)
res = pd.DataFrame(rows)
print(res.round(3).to_string())

# leave-one-out on the two-stage and the reported
for lab, y in [("reported", d20.reported_effect), ("two-stage", d20.change_2stage)]:
    loo = [sp(d20.tremor_off.drop(i), y.drop(i))[0] for i in d20.index]
    print(f"LOO rho ({lab}): min {min(loo):.3f} max {max(loo):.3f}; most influential: {d20.index[int(np.argmin(loo))]}")
# robust regression (Huber) and Theil-Sen on two-stage
x = (d20.tremor_off - d20.tremor_off.mean()) / d20.tremor_off.std()
for lab, y in [("reported", d20.reported_effect), ("two-stage", d20.change_2stage)]:
    rlm = sm.RLM(y, sm.add_constant(x), M=sm.robust.norms.HuberT()).fit()
    ts = stats.theilslopes(y, x)
    ols = sm.OLS(y, sm.add_constant(x)).fit()
    print(f"{lab}: OLS slope/SD {ols.params.iloc[1]:.3f} (p={ols.pvalues.iloc[1]:.3f}); Huber {rlm.params.iloc[1]:.3f} (p={rlm.pvalues.iloc[1]:.3f}); Theil-Sen {ts.slope:.3f} [{ts.low_slope:.3f},{ts.high_slope:.3f}]")
# measurement error: weighted regression with known sampling variance + residual tau (meta-regression, REML via statsmodels-free approach)
from scipy import optimize
def metareg(y, v, Xm):
    def nll(t2):
        W = 1 / (v + t2); XtW = Xm.T * W
        beta = np.linalg.solve(XtW @ Xm, XtW @ y); r = y - Xm @ beta
        return 0.5 * (np.sum(np.log(v + t2)) + np.log(np.linalg.det(XtW @ Xm)) + np.sum(W * r ** 2))
    t2 = optimize.minimize_scalar(nll, bounds=(0, 25), method="bounded").x
    W = 1 / (v + t2); XtW = Xm.T * W
    cov = np.linalg.inv(XtW @ Xm); beta = cov @ (XtW @ y)
    return beta, np.sqrt(np.diag(cov)), np.sqrt(t2)
for lab, Xcols in [("tremor", ["tremor_off"]), ("tremor + brady", ["tremor_off", "brady_off"]),
                   ("tremor + subtype", ["tremor_off", "td"]), ("UPDRS off", ["updrs_off"]), ("brady", ["brady_off"])]:
    dd = d20.assign(td=(d20.subtype == "tremor").astype(float))
    Z = dd[Xcols].astype(float); Z = (Z - Z.mean()) / Z.std()
    Xm = np.column_stack([np.ones(len(dd)), Z.to_numpy()])
    beta, se, tau = metareg(dd.change_2stage.to_numpy(), dd.se_change.to_numpy() ** 2, Xm)
    print(f"meta-regression [{lab}]: " + ", ".join(f"{c} {b:+.3f} (SE {e:.3f}, z={b / e:.2f})" for c, b, e in zip(Xcols, beta[1:], se[1:])) + f"; residual tau {tau:.3f}")
# prediction: LOO-CV of change from single predictors (OLS), compared with intercept-only
def loocv(y, Xdf):
    errs = []
    for i in range(len(y)):
        tr = np.ones(len(y), bool); tr[i] = False
        if Xdf is None:
            pred = y[tr].mean()
        else:
            Xm = sm.add_constant(Xdf.to_numpy()[tr], has_constant="add"); b = np.linalg.lstsq(Xm, y[tr], rcond=None)[0]
            pred = np.r_[1, Xdf.to_numpy()[i]] @ b
        errs.append((y[i] - pred) ** 2)
    return np.mean(errs)
y = d20.change_2stage.to_numpy()
base = loocv(y, None)
for lab, cols in [("tremor OFF", ["tremor_off"]), ("brady OFF", ["brady_off"]), ("UPDRS OFF", ["updrs_off"]), ("subtype", ["td"]), ("age", ["age"])]:
    dd = d20.assign(td=(d20.subtype == "tremor").astype(float))
    mse = loocv(y, dd[cols].astype(float))
    print(f"LOO-CV: {lab:32s} MSE {mse:.3f} vs intercept {base:.3f}  -> R2_cv {1 - mse / base:+.3f}")
catalog.save_table(d.reset_index(), "audit/a11_patient_changes.csv", index=False)
catalog.save_table(res, "audit/a11_tremor_correlations.csv", index=False)
print(d20[["subtype", "tremor_off", "reported_effect", "change_2stage", "se_change", "dk", "two_prior", "b_off", "b_on"]].sort_values("tremor_off").round(2).to_string())
