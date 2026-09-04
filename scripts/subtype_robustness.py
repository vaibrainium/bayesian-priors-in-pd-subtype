"""Stop-ship checks on the motor-subtype claim in the main manuscript.

Run:  .venv/bin/python scripts/subtype_robustness.py

  1. classification_audit -- do the assigned subtypes obey config/main.yaml's own rule?
  2. scheme_robustness    -- does the colour x medication dissociation survive alternative
                             subtyping schemes, and is there a continuous correlate?

Findings as of 4 Sep 2026: three patients violate the rule (P11 has zero tremor and is
labelled tremor-dominant), and the dissociation roughly halves under every alternative
scheme with no continuous correlate (rho = +0.19, p = 0.40).
"""
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

warnings.filterwarnings("ignore")
PROC = "/mnt/pd-data/processed"

# config/main.yaml -> MDS_UPDRS.TREM_VS_BRADY
BRADY_MAX, TREMOR_MIN = 0.8, 1.0


def load():
    d = pd.read_csv(f"{PROC}/processed_all_data_accu_60_filtered.csv")
    m = pd.read_csv(f"{PROC}/processed_metadata_all_data_accu_60.csv")
    d["coh"] = d.signed_coherence / 100.0
    d["col_"] = np.where(d.color == 1, 0.5, -0.5)
    d["med"] = np.where(d.medication == "on", 0.5, -0.5)
    return d, m


def classification_audit(m):
    u = m[m.is_pd == 1].drop_duplicates("subject_id").set_index("subject_id")
    q = u[["tremor_score", "bradykinesia_score", "trem_by_brady",
           "trem_vs_brady_type"]].dropna(subset=["trem_by_brady"]).copy()
    q["recomputed"] = q.tremor_score / q.bradykinesia_score

    def rule(r):
        if r < BRADY_MAX:
            return "bradykinetic"
        return "intermediate" if r < TREMOR_MIN else "tremor"

    q["expected"] = q.trem_by_brady.map(rule)
    bad = q[q.expected != q.trem_vs_brady_type]
    print("=" * 84 + "\n1. CLASSIFICATION AUDIT vs config/main.yaml TREM_VS_BRADY\n" + "=" * 84)
    consistent = np.allclose(q.trem_by_brady, q.recomputed, equal_nan=True)
    print(f"ratio column == tremor_score/bradykinesia_score : {consistent}")
    print(f"patients violating the rule: {len(bad)} / {len(q)}")
    if len(bad):
        print(bad[["tremor_score", "bradykinesia_score", "trem_by_brady",
                   "trem_vs_brady_type", "expected"]].round(3).to_string())
        print("\n^ the ratio is right, so the bug is in the LABELLING step downstream.")
    return q


def _interaction(sub, lab):
    sub = sub.copy()
    sub["subj"] = sub.subject_id
    if sub.subject_id.nunique() < 5:
        print(f"  {lab:24s} n={sub.subject_id.nunique():2d}  too small")
        return
    r = BinomialBayesMixedGLM.from_formula(
        "choice ~ coh + col_*med",
        {"s_int": "0+C(subj)", "s_col": "0+C(subj):col_", "s_med": "0+C(subj):med",
         "s_cm": "0+C(subj):col_:med"}, sub).fit_vb(verbose=False)
    fe = pd.DataFrame({"est": r.fe_mean, "sd": r.fe_sd}, index=r.model.exog_names)
    e, sd = fe.loc["col_:med", "est"], fe.loc["col_:med", "sd"]
    z = e / sd
    print(f"  {lab:24s} n={sub.subject_id.nunique():2d} trials={len(sub):6d}  "
          f"col x med = {e:+.4f} +/- {sd:.4f}  z={z:+.2f}  p={2 * (1 - norm.cdf(abs(z))):.3g}")


def scheme_robustness(d, m):
    u = m.drop_duplicates("subject_id").set_index("subject_id")
    print("\n" + "=" * 84 + "\n2. ROBUSTNESS TO SUBTYPING SCHEME\n" + "=" * 84)
    for name, col in [("A. tremor vs brady (REPORTED)", "trem_vs_brady_type"),
                      ("B. tremor vs PIGD", "pigd_vs_tremor_type"),
                      ("C. tremor vs non-tremor", "trem_vs_non-trem"),
                      ("D. PIGD vs non-PIGD", "pigd_vs_non-pigd")]:
        print(f"\n{name}  [{col}]")
        dd = d.copy()
        dd["st"] = dd.subject_id.map(u[col].to_dict())
        for lev in dd.st.dropna().unique():
            _interaction(dd[dd.st == lev], str(lev))

    print("\n" + "-" * 84 + "\nCONTINUOUS correlate (no threshold chosen)\n" + "-" * 84)
    p = m[m.is_pd == 1].copy()
    p["pe"] = p.positive_bias - p.equal_bias
    piv = p.pivot_table(index="subject_id", columns="treatment", values="pe").dropna()
    piv["delta"] = piv.ON - piv.OFF
    piv["ratio"] = piv.index.map(u["trem_by_brady"].to_dict())
    q = piv.dropna(subset=["ratio"])
    rs, rp = stats.spearmanr(q.ratio, q.delta), stats.pearsonr(q.ratio, q.delta)
    print(f"Spearman(tremor/brady ratio, OFF->ON change in prior effect) "
          f"rho={rs.statistic:+.3f} p={rs.pvalue:.4f}  n={len(q)}")
    print(f"Pearson                                                      "
          f"r  ={rp.statistic:+.3f} p={rp.pvalue:.4f}")
    print("\nmost/least tremor-dominant patients (non-monotonicity check):")
    print(q.sort_values("ratio")[["ratio", "OFF", "ON", "delta"]].tail(5).round(3).to_string())


if __name__ == "__main__":
    d, m = load()
    classification_audit(m)
    scheme_robustness(d, m)
