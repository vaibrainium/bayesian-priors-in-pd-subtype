"""Feasibility probes for an RL-model follow-on paper from the Glass-pattern PD dataset.

Every number in dissemination/rl_paper_report.md is produced here.
Run:  .venv/bin/python scripts/rl_paper_probes.py

Three probes:
  1. credit_assignment  -- is reward credited to the ACTION or the COLOUR CONTEXT?
  2. rl_fits            -- context-conditioned RL: learning rate vs value->choice gain
  3. explicitness       -- does awareness of the contingency track prior expression?
"""
import numpy as np
import pandas as pd
import warnings
from scipy import stats
from scipy.stats import norm
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

warnings.filterwarnings("ignore")
PROC = "/mnt/pd-data/processed"
TRIALS = f"{PROC}/processed_all_data_accu_60_filtered.csv"
META = f"{PROC}/processed_metadata_all_data_accu_60.csv"


def load():
    d = pd.read_csv(TRIALS)
    m = pd.read_csv(META)
    s2t = m.drop_duplicates("subject_id").set_index("subject_id")["trem_vs_brady_type"].to_dict()
    d["subtype"] = d.subject_id.map(s2t)
    d["coh"] = d.signed_coherence / 100.0
    return d, m


# ---------------------------------------------------------------- probe 1
def credit_assignment(d):
    """Split 1-back reward history by whether the previous trial shared the
    current trial's colour context. Dopamine re-routing credit from action to
    context predicts a selective rc_same x medication interaction."""
    d = d.copy()
    d["c"] = d.choice * 2 - 1
    d["rc"] = d["c"] * np.where(d.outcome == 1, 1, -1)
    g = d.groupby("session_filename")
    d["prev_rc"], d["prev_c"] = g["rc"].shift(), g["c"].shift()
    d["prev_col"] = g["color"].shift()
    d = d.dropna(subset=["prev_rc", "prev_col"]).copy()

    same = (d.prev_col == d.color).astype(float)
    d["rc_same"], d["rc_diff"] = d.prev_rc * same, d.prev_rc * (1 - same)
    d["c_same"], d["c_diff"] = d.prev_c * same, d.prev_c * (1 - same)
    d["col"] = np.where(d.color == 1, 0.5, -0.5)
    d["med"] = np.where(d.medication == "on", 0.5, -0.5)

    fml = "choice ~ coh + col*med + rc_same*med + rc_diff*med + c_same*med + c_diff*med"
    vc = {"s_int": "0+C(subj)", "s_rcs": "0+C(subj):rc_same",
          "s_rcd": "0+C(subj):rc_diff", "s_med": "0+C(subj):med", "s_col": "0+C(subj):col"}
    keep = ["col", "col:med", "rc_same", "rc_same:med", "rc_diff", "rc_diff:med",
            "c_same", "c_same:med", "c_diff", "c_diff:med"]

    print("\n" + "=" * 70 + "\nPROBE 1: context-specific credit assignment\n" + "=" * 70)
    for lab, sub in [("HEALTHY CONTROLS", d[d.group == "hc"]), ("ALL PD", d[d.group == "pd"]),
                     ("TREMOR-DOMINANT", d[d.subtype == "tremor"]),
                     ("BRADYKINESIA-DOMINANT", d[d.subtype == "bradykinetic"])]:
        sub = sub.copy()
        sub["subj"] = sub.subject_id
        if lab == "HEALTHY CONTROLS":  # single session, no medication factor
            f2 = "choice ~ coh + col + rc_same + rc_diff + c_same + c_diff"
            v2 = {"s_int": "0+C(subj)", "s_rcs": "0+C(subj):rc_same", "s_rcd": "0+C(subj):rc_diff"}
            r = BinomialBayesMixedGLM.from_formula(f2, v2, sub).fit_vb(verbose=False)
        else:
            r = BinomialBayesMixedGLM.from_formula(fml, vc, sub).fit_vb(verbose=False)
        fe = pd.DataFrame({"est": r.fe_mean, "sd": r.fe_sd}, index=r.model.exog_names)
        fe["z"] = fe.est / fe.sd
        fe["p"] = 2 * (1 - norm.cdf(fe.z.abs()))
        print(f"\n--- {lab}: {sub.subject_id.nunique()} subj, {len(sub)} trials ---")
        print(fe.loc[[k for k in keep if k in fe.index]].round(4).to_string())


# ---------------------------------------------------------------- probe 2
def _qtraj(col, ch, rw, a_pos, a_neg):
    """Q(colour, action) delta-rule trajectory. Exogenous given the learning
    rates because updates use OBSERVED choices/outcomes, not model predictions."""
    n = len(ch)
    q = np.full(4, 0.5)
    out = np.empty(n)
    for i in range(n):
        c = col[i] << 1
        out[i] = q[c + 1] - q[c]
        j = c + ch[i]
        a = a_pos if rw[i] == 1.0 else a_neg
        q[j] += a * (rw[i] - q[j])
    return out


def _newton(X, y, iters=25):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        W = p * (1 - p) + 1e-9
        try:
            step = np.linalg.solve((X * W[:, None]).T @ X + 1e-6 * np.eye(len(b)), X.T @ (y - p))
        except np.linalg.LinAlgError:
            break
        b = b + step
        if np.max(np.abs(step)) < 1e-8:
            break
    p = np.clip(1 / (1 + np.exp(-np.clip(X @ b, -30, 30))), 1e-12, 1 - 1e-12)
    return b, float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def rl_fits(d):
    """Per-session context-conditioned RL. Grid the learning rate(s); the
    logistic read-out is then convex and solved exactly by Newton."""
    grid = np.round(np.logspace(np.log10(0.005), np.log10(0.8), 28), 5)
    coarse = grid[::2]
    rows = []
    for sf, sub in d.groupby("session_filename"):
        col = sub.color.values.astype(np.int64)
        ch = sub.choice.values.astype(np.int64)
        rw = sub.outcome.values.astype(float)
        y, n = ch.astype(float), len(sub)
        base = [np.ones(n), sub.coh.values]

        best = (-np.inf,)
        for a in grid:
            b, ll = _newton(np.column_stack(base + [_qtraj(col, ch, rw, a, a)]), y)
            if ll > best[0]:
                best = (ll, a, b)
        ll1, a1, b1 = best

        best2 = (-np.inf,)
        for ap in coarse:
            for am in coarse:
                b, ll = _newton(np.column_stack(base + [_qtraj(col, ch, rw, ap, am)]), y)
                if ll > best2[0]:
                    best2 = (ll, ap, am, b)
        ll2, ap2, am2, b2 = best2
        _, ll0 = _newton(np.column_stack(base), y)

        rows.append(dict(
            session_filename=sf, subject_id=sub.subject_id.iloc[0], group=sub.group.iloc[0],
            subtype=sub.subtype.iloc[0],
            medication=sub.medication.iloc[0] if isinstance(sub.medication.iloc[0], str) else "none",
            alpha=a1, beta_s=b1[1], beta_v=b1[2], alpha_pos=ap2, alpha_neg=am2,
            bic0=-2 * ll0 + 2 * np.log(n), bic1=-2 * ll1 + 3 * np.log(n),
            bic2=-2 * ll2 + 4 * np.log(n)))
    R = pd.DataFrame(rows)
    R["asym"] = R.alpha_pos - R.alpha_neg
    R.to_csv(f"{PROC}/rl_fits.csv", index=False)   # consumed by rl_runnerup_probes.py

    print("\n" + "=" * 70 + "\nPROBE 2: RL fits (learning rate vs value->choice gain)\n" + "=" * 70)
    print("RL beats no-RL (BIC) in %d/%d sessions; dual-rate beats single in %d/%d"
          % ((R.bic1 < R.bic0).sum(), len(R), (R.bic2 < R.bic1).sum(), len(R)))
    print("\n--- OFF vs ON, paired Wilcoxon ---")
    for lab, g in [("All PD", R[R.group == "pd"]), ("Tremor", R[R.subtype == "tremor"]),
                   ("Brady", R[R.subtype == "bradykinetic"])]:
        for v in ["alpha", "alpha_pos", "alpha_neg", "asym", "beta_v", "beta_s"]:
            piv = g.pivot_table(index="subject_id", columns="medication", values=v).dropna()
            if len(piv) < 5:
                continue
            _, pv = stats.wilcoxon(piv["off"], piv["on"])
            print(f"{lab:7s} {v:10s} n={len(piv):2d} OFF={piv['off'].mean():+.3f} "
                  f"ON={piv['on'].mean():+.3f} p={pv:.4f}")
    print("\n--- HC vs PD-OFF, Mann-Whitney ---")
    hc, po = R[R.group == "hc"], R[(R.group == "pd") & (R.medication == "off")]
    for v in ["alpha", "asym", "beta_v", "beta_s"]:
        _, pv = stats.mannwhitneyu(hc[v], po[v])
        print(f"{v:10s} HC={hc[v].mean():+.3f} PDoff={po[v].mean():+.3f} p={pv:.4f}")
    return R


# ---------------------------------------------------------------- probe 3
def explicitness(m):
    """Does self-reported awareness of the colour contingency track either
    medication state or the size of the prior effect?"""
    p = m[m.is_pd == 1].copy()
    p["prior_eff"] = p.positive_bias - p.equal_bias
    print("\n" + "=" * 70 + "\nPROBE 3: explicit awareness vs prior expression\n" + "=" * 70)
    piv = p.pivot_table(index="subject_id", columns="treatment",
                        values="explicitness", aggfunc="first").dropna()
    print(f"PD subjects with an awareness rating in BOTH sessions: {len(piv)}")
    print(pd.crosstab(piv.OFF, piv.ON, margins=True).to_string())
    _, pv = stats.wilcoxon(piv.OFF, piv.ON, zero_method="zsplit")
    print(f"paired Wilcoxon  OFF={piv.OFF.mean():.2f}  ON={piv.ON.mean():.2f}  p={pv:.4f}")

    sub = p.dropna(subset=["explicitness"])
    print("\n--- prior effect by awareness level ---")
    print(sub.groupby(["treatment", "explicitness"]).prior_eff.agg(["mean", "size"]).round(4).to_string())
    r = stats.spearmanr(sub.explicitness, sub.prior_eff)
    print(f"\nSpearman awareness vs prior effect: rho={r.statistic:+.3f} p={r.pvalue:.4f} n={len(sub)}")
    for t in ["OFF", "ON"]:
        s2 = sub[sub.treatment == t]
        r = stats.spearmanr(s2.explicitness, s2.prior_eff)
        print(f"  {t}: rho={r.statistic:+.3f} p={r.pvalue:.4f} n={len(s2)}")


# ---------------------------------------------------------------- probe 4
def per_cell(d, m):
    """Fully-crossed breakdown: fit every subtype x medication cell SEPARATELY
    rather than reading them off interaction terms, plus a precision check on
    what these cell sizes can actually detect."""
    from scipy.optimize import brentq

    d = d.copy()
    d["subtype"] = d.subtype.fillna("unclassified")
    d["c"] = d.choice * 2 - 1
    d["rc"] = d["c"] * np.where(d.outcome == 1, 1, -1)
    g = d.groupby("session_filename")
    d["prev_rc"], d["prev_c"] = g["rc"].shift(), g["c"].shift()
    d["prev_col"] = g["color"].shift()
    d = d.dropna(subset=["prev_rc", "prev_col"]).copy()
    same = (d.prev_col == d.color).astype(float)
    d["rc_same"], d["rc_diff"] = d.prev_rc * same, d.prev_rc * (1 - same)
    d["c_same"], d["c_diff"] = d.prev_c * same, d.prev_c * (1 - same)
    d["col"] = np.where(d.color == 1, 0.5, -0.5)

    cells = [("Healthy controls", d[d.group == "hc"])]
    for st, lab in [("tremor", "Tremor"), ("bradykinetic", "Brady"), ("unclassified", "Unclassif.")]:
        for med in ["off", "on"]:
            cells.append((f"{lab} {med.upper()}", d[(d.subtype == st) & (d.medication == med)]))

    fml = "choice ~ coh + col + rc_same + rc_diff + c_same + c_diff"
    vc = {"s_int": "0+C(subj)", "s_col": "0+C(subj):col", "s_rcs": "0+C(subj):rc_same"}
    print("\n" + "=" * 88 + "\nPROBE 4a: per-cell trial-level GLMM (each cell fit separately)\n" + "=" * 88)
    print(f"{'cell':18s} {'n':>3s} {'trials':>7s} | {'colour prior [95% CI]':>28s} | {'rc_same':>8s}")
    for lab, sub in cells:
        sub = sub.copy()
        sub["subj"] = sub.subject_id
        r = BinomialBayesMixedGLM.from_formula(fml, vc, sub).fit_vb(verbose=False)
        fe = pd.DataFrame({"est": r.fe_mean, "sd": r.fe_sd}, index=r.model.exog_names)
        e, sd = fe.loc["col", "est"], fe.loc["col", "sd"]
        ci = f"{e:+.3f} [{e - 1.96 * sd:+.2f},{e + 1.96 * sd:+.2f}]"
        print(f"{lab:18s} {sub.subject_id.nunique():3d} {len(sub):7d} | {ci:>28s} | "
              f"{fe.loc['rc_same', 'est']:+8.3f}")

    def dz_needed(n, power=0.80, alpha=0.05):
        def f(dz):
            return stats.nct.sf(stats.t.ppf(1 - alpha / 2, n - 1), n - 1, dz * np.sqrt(n)) - power
        return brentq(f, 0.01, 2.5) * 1.05   # ~5% Wilcoxon penalty

    mm = m.copy()
    mm["subtype"] = mm.subject_id.map(
        m.drop_duplicates("subject_id").set_index("subject_id")["trem_vs_brady_type"].to_dict()
    ).fillna("unclassified")
    mm["prior_eff"] = mm.positive_bias - mm.equal_bias
    print("\n" + "=" * 88 + "\nPROBE 4b: detectable vs observed within-subject effect (OFF->ON prior change)\n" + "=" * 88)
    print(f"{'group':16s} {'n':>3s} {'dz needed':>10s} {'dz observed':>12s} {'p':>8s}  verdict")
    for lab, q in [("Tremor", mm[mm.subtype == "tremor"]),
                   ("Brady", mm[mm.subtype == "bradykinetic"]),
                   ("Unclassified", mm[mm.subtype == "unclassified"]),
                   ("All PD", mm[mm.is_pd == 1])]:
        piv = q.pivot_table(index="subject_id", columns="treatment", values="prior_eff").dropna()
        dif = piv.ON - piv.OFF
        dz = dif.mean() / dif.std(ddof=1)
        _, pv = stats.wilcoxon(piv.OFF, piv.ON)
        need = dz_needed(len(piv))
        verdict = "detectable" if abs(dz) >= need else "UNDERPOWERED"
        print(f"{lab:16s} {len(piv):3d} {need:10.2f} {dz:+12.2f} {pv:8.4f}  {verdict}")
    print("\n  n required for 80% power at the observed tremor dz=0.56: n = 30")


if __name__ == "__main__":
    d, m = load()
    credit_assignment(d)
    rl_fits(d)
    explicitness(m)
    per_cell(d, m)
