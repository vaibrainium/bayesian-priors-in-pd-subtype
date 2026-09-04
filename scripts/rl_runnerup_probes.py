"""Runner-up probes for the RL follow-on paper (see dissemination/rl_paper_report.md).

Run:  .venv/bin/python scripts/rl_runnerup_probes.py

  A  two_channels  -- reward-driven vs observation-driven acquisition of the prior
  B  reversal      -- contextual reversal across sessions; robustness of the med effect
  C  state_gain    -- do GLM-HMM states organise by RL value read-out gain?
  D  equivalence   -- are the history-side nulls evidence of ABSENCE at trial level?

Headline outcomes: A is structurally impossible in this design, B passes as a
robustness check, C is refuted with a working positive control, D is what makes
the elimination argument usable.
"""
import pickle
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

warnings.filterwarnings("ignore")
PROC = "/mnt/pd-data/processed"
HMM = f"{PROC}/shared_glm_hmm/XY__tgt/all_subjects_final.pkl"


def load():
    d = pd.read_csv(f"{PROC}/processed_all_data_accu_60_filtered.csv")
    m = pd.read_csv(f"{PROC}/processed_metadata_all_data_accu_60.csv")
    s2t = m.drop_duplicates("subject_id").set_index("subject_id")["trem_vs_brady_type"].to_dict()
    d["subtype"] = d.subject_id.map(s2t).fillna("unclassified")
    d["coh"] = d.signed_coherence / 100.0
    return d, m, s2t


def _newton(X, y, w=None, iters=30, ridge=1e-6):
    w = np.ones(len(y)) if w is None else w
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        W = w * (p * (1 - p)) + 1e-9
        try:
            step = np.linalg.solve((X * W[:, None]).T @ X + ridge * np.eye(len(b)),
                                   X.T @ (w * (y - p)))
        except np.linalg.LinAlgError:
            break
        b = b + step
        if np.max(np.abs(step)) < 1e-8:
            break
    p = np.clip(1 / (1 + np.exp(-np.clip(X @ b, -30, 30))), 1e-12, 1 - 1e-12)
    return b, float(np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p))))


# ---------------------------------------------------------------- probe A
def two_channels(d):
    """Reward-driven value (updated only at 0% coherence, where reward is the only
    signal) vs observation-driven base rate (updated only at >=35%, where perception
    is reliable). The two are unidentifiable because the design holds the base rate
    constant across coherence -- both estimate the same number."""
    def traj(col, ch, rw, tgt, coh, a_rew, a_obs):
        q, b = np.full(4, 0.5), np.full(2, 0.5)
        dq, db = np.empty(len(ch)), np.empty(len(ch))
        for i in range(len(ch)):
            c = col[i]
            dq[i] = q[(c << 1) + 1] - q[c << 1]
            db[i] = 2 * b[c] - 1
            if coh[i] == 0.0:
                j = (c << 1) + ch[i]
                q[j] += a_rew * (rw[i] - q[j])
            elif abs(coh[i]) >= 0.35:
                b[c] += a_obs * (tgt[i] - b[c])
        return dq, db

    grid = np.round(np.logspace(np.log10(0.01), np.log10(0.6), 12), 4)
    rows = []
    for sf, sub in d.groupby("session_filename"):
        col = sub.color.values.astype(np.int64)
        ch = sub.choice.values.astype(np.int64)
        rw, tgt = sub.outcome.values.astype(float), sub.target.values.astype(float)
        coh, y, n = sub.coh.values, sub.choice.values.astype(float), len(sub)
        best = (-np.inf,)
        for ar in grid:
            for ao in grid:
                dq, db = traj(col, ch, rw, tgt, coh, ar, ao)
                b, ll = _newton(np.column_stack([np.ones(n), coh, dq, db]), y)
                if ll > best[0]:
                    best = (ll, ar, ao, b, dq, db)
        _, ar, ao, b, dq, db = best
        rows.append(dict(subject_id=sub.subject_id.iloc[0], group=sub.group.iloc[0],
                         subtype=sub.subtype.iloc[0],
                         medication=sub.medication.iloc[0] if isinstance(sub.medication.iloc[0], str) else "none",
                         beta_rew=b[2], beta_obs=b[3], alpha_rew=ar, alpha_obs=ao,
                         corr=np.corrcoef(dq, db)[0, 1]))
    R = pd.DataFrame(rows)
    print("\n" + "=" * 88 + "\nPROBE A: reward vs observation channel\n" + "=" * 88)
    print(f"median corr(dQ_rew, dQ_obs) = {R['corr'].median():.3f}   <-- collinearity")
    print(R.groupby(["group", "subtype", "medication"], dropna=False)[
        ["beta_rew", "beta_obs"]].agg(["mean", "size"]).round(3).to_string())
    print("\nNOTE: base rate is constant across coherence by design "
          "(P(target=1) = .748/.747/.746/.745), so the two channels estimate the same\n"
          "quantity and are not identifiable from choices. Coefficients sign-flip across "
          "groups. Retire this idea.")
    return R


# ---------------------------------------------------------------- probe B
def reversal(d, m, s2t):
    """Colour/direction remapping between sessions. Used as a robustness control on
    the medication effect, NOT as a finding -- the flip cells are far too small."""
    sess = d.groupby(["subject_id", "medication"])[["prior_color", "prior_direction"]].first().reset_index()
    pc = sess.pivot_table(index="subject_id", columns="medication", values="prior_color", aggfunc="first").dropna()
    pdir = sess.pivot_table(index="subject_id", columns="medication", values="prior_direction", aggfunc="first").dropna()
    flip = pd.DataFrame({"col_flip": (pc.off != pc.on).astype(int),
                         "dir_flip": (pdir.off != pdir.on).astype(int)})
    print("\n" + "=" * 88 + "\nPROBE B: contextual reversal\n" + "=" * 88)
    print(f"colour flipped: {int(flip.col_flip.sum())}/{len(flip)}   "
          f"direction flipped: {int(flip.dir_flip.sum())}/{len(flip)}")

    dd = d[d.group == "pd"].merge(flip, left_on="subject_id", right_index=True, how="left").dropna(subset=["col_flip"])
    dd["col_"] = np.where(dd.color == 1, 0.5, -0.5)
    dd["med"] = np.where(dd.medication == "on", 0.5, -0.5)
    dd["cf"], dd["df"] = dd.col_flip * 2 - 1, dd.dir_flip * 2 - 1
    for st in ["tremor", "bradykinetic"]:
        sub = dd[dd.subtype == st].copy()
        sub["subj"] = sub.subject_id
        r = BinomialBayesMixedGLM.from_formula(
            "choice ~ coh + col_*med + col_*cf + col_*df",
            {"s_int": "0+C(subj)", "s_col": "0+C(subj):col_", "s_med": "0+C(subj):med"},
            sub).fit_vb(verbose=False)
        fe = pd.DataFrame({"est": r.fe_mean, "sd": r.fe_sd}, index=r.model.exog_names)
        fe["z"] = fe.est / fe.sd
        fe["p"] = 2 * (1 - norm.cdf(fe.z.abs()))
        n_cf = sub.groupby("subject_id").cf.first().value_counts().to_dict()
        n_df = sub.groupby("subject_id").df.first().value_counts().to_dict()
        print(f"\n  {st} (n={sub.subject_id.nunique()}); colour-flip split {n_cf}, dir-flip split {n_df}")
        print(fe.loc[[k for k in ["col_", "col_:med", "col_:cf", "col_:df"] if k in fe.index]].round(4).to_string())
    print("\nNOTE: colour x medication is unchanged by these controls (+0.4699 vs published "
          "+0.4706).\nThe flip terms themselves rest on 1-2 subjects and are pseudo-replication.")


# ---------------------------------------------------------------- probe C
def state_gain(d, s2t, alphas):
    """Do GLM-HMM states organise by RL value read-out gain, or just by colour?
    Positive control: the same per-state fits must recover the HMM's own colour weights.
    NOTE: stored HMM weights are sign-inverted relative to P(choice=1)."""
    o = pickle.load(open(HMM, "rb"))
    feat = o["config"]["model_features"]
    cix = feat.index("color")

    def qtraj(col, ch, rw, a):
        q, out = np.full(4, 0.5), np.empty(len(ch))
        for i in range(len(ch)):
            c = col[i] << 1
            out[i] = q[c + 1] - q[c]
            j = c + ch[i]
            q[j] += a * (rw[i] - q[j])
        return out

    rows = []
    for sid, df in o["data"].items():
        sub = d[d.session_id == sid]
        if len(sub) == 0:
            continue
        sf = sub.session_filename.iloc[0]
        if sf not in alphas:
            continue
        msk = np.asarray(df["mask"]).astype(bool)
        X = np.column_stack([np.asarray(df[f]) for f in feat])[msk]
        y = np.asarray(df["choices"])[msk].astype(int)
        hmm = o["model"]["models"][sid]
        Ez, _, _ = hmm.expected_states(y.reshape(-1, 1), input=X)
        s = sub.iloc[1:]                       # HMM drops trial 1 for history regressors
        assert len(s) == msk.sum(), (sid, len(s), msk.sum())
        dq = qtraj(s.color.values.astype(np.int64), s.choice.values.astype(np.int64),
                   s.outcome.values.astype(float), alphas[sf])
        colv = s.color.values.astype(float)
        A = np.column_stack([np.ones(len(colv)), colv])
        dq_r = dq - A @ np.linalg.lstsq(A, dq, rcond=None)[0]
        sd = dq_r.std()
        dq_r = dq_r / sd if sd > 1e-9 else dq_r
        coh, yy = (s.signed_coherence / 100).values, s.choice.values.astype(float)
        W = np.array(hmm.observations.params)[:, 0, :]
        for k in range(hmm.K):
            w = Ez[:, k]
            if w.sum() < 40:
                continue
            b, _ = _newton(np.column_stack([np.ones(len(yy)), coh, colv, dq_r]), yy, w, ridge=1e-3)
            rows.append(dict(session_id=sid, state=k, beta_col=b[2], beta_resid=b[3],
                             hmm_color=-W[k, cix]))
    R = pd.DataFrame(rows)
    R["subject_id"] = R.session_id.str.replace("_OFF", "", regex=False).str.replace("_ON", "", regex=False)
    R["subtype"] = R.subject_id.map(s2t).fillna("unclassified")
    R["medication"] = np.where(R.session_id.str.endswith("_ON"), "on",
                               np.where(R.session_id.str.endswith("_OFF"), "off", "none"))
    R["group"] = np.where(R.medication == "none", "hc", "pd")

    print("\n" + "=" * 88 + "\nPROBE C: GLM-HMM state identity vs RL value gain\n" + "=" * 88)
    print(f"{'group':18s} {'n':>4s} | {'colour wt (control)':>22s} | {'RESIDUAL value':>22s}")
    for lab, g in [("ALL", R), ("Healthy controls", R[R.group == "hc"]),
                   ("Tremor OFF", R[(R.subtype == "tremor") & (R.medication == "off")]),
                   ("Tremor ON", R[(R.subtype == "tremor") & (R.medication == "on")]),
                   ("Brady OFF", R[(R.subtype == "bradykinetic") & (R.medication == "off")]),
                   ("Brady ON", R[(R.subtype == "bradykinetic") & (R.medication == "on")])]:
        if len(g) < 8:
            continue
        r1 = stats.spearmanr(g.hmm_color, g.beta_col)
        r2 = stats.spearmanr(g.hmm_color, g.beta_resid)
        print(f"{lab:18s} {len(g):4d} | rho={r1.statistic:+.3f} p={r1.pvalue:.1e} |"
              f" rho={r2.statistic:+.3f} p={r2.pvalue:.4f}")
    print("\nNOTE: the raw (non-orthogonalised) correlation is rho=+0.84 but dQ is largely the\n"
          "colour regressor (|r|=0.83 median). With colour partialled out the residual value\n"
          "signal carries no state information. States organise by colour, not by value gain.")
    return R


# ---------------------------------------------------------------- probe D
def equivalence():
    """Are the history-side nulls evidence of ABSENCE? ROPE = half the colour x
    medication effect, i.e. the smallest effect we would call comparable."""
    eff = {"colour x med (moves)": (0.4756, 0.0822),
           "rc_same x med": (-0.1236, 0.0594),
           "rc_diff x med": (-0.0844, 0.0570),
           "context-specificity x med": (-0.1236 + 0.0844, np.hypot(0.0594, 0.0570)),
           "c_same x med": (-0.0096, 0.0594),
           "c_diff x med": (-0.0578, 0.0570)}
    rope = 0.4756 / 2
    print("\n" + "=" * 88 + f"\nPROBE D: equivalence tests (tremor, trial level; ROPE = +/-{rope:.3f})\n" + "=" * 88)
    print(f"{'effect':30s} {'est':>8s} {'95% CI':>20s} {'P(|eff|<ROPE)':>14s}")
    for k, (e, s) in eff.items():
        p_in = norm.cdf(rope, e, s) - norm.cdf(-rope, e, s)
        print(f"{k:30s} {e:+8.3f} [{e-1.96*s:+.3f},{e+1.96*s:+.3f}] {p_in:14.3f}")
    print("\nNOTE: rescues TRIAL-level nulls only. Learning rate, alpha+/alpha- and awareness\n"
          "are subject-level n=11 and need a hierarchical refit to reach the same regime.")


if __name__ == "__main__":
    d, m, s2t = load()
    two_channels(d)
    reversal(d, m, s2t)
    try:
        alphas = pd.read_csv(f"{PROC}/rl_fits.csv").set_index("session_filename")["alpha"].to_dict()
        state_gain(d, s2t, alphas)
    except FileNotFoundError:
        print(f"\n[probe C skipped: run rl_paper_probes.py first -- it writes {PROC}/rl_fits.csv]")
    equivalence()
