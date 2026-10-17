"""Shared helpers for the manuscript audit (scripts/audit). Outputs go to outputs/audit/."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.data_io import catalog

TWO_PRIOR = {"HC13", "P28_OFF", "P29_OFF", "P32_ON", "P33_ON", "P9_ON"}  # both colours biased (a01)


def trials(kind="valid", **kw):
    t = catalog.load_trials(kind=kind, **{"max_equal_prior_bias": None, **kw})
    t = t.copy()
    t["signed_choice"] = t["choice"] * 2 - 1
    t["rc"] = t["signed_choice"] * np.where(t["outcome"] == 1, 1, -1)
    t["prev_rc"] = t.groupby("session_filename")["rc"].shift()
    t["prev_choice"] = t.groupby("session_filename")["signed_choice"].shift()
    t["prev_target"] = t.groupby("session_filename")["target"].shift() * 2 - 1
    t["coh"] = t["signed_coherence"] / 100
    t["col"] = np.where(t["color"] == 1, 0.5, -0.5)
    t["two_prior"] = t["session_id"].isin(TWO_PRIOR)
    t["trial_index"] = t.groupby("session_filename").cumcount()
    return t


def session_table():
    """One row per analysed session with design factors (order, mapping, site)."""
    m = catalog.load_metadata(max_equal_prior_bias=None)
    m = m.copy()
    m["group"] = np.where(m["is_pd"] == 1, "pd", "hc")
    m["med"] = m["treatment"].str.lower()
    # session order: off_med_first is per patient; COH has contradictory flags; Case Western missing
    first = m[m.group == "pd"].groupby("subject_id")["off_med_first"].agg(lambda s: s.dropna().unique().tolist())
    m["order_flags"] = m["subject_id"].map(first)
    def visit(r):
        if r.group != "pd":
            return 1
        f = r.order_flags
        if len(f) != 1:
            return np.nan
        off_first = f[0] == 1
        return 1 if (r.med == "off") == off_first else 2
    m["visit"] = m.apply(visit, axis=1)
    m["prior_color"] = m["prior_condition"].str[0]
    m["prior_dir"] = m["prior_condition"].str[1]
    m["two_prior"] = m["session_id"].isin(TWO_PRIOR)
    return m


def logit(y, X):
    """statsmodels Logit with a fallback to light ridge if separation occurs. Returns params, bse."""
    try:
        r = sm.Logit(y, X).fit(disp=0, maxiter=200)
        return r.params, r.bse
    except Exception:
        r = sm.Logit(y, X).fit_regularized(alpha=0.01, disp=0)
        return r.params, pd.Series(np.nan, index=r.params.index)
