"""Posterior-predictive-style check of the drift-only Wiener model: refit per session, simulate the trial list 5 times,
compare choice proportions and RT quantiles (correct and error) with the data, pooled by group."""
from multiprocessing import Pool

import numpy as np
import pandas as pd

from scripts.audit.a13_ddm_compare import BASE, TMAX, fit, session_data, simulate, unpack
from scripts.audit.common import TWO_PRIOR, trials
from src.data_io import catalog


def one(args):
    sid, d = args
    d = d[d.reaction_time / 1000 < TMAX]
    D = session_data(d)
    names = BASE + ["dv"]
    p = unpack(fit(names, D, starts=3).x, names)
    rng = np.random.default_rng(abs(hash(sid)) % 2 ** 32)
    rows = []
    for src, S in [("data", D)] + [("model", simulate(p, D, rng)) for _ in range(5)]:
        df = pd.DataFrame({"rt": S["rt"], "fav": S["up"], "pos": S["pos"], "lev": S["lev"], "sc": S["sc"]})
        df["correct"] = np.where(df.lev == 0, np.nan, (df.fav == (df.sc > 0)).astype(float))
        df["session_id"], df["src"] = sid, src
        rows.append(df)
    return pd.concat(rows)


if __name__ == "__main__":
    t = trials(); t = t[~t.session_id.isin(TWO_PRIOR)]
    with Pool(16) as pool:
        out = pd.concat(pool.map(one, list(t.groupby("session_id"))))
    s = pd.read_csv("outputs/audit/a04_session_design.csv").set_index("session_id")
    out["cond"] = out.session_id.map(lambda x: "hc" if s.loc[x, "group"] == "hc" else s.loc[x, "med"])
    out["signed"] = out.lev * np.where(out.sc < 0, -1, 1)
    ch = out.groupby(["cond", "src", "pos", "signed"]).fav.mean().unstack("src")
    ch["diff"] = ch.model - ch.data
    rtq = out[out.lev > 0].groupby(["cond", "src", "lev", "correct"]).rt.quantile([0.1, 0.5, 0.9]).unstack("src")
    rtq["diff_ms"] = (rtq.model - rtq.data) * 1000
    rt0 = out[out.lev == 0].groupby(["cond", "src", "pos", "fav"]).rt.quantile([0.1, 0.5, 0.9]).unstack("src")
    rt0["diff_ms"] = (rt0.model - rt0.data) * 1000
    catalog.save_table(ch.reset_index(), "audit/a18_ppc_choice.csv", index=False)
    catalog.save_table(rtq.reset_index(), "audit/a18_ppc_rt.csv", index=False)
    catalog.save_table(rt0.reset_index(), "audit/a18_ppc_rt0.csv", index=False)
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
    print("choice: max |model-data| by cond:", ch.groupby("cond")["diff"].apply(lambda x: x.abs().max()).round(3).to_dict())
    print(ch.loc[(slice(None), slice(None), 0)].round(3) if False else ch.xs(0, level="signed").round(3))
    print(rtq.round(3).to_string())
    print(rt0.round(3).to_string())
