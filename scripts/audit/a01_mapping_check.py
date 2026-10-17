"""Audit: infer each session's colour/orientation prior from its own target sequence and compare with the labels."""
import numpy as np
import pandas as pd

from src.data_io import catalog

c = catalog.load_trials(stage="compiled", min_easy_accuracy=None, max_equal_prior_bias=None)
rows = []
for sess, d in c.groupby("session_filename", sort=False):
    rec = {"session_filename": sess, "subject_id": d.subject_id.iat[0], "medication": d.medication.iat[0], "prior_label": d.prior.iat[0], "n": len(d)}
    for col in ("green", "red"):
        dd = d[d.color == col]
        rec[f"p_right_{col}"] = (dd.target == "right").mean()
        rec[f"n_{col}"] = len(dd)
    # inferred favoured colour = the colour whose target distribution is further from 0.5
    dg, dr = abs(rec["p_right_green"] - 0.5), abs(rec["p_right_red"] - 0.5)
    fav = "g" if dg > dr else "r"
    p = rec[f"p_right_{'green' if fav == 'g' else 'red'}"]
    rec["prior_inferred"] = fav + ("r" if p > 0.5 else "l")
    rec["dev_fav"], rec["dev_other"] = max(dg, dr), min(dg, dr)
    rows.append(rec)
s = pd.DataFrame(rows)
s["file_code"] = s.session_filename.str.extract(r"_([gr][lr])\.parquet$")[0]
s["label_ok"] = s.prior_label == s.prior_inferred
s["file_ok"] = s.file_code == s.prior_inferred
catalog.save_table(s, "audit/a01_session_mapping_check.csv", index=False)
pd.set_option("display.width", 250)
print(s[(~s.label_ok) | (~s.file_ok) | (s.prior_label == "eq")].to_string())
print(len(s), "sessions;", (~s.label_ok & (s.prior_label != "eq")).sum(), "label mismatches")
