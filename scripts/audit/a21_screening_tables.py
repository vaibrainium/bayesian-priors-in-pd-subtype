"""Subject- and site-level tables for docs/data-screening.md: every subject with compiled data, its accuracy,
design check and MoCA, and whether it is kept at each accuracy threshold with and without a strict MoCA rule.
Prints Markdown and writes outputs/audit/screening_subjects.csv."""
import numpy as np
import pandas as pd

from config import main_config
from src.data_io import catalog

THRESHOLDS = (0.6, 0.8, 0.85, 0.9)
MANUAL = set(main_config.moca_rejection)

acc = catalog._compiled_session_accuracy()  # subject_id, session_filename, medication, easy_accuracy_pct (biased sessions)
design = catalog._session_design("compiled")
acc = acc.join(design, on="session_filename")
meta = catalog.load_metadata(min_easy_accuracy=None, prior_design_tolerance=None)
site = meta.drop_duplicates("subject_id").set_index("subject_id").experiment_site
moca = meta.groupby("subject_id").MoCA.agg(["min", "max"])
is_pd = meta.drop_duplicates("subject_id").set_index("subject_id").is_pd

rows = []
for subj, d in acc.groupby("subject_id", sort=False):
    group = "PD" if (subj in is_pd.index and is_pd[subj] == 1) or subj in MANUAL else "HC"
    if subj not in is_pd.index and subj not in MANUAL:
        group = "PD" if d.medication.notna().any() else "HC"
    st = site.get(subj, "Stanford" if subj in MANUAL else "?")
    rec = {"subject": subj, "group": group, "site": st}
    for med in ("off", "on"):
        s = d[d.medication == med] if group == "PD" else d
        if s.empty:
            rec[f"acc_{med}"], rec[f"design_{med}"] = np.nan, ""
        else:
            r = s.iloc[0]
            rec[f"acc_{med}"] = r.easy_accuracy_pct
            rec[f"design_{med}"] = "ok" if r.prior_design_error <= 0.05 else f"{r.biased_pct:.2f} / {r.equal_pct:.2f}"
        if group == "HC":
            break
    rec["n_sessions"] = len(d)
    mn, mx = (moca.loc[subj, "min"], moca.loc[subj, "max"]) if subj in moca.index else (np.nan, np.nan)
    rec["moca"] = "manual list (no score in data)" if subj in MANUAL else ("none recorded" if np.isnan(mn) else (f"{mn:.0f}" if mn == mx else f"{mn:.0f} / {mx:.0f}"))
    rec["moca_min"] = mn
    accs = d.easy_accuracy_pct
    design_ok = (d.prior_design_error <= 0.05).all()
    complete = group == "HC" or set(d.medication.dropna()) == {"off", "on"}
    base_ok = design_ok and complete and subj not in MANUAL
    for th in THRESHOLDS:
        rec[f"keep_{int(th * 100)}"] = bool(base_ok and (accs >= th * 100).all())
    rec["strict_moca_ok"] = group == "HC" or not (mn <= 26)
    reasons = []
    if subj in MANUAL:
        reasons.append("manual MoCA list")
    if not complete:
        reasons.append("only one session")
    if not design_ok:
        reasons.append("design")
    if (accs < 60).any():
        reasons.append("accuracy < 60%")
    rec["rejected_at_60_because"] = ", ".join(reasons)
    rows.append(rec)
t = pd.DataFrame(rows).sort_values(["group", "site", "subject"], key=lambda c: c.str.replace(r"(\d+)", lambda m: m.group(1).zfill(3), regex=True) if c.name == "subject" else c)
# sanity check against the loader
for th in THRESHOLDS:
    loaded = set(catalog.load_metadata(min_easy_accuracy=th).subject_id)
    assert loaded == set(t[t[f"keep_{int(th * 100)}"]].subject), th
catalog.save_table(t, "audit/screening_subjects.csv", index=False)


def fmt_acc(x):
    return "" if pd.isna(x) else f"{x:.0f}"


def yn(b):
    return "yes" if b else "no"


for group in ("PD", "HC"):
    g = t[t.group == group]
    print(f"\n### {group}\n")
    if group == "PD":
        print("| Subject | Site | Acc OFF | Acc ON | Design OFF | Design ON | MoCA | 60% | 80% | 85% | 90% | MoCA > 26 | Rejected at 60% because |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in g.iterrows():
            print(f"| {r.subject} | {r.site} | {fmt_acc(r.acc_off)} | {fmt_acc(r.acc_on)} | {r.design_off} | {r.design_on} | {r.moca} | "
                  f"{yn(r.keep_60)} | {yn(r.keep_80)} | {yn(r.keep_85)} | {yn(r.keep_90)} | {yn(r.strict_moca_ok)} | {r.rejected_at_60_because} |")
    else:
        print("| Subject | Site | Accuracy | Design | 60% | 80% | 85% | 90% | Rejected at 60% because |")
        print("|---|---|---|---|---|---|---|---|---|")
        for _, r in g.iterrows():
            print(f"| {r.subject} | {r.site} | {fmt_acc(r.acc_off)} | {r.design_off} | {yn(r.keep_60)} | {yn(r.keep_80)} | {yn(r.keep_85)} | {yn(r.keep_90)} | {r.rejected_at_60_because} |")

print("\n### Sessions per site (OFF = ON = patients kept; HC = controls)\n")
print("| Site | " + " | ".join(f"{int(th * 100)}%" for th in THRESHOLDS) + " | " + " | ".join(f"{int(th * 100)}% + MoCA > 26" for th in THRESHOLDS) + " |")
print("|---|" + "---|" * (2 * len(THRESHOLDS)))
for st in ["Case_Western", "Harvard", "Stanford", "UCLA", "Total"]:
    cells = []
    for strict in (False, True):
        for th in THRESHOLDS:
            k = t[t[f"keep_{int(th * 100)}"] & (t.strict_moca_ok if strict else True)]
            if st != "Total":
                k = k[k.site == st]
            cells.append(f"{(k.group == 'PD').sum()} PD, {(k.group == 'HC').sum()} HC")
    print(f"| {st} | " + " | ".join(cells) + " |")
