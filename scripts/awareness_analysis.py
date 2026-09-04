"""Awareness (`explicitness`) analysis.

Run:  .venv/bin/python scripts/awareness_analysis.py

Coding confirmed by the author 4 Sep 2026:
   0 = fully implicit
   1 = knows OVERALL direction of prior, NOT colour-specific
   2 = knows COLOUR-SPECIFIC prior
All self-learned; subjects were never told about the prior.

Predictions (following Thakur et al. 2021):
  level 1 -> a GENERAL bias, applied to both colours  -> shows up in equal_bias,
             implemented via STARTING POINT
  level 2 -> a COLOUR-SPECIFIC bias                   -> shows up in positive-equal,
             implemented via DRIFT-RATE OFFSET
"""
import numpy as np, pandas as pd
from scipy import stats

P="/mnt/pd-data/processed"
m=pd.read_csv(f"{P}/processed_metadata_all_data_accu_60.csv")
dd=pd.read_csv(f"{P}/processed_metadata_all_data_accu_60_ddm_params.csv")
q=m[m.explicitness.notna()].copy()
q["gen_bias"]  = q.equal_bias - 0.5                      # bias where colour gives no reason
q["col_bias"]  = q.positive_bias - q.equal_bias          # colour-specific effect
q["abs_gen"]   = q.gen_bias.abs()
LAB={0:"0 implicit",1:"1 general only",2:"2 colour-specific"}
q["lvl"]=q.explicitness.map(LAB)

print("n sessions per level:", q.lvl.value_counts().to_dict(), "| subjects:", q.subject_id.nunique())

def show(col, name, signed=True):
    print(f"\n--- {name} ---")
    g=q.groupby("lvl")[col].agg(["mean","sem","size"]).round(4)
    print(g.to_string())
    # one-sample: does each level differ from 0?
    for lv in sorted(q.lvl.unique()):
        x=q[q.lvl==lv][col].dropna()
        if len(x)<4: print(f"   {lv:20s} n={len(x)} too small"); continue
        w,p=stats.wilcoxon(x)
        print(f"   {lv:20s} n={len(x):2d} mean={x.mean():+.4f}  vs 0: p={p:.4f}")
    # targeted contrasts
    a=q[q.explicitness==1][col].dropna(); b=q[q.explicitness==2][col].dropna(); c=q[q.explicitness==0][col].dropna()
    if len(a)>=4 and len(c)>=4:
        u,p=stats.mannwhitneyu(a,c); print(f"   level1 vs level0: p={p:.4f}")
    if len(b)>=4 and len(c)>=4:
        u,p=stats.mannwhitneyu(b,c); print(f"   level2 vs level0: p={p:.4f}")
    if len(a)>=4 and len(b)>=4:
        u,p=stats.mannwhitneyu(a,b); print(f"   level1 vs level2: p={p:.4f}")

print("\n"+"="*78); print("BEHAVIOUR"); print("="*78)
show("col_bias","COLOUR-SPECIFIC effect  (positive_bias - equal_bias)   [predict: level 2 highest]")
show("gen_bias","GENERAL bias in the EQUAL colour  (equal_bias - 0.5)   [predict: level 1 highest]")
show("abs_gen","|general bias|  (direction-agnostic)")

print("\n"+"="*78); print("DDM MECHANISM  (Thakur 2021: general->start point, specific->drift)"); print("="*78)
d=dd[dd.explicitness.notna()].copy()
d["lvl"]=d.explicitness.map(LAB)
d["z_mean"]   = d[["z_prior_cond_1","z_prior_cond_0"]].mean(axis=1)
d["z_dev"]    = (d.z_mean-0.5)
d["z_diff"]   = d.z_prior_cond_1-d.z_prior_cond_0
d["do_diff"]  = d.drift_offset_prior_cond_1-d.drift_offset_prior_cond_0
d["do_mean"]  = d[["drift_offset_prior_cond_1","drift_offset_prior_cond_0"]].mean(axis=1)
for col,name in [("z_dev","STARTING POINT deviation from 0.5, mean of both colours  [predict: level 1]"),
                 ("do_mean","DRIFT OFFSET, mean of both colours (general)            [predict: level 1]"),
                 ("z_diff","STARTING POINT difference between colours"),
                 ("do_diff","DRIFT OFFSET difference between colours                 [predict: level 2]")]:
    print(f"\n--- {name} ---")
    print(d.groupby("lvl")[col].agg(["mean","sem","size"]).round(4).to_string())
    for lv in sorted(d.lvl.dropna().unique()):
        x=d[d.lvl==lv][col].dropna()
        if len(x)<4: print(f"   {lv:20s} n={len(x)} too small"); continue
        w,p=stats.wilcoxon(x); print(f"   {lv:20s} n={len(x):2d} mean={x.mean():+.4f} vs 0: p={p:.4f}")
    a=d[d.explicitness==1][col].dropna(); b=d[d.explicitness==2][col].dropna(); c=d[d.explicitness==0][col].dropna()
    for l1,l2,n1,n2 in [(a,c,"level1","level0"),(b,c,"level2","level0"),(a,b,"level1","level2")]:
        if len(l1)>=4 and len(l2)>=4:
            u,p=stats.mannwhitneyu(l1,l2); print(f"   {n1} vs {n2}: p={p:.4f}")


    # ---- confound checks: awareness must not be a proxy for medication or subtype ----
    print("\n" + "=" * 78); print("CONFOUND CHECKS"); print("=" * 78)
    s2t = m.drop_duplicates("subject_id").set_index("subject_id")["trem_vs_brady_type"].to_dict()
    q["subtype"] = q.subject_id.map(s2t).fillna("unclassified")
    print(pd.crosstab(q.explicitness, q.treatment, margins=True).to_string())
    print()
    print(pd.crosstab(q.explicitness, q.subtype, margins=True).to_string())
    print("\nlevel 0 vs level 2 WITHIN each medication state (removes the med confound):")
    for t in ["OFF", "ON"]:
        sub = q[q.treatment == t]
        a, b = sub[sub.explicitness == 0].col_bias, sub[sub.explicitness == 2].col_bias
        if len(a) >= 4 and len(b) >= 4:
            _, pv = stats.mannwhitneyu(a, b)
            print(f"  {t}: L0 n={len(a):2d} {a.mean():+.4f}   L2 n={len(b):2d} {b.mean():+.4f}   p={pv:.4f}")
    print("\nWARNING: all level-1 sessions are OFF medication -> level 1 is fully confounded.")
    print("WARNING: ~24 tests run here; nothing survives multiplicity correction. Exploratory.")
