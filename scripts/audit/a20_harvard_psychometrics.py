"""Colour-specific psychometric functions for every Harvard PD patient (P25-P34), OFF and ON, with screening status.

x: signed coherence toward the session's favoured side (%); y: P(choose favoured side).
Points: observed proportion +/- binomial SE; lines: logit_4 fit (src.utils.pmf_utils), as in Figure 1.
Colours follow the manuscript (2.30): prior colour tan, equal colour purple. In two-prior sessions the
"equal" colour actually carried a 75:25 prior toward the opposite side."""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.data_io import catalog
from src.utils.pmf_utils import PsychometricFunction

POS, EQ, INK, MUTED = "#C2B39F", "#8C5A96", "#2F3A45", "#8a8a85"
SUBJECTS = [f"P{i}" for i in range(25, 35)]

trials = catalog.load_trials(min_easy_accuracy=None, prior_design_tolerance=None)
trials = trials[trials.subject_id.isin(SUBJECTS)]
sessions = catalog._load("metadata_sessions").set_index("file_name")
design = catalog._session_design("processed")
keep = catalog.passing_subjects(catalog._session_accuracy("processed"))


def session_status(fname):
    acc, err = sessions.loc[fname, "easy_accuracy_pct"], design.loc[fname, "prior_design_error"]
    notes = []
    if acc < 60:
        notes.append(f"accuracy {acc:.0f}% < 60%")
    if err > 0.05:
        b, e = design.loc[fname, "biased_pct"], design.loc[fname, "equal_pct"]
        notes.append("75:25 on both colours" if e > 70 else f"off-design ({b:.0f}:{100 - b:.0f} / {e:.0f}:{100 - e:.0f})")
    return acc, notes


fig, axes = plt.subplots(5, 4, figsize=(11, 13.5), sharex=True, sharey=True)
xs = np.linspace(-100, 100, 401)
for i, subj in enumerate(SUBJECTS):
    for j, med in enumerate(["off", "on"]):
        ax = axes[i // 2, (i % 2) * 2 + j]
        d = trials[(trials.subject_id == subj) & (trials.medication == med)]
        fname = d.session_filename.iat[0]
        for colour, c, label in [(1, POS, "prior colour"), (0, EQ, "equal colour")]:
            dc = d[d.color == colour]
            g = dc.groupby("signed_coherence").choice.agg(["mean", "size"]).reset_index()
            se = np.sqrt(g["mean"] * (1 - g["mean"]) / g["size"])
            ax.errorbar(g.signed_coherence, g["mean"], yerr=se, fmt="o", ms=4.5, color=c, mec="white", mew=0.6, elinewidth=1, capsize=0, zorder=3, label=label)
            try:
                f = PsychometricFunction(model="logit_4").fit(g.signed_coherence.to_numpy(float), g["mean"].to_numpy(float), g["size"].to_numpy(float))
                ax.plot(xs, f.predict(xs), color=c, lw=2, zorder=2)
            except Exception:
                pass
        ax.axhline(0.5, color="0.85", lw=0.8, zorder=0)
        ax.axvline(0, color="0.85", lw=0.8, zorder=0)
        acc, notes = session_status(fname)
        prior = sessions.loc[fname, "prior_condition"]
        ax.set_title(f"{subj} {med.upper()}  ·  {prior}  ·  {acc:.0f}% at 100%", fontsize=9, color=INK, loc="left")
        if notes:
            ax.text(0.03, 0.95, "\n".join(notes), transform=ax.transAxes, fontsize=8, va="top", color="#B5541D")
        ax.set_xticks([-100, -35, 0, 35, 100])
        ax.tick_params(labelsize=8, colors=MUTED)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("0.75")
    # patient-level verdict, on the OFF panel's row label
    verdict = "accepted" if subj in keep else "rejected"
    axes[i // 2, (i % 2) * 2].annotate(verdict, xy=(0, 1.16), xycoords="axes fraction", fontsize=9, fontweight="bold",
                                       color=INK if verdict == "accepted" else "#B5541D")
for ax in axes[-1]:
    ax.set_xlabel("Coherence toward favoured side (%)", fontsize=8.5, color=INK)
for ax in axes[:, 0]:
    ax.set_ylabel("P(favoured choice)", fontsize=8.5, color=INK)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper right", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.99, 1.0))
fig.suptitle("Harvard PD patients: colour-specific psychometric functions, OFF and ON", x=0.01, ha="left", fontsize=12, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=2.2)
catalog.save_figure(fig, "audit/harvard_pd_psychometrics")
