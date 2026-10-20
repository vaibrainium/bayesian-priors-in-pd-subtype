"""Per-subject profiles from the compiled sessions (compiled/<site>/, notebooks 0.02-0.05).

For every subject in the study (the four sites; not the Perugini controls):

1. practice: accuracy if the practice was at 100% coherence only, a psychometric function if it included lower
   coherences (per session);
2. / 3. ON and OFF medication (and UCLA's explicit sessions): the psychometric function of each colour, for the whole
   session when all its trial blocks implied the same prior, otherwise separately for each prior condition, labelled
   with the condition and the blocks it covers;
4. experiment site;
5. whether the OFF session came first: as recorded in the metadata, and as the recordings date it (Harvard and
   Stanford by their block timestamps, Case Western by the clock where OFF and ON were run in one sitting; UCLA's
   recording dates are lost);
6. tremor- vs bradykinesia-dominant subtype from the OFF-medication MDS-UPDRS exam, with the rule and thresholds of
   notebook 0.30 (`src.utils.classification`), for Stanford and Harvard patients only (the sites with full UPDRS
   exams); never from folder names.

Trials are the valid ones with a reaction time of at least `BEHAVIOR.min_rt` seconds (compiled reaction times are in
ms). Psychometric functions plot P(choose right) against signed coherence (positive = target right) and are fitted
with the 4-parameter logistic of `src.utils.pmf_utils`.

Writes to outputs/subject_profiles/: profiles.csv (one row per subject), psychometrics.csv (one row per subject x
session x condition x colour, with the fitted parameters) and a figure per subject (<subject>.png / .pdf).

    .venv/bin/python -m scripts.pipeline.subject_profiles [SUBJECT ...]
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import main_config
from src.data_io import catalog
from src.utils.classification import subtype_label, tremor_bradykinesia_ratio
from src.utils.pmf_utils import fit_psychometric_function

OUT = "subject_profiles"
SITE_NAMES = {"harvard": "Harvard", "stanford": "Stanford", "case_western": "Case Western", "ucla": "UCLA"}
SUBTYPED_SITES = ("harvard", "stanford")  # full MDS-UPDRS part III
MIN_RT_MS = main_config.BEHAVIOR.min_rt * 1000
UPDRS = main_config.MDS_UPDRS
COLOUR_STYLE = {  # stimulus colours, told apart by marker and line as well as hue
    "green": {"color": "#008300", "marker": "o", "label": "green"},
    "red": {"color": "#e34948", "marker": "^", "label": "red"},
}
INK, MUTED = "#0b0b0b", "#52514e"


# --------------------------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------------------------- #


def analysable(trials):
    """Valid trials at or above the reaction-time floor, with signed coherence and a 0/1 rightward choice."""
    valid = trials[trials.is_valid & (trials.reaction_time >= MIN_RT_MS)].dropna(subset=["color", "target", "choice", "coherence"]).copy()
    valid["signed_coherence"] = np.where(valid.target == "right", valid.coherence, -valid.coherence)
    valid["chose_right"] = (valid.choice == "right").astype(float)
    return valid


def psychometric(trials):
    """Points (signed coherence, P(right), n) and, when there are at least three levels, a logit_4 fit."""
    points = trials.groupby("signed_coherence").chose_right.agg(["mean", "size"]).reset_index()
    fit = None
    if len(points) >= 3:
        try:
            fit = fit_psychometric_function(points.signed_coherence.to_numpy(float), points["mean"].to_numpy(float), points["size"].to_numpy(float), "logit_4")
        except (RuntimeError, ValueError):
            fit = None
    return points, fit


def session_order(blocks):
    """Recorded start of each session: earliest block timestamp (Harvard, Stanford)."""
    used = blocks.dropna(subset=["filename_time"])
    return used.groupby("session_id").filename_time.min()


def recorded_off_first(subject, site, starts, summaries):
    """Whether the OFF session came first as the recordings show, and how that is known."""
    off, on = f"{subject}_off", f"{subject}_on"
    if site in ("harvard", "stanford"):
        if off in starts and on in starts:
            return bool(starts[off] < starts[on]), f"session dates ({starts[off]:%Y-%m-%d} OFF, {starts[on]:%Y-%m-%d} ON)"
        return None, "one session missing"
    if site == "case_western":
        row = summaries["case_western"].set_index("session_id").loc[off]
        if row.same_sitting:
            return row.order_by_clock.startswith("OFF"), f"clock, one sitting ({row.awake_gap_min:.0f} min apart)"
        return None, "clock, separate sittings: order unknown"
    return None, "recording dates lost (batch re-saved files)"


def metadata_off_first(metadata, subject):
    values = metadata.loc[metadata.subject_id == subject, "off_med_first"].dropna().unique()
    if len(values) == 0:
        return None, "not recorded" if (metadata.subject_id == subject).any() else "no metadata"
    if len(values) > 1:
        return None, "conflicting rows"
    return bool(values[0]), "metadata"


def subtype(metadata, subject, site, group):
    if group != "pd":
        return {"subtype": None, "subtype_note": "control"}
    if site not in SUBTYPED_SITES:
        return {"subtype": None, "subtype_note": f"no full UPDRS at {SITE_NAMES[site]}"}
    off_exam = metadata[(metadata.subject_id == subject) & (metadata.treatment == "OFF")]
    if off_exam.empty:
        return {"subtype": None, "subtype_note": "no OFF UPDRS exam in the metadata"}
    tremor, bradykinesia, ratio = tremor_bradykinesia_ratio(off_exam.iloc[0], UPDRS.TREMOR_ITEMS, UPDRS.BRADYKINESIA_ITEMS)
    label = subtype_label(ratio, dict(UPDRS.TREM_VS_BRADY))
    return {"subtype": label, "subtype_note": "OFF exam" if label else "UPDRS items missing", "tremor_score": tremor,
            "bradykinesia_score": bradykinesia, "tremor_bradykinesia_ratio": ratio}


# --------------------------------------------------------------------------------------------- #
# One subject
# --------------------------------------------------------------------------------------------- #


def session_label(session_id, medication):
    if medication is None or pd.isna(medication):
        return "session"
    return f"{medication.upper()}" + (" explicit" if session_id.endswith("_explicit") else "")


def conditions(session):
    """Practice and one group per prior condition of the trial blocks, in block order."""
    groups = []
    practice = session[session.phase == "practice"]
    if len(practice):
        groups.append(("practice", "practice", practice))
    trial = session[session.phase == "trial"]
    for prior in dict.fromkeys(trial.sort_values("block").block_prior):  # each prior once, in block order
        groups.append(("trial", prior, trial[trial.block_prior == prior]))
    return groups


def colour_role(condition, colour):
    """`prior` for a colour carrying the condition's prior, `equal` for the 50:50 colour, "" without a prior."""
    if condition in ("none", "practice"):
        return ""
    return "prior" if colour[0] in {p[0] for p in condition.split("+")} else "equal"


def describe_prior(prior):
    """`gr+rl` -> "green 75% right, red 75% left"."""
    if prior in ("none", "practice"):
        return "no prior"
    return ", ".join(f"{'green' if p[0] == 'g' else 'red'} 75% {'right' if p[1] == 'r' else 'left'}" for p in prior.split("+"))


def subject_rows(subject, trials):
    """Psychometric rows (one per session x condition x colour) for one subject."""
    rows = []
    for session_id, session in trials.groupby("session_id", sort=False):
        medication = session.medication.iloc[0]
        for phase, condition, part in conditions(session):
            valid = analysable(part)
            blocks = sorted(part.block.unique())
            base = {"subject_id": subject, "session_id": session_id, "session": session_label(session_id, medication), "phase": phase,
                    "condition": condition, "blocks": ",".join(map(str, blocks)), "coherences": ",".join(str(int(c)) for c in sorted(part.coherence.dropna().unique()))}
            only_easy = set(part.coherence.dropna().unique()) <= {100}
            for colour in ("green", "red"):
                c = valid[valid.color == colour]
                easy = c[c.coherence == 100]
                row = {**base, "color": colour, "n_trials": len(c), "accuracy_100_pct": 100 * easy.outcome.mean() if len(easy) else np.nan,
                       "color_role": colour_role(condition, colour)}
                if not only_easy and len(c):
                    _, fit = psychometric(c)
                    if fit is not None:
                        row.update({f"fit_{k}": v for k, v in fit.coefs_.items()})
                rows.append(row)
    return rows


def draw_psychometric(ax, valid, title, colour_roles):
    for colour, style in COLOUR_STYLE.items():
        c = valid[valid.color == colour]
        if c.empty:
            continue
        points, fit = psychometric(c)
        role = colour_roles.get(colour, "")  # "" when there is no prior
        ax.plot(points.signed_coherence, points["mean"], ls="none", marker=style["marker"], ms=5, color=style["color"],
                mec="white", mew=0.8, label=f"{style['label']}{' (' + role + ')' if role else ''}", zorder=3)
        if fit is not None:
            x = np.linspace(-100, 100, 201)
            ax.plot(x, fit.predict(x), color=style["color"], lw=1.6, ls="-" if role != "equal" else "--", zorder=2)
    ax.axhline(0.5, color="0.85", lw=0.8, zorder=0)
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    ax.set_xlim(-108, 108)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks([-100, -35, 0, 35, 100])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelsize=7, colors=MUTED, length=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("0.7")
    ax.set_title(title, fontsize=8, color=INK, loc="left")
    ax.set_ylabel("P(choose right)", fontsize=7, color=MUTED)
    ax.set_xlabel("signed coherence (%)", fontsize=7, color=MUTED)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left", handletextpad=0.2)


def draw_accuracy(ax, valid, title):
    """100%-coherence-only practice: accuracy per colour as text and bars."""
    accuracy = valid.groupby("color").outcome.agg(["mean", "size"]).reindex(["green", "red"])
    for i, (colour, row) in enumerate(accuracy.iterrows()):
        if pd.isna(row["mean"]):
            continue
        ax.bar(i, row["mean"], color=COLOUR_STYLE[colour]["color"], width=0.55, hatch=None if colour == "green" else "///", edgecolor="white")
        ax.text(i, row["mean"] + 0.03, f"{100 * row['mean']:.0f}%\nn={row['size']:.0f}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks([0, 1], ["green", "red"], fontsize=7)
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelsize=7, colors=MUTED, length=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("0.7")
    ax.set_title(title, fontsize=8, color=INK, loc="left")
    ax.set_ylabel("accuracy (100% coherence)", fontsize=7, color=MUTED)


def subject_figure(subject, trials, profile):
    sessions = list(dict.fromkeys(trials.session_id))
    layouts = {s: conditions(trials[trials.session_id == s]) for s in sessions}
    ncols = max(len(g) for g in layouts.values())
    fig, axes = plt.subplots(len(sessions), ncols, figsize=(2.6 * ncols + 0.4, 2.45 * len(sessions) + 0.4), squeeze=False, layout="constrained")
    for r, session_id in enumerate(sessions):
        session = trials[trials.session_id == session_id]
        label = session_label(session_id, session.medication.iloc[0])
        for c, (phase, condition, part) in enumerate(layouts[session_id]):
            ax = axes[r, c]
            valid = analysable(part)
            blocks = sorted(part.block.unique())
            block_text = f"block{'s' if len(blocks) > 1 else ''} {','.join(map(str, blocks))}"
            if phase == "practice":
                if set(part.coherence.dropna().unique()) <= {100}:
                    draw_accuracy(ax, valid, f"{label} practice\n100% coherence only · {len(valid)} trials")
                else:
                    draw_psychometric(ax, valid, f"{label} practice\nall coherences, no prior · {len(valid)} trials", {})
            else:
                roles = {colour: colour_role(condition, colour) for colour in COLOUR_STYLE}
                draw_psychometric(ax, valid, f"{label} · {condition} ({describe_prior(condition)})\n{block_text} · {len(valid)} trials", roles)
        for c in range(len(layouts[session_id]), ncols):
            axes[r, c].axis("off")
    fig.suptitle(header(subject, profile), fontsize=9, x=0.01, ha="left", color=INK)
    return fig


def yes_no(value):
    return "unknown" if value is None or pd.isna(value) else ("yes" if value else "no")


def header(subject, p):
    parts = [f"{subject} · {p['site']} · {'patient' if p['group'] == 'pd' else 'control'}"]
    if p["group"] == "pd":
        ratio = p.get("tremor_bradykinesia_ratio")
        if p["subtype"]:
            parts.append(f"subtype: {p['subtype']} (tremor/bradykinesia {ratio:.2f}, OFF exam)")
        else:
            parts.append(f"subtype: not classified ({p['subtype_note']})")
        parts.append(f"OFF first: metadata {yes_no(p['off_first_metadata'])}, recorded {yes_no(p['off_first_recorded'])}")
    return "   ".join(parts)


# --------------------------------------------------------------------------------------------- #
# All subjects
# --------------------------------------------------------------------------------------------- #


def main(only=None):
    trials = catalog.load_block_sessions()
    metadata = catalog.load("raw_session_metadata")
    summaries = {site: catalog.load(f"compiled_{site}_session_summary") for site in catalog.BLOCK_SITES}
    starts = pd.concat([session_order(catalog.load(f"compiled_{site}_blocks")) for site in ("harvard", "stanford")])

    profiles, psych_rows = [], []
    for subject, subject_trials in trials.groupby("subject_id", sort=False):
        if only and subject not in only:
            continue
        site, group = subject_trials.site.iloc[0], subject_trials.group.iloc[0]
        profile = {"subject_id": subject, "site": SITE_NAMES[site], "group": group,
                   "sessions": ", ".join(dict.fromkeys(subject_trials.session_id))}
        if group == "pd":
            profile["off_first_metadata"], profile["off_first_metadata_note"] = metadata_off_first(metadata, subject)
            profile["off_first_recorded"], profile["off_first_recorded_note"] = recorded_off_first(subject, site, starts, summaries)
            known = [v for v in (profile["off_first_metadata"], profile["off_first_recorded"]) if v is not None]
            profile["off_first_agree"] = None if len(known) < 2 else known[0] == known[1]
        profile.update(subtype(metadata, subject, site, group))
        rows = subject_rows(subject, subject_trials)
        psych_rows += rows
        table = pd.DataFrame(rows)
        for session in table.session.unique():
            s = table[table.session == session]
            key = session.lower().replace(" ", "_")
            practice = s[s.phase == "practice"]
            profile[f"{key}_practice"] = ("none" if practice.empty else
                                          ("accuracy " + ", ".join(f"{r.color} {r.accuracy_100_pct:.0f}%" for r in practice.itertuples())
                                           if practice.coherences.iloc[0] == "100" else "psychometric (all coherences)"))
            trial = s[s.phase == "trial"].drop_duplicates("condition")
            profile[f"{key}_conditions"] = " | ".join(f"{r.condition} (blocks {r.blocks})" for r in trial.itertuples())
        profiles.append(profile)

        fig = subject_figure(subject, subject_trials, profile)
        catalog.save_figure(fig, f"{OUT}/{subject}")
        plt.close(fig)

    catalog.save_table(pd.DataFrame(profiles), f"{OUT}/profiles.csv", index=False)
    catalog.save_table(pd.DataFrame(psych_rows), f"{OUT}/psychometrics.csv", index=False)


if __name__ == "__main__":
    main(set(sys.argv[1:]) or None)
