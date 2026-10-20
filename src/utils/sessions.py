"""Decoding raw task sessions into trial tables (compile stage: 0.00, 0.01; read by 0.10).

Every source (the per-session .mat files, the Harvard trial fragments and the Perugini reports)
ends up as one table per session with `SESSION_COLUMNS`, labelled with who ran it and under which
prior. Event codes (columns of the `event` matrix, counting from 1):

    3rd  target: 4000 left, 4001 right
    4th  coherence: 4100 100%, 4101 35%, 4102 13%, 4103 0%
    5th  Glass-pattern colour: 6101 green, 6102 red
    8th  5005 anticipatory saccade, 5006 chose distractor, 5008 / 0 invalid
    9th  5007 failed to respond in time, 5510 correct

Reaction time is the 8th minus the 7th column of the `time` matrix.
"""

import numpy as np
import pandas as pd

SESSION_COLUMNS = [
    "subject_id",
    "group",
    "medication",
    "prior",
    "prior_direction",
    "prior_color",
    "color",
    "coherence",
    "target",
    "is_valid",
    "outcome",
    "choice",
    "reaction_time",
    "session_filename",
]


def determine_choice(row):
    if row["is_valid"]:
        if row["outcome"]:
            return row["target"]
        else:
            return "left" if row["target"] == "right" else "right"
    else:
        return np.nan


def get_prior_condition(df):
    valid_df = df[df["is_valid"]].copy()  # Ensure valid_df is a copy to avoid SettingWithCopyWarning

    # Calculate trial counts and percentages for each condition within valid trials
    condition_counts = valid_df.groupby(["target", "color"]).size().reset_index(name="counts")
    total_counts = condition_counts.groupby("color")["counts"].transform("sum")
    condition_counts["percentage"] = (condition_counts["counts"] / total_counts) * 100

    # Filter conditions meeting the 60% criterion
    conditions_met = condition_counts[(condition_counts["percentage"] > 60)].copy()  # Make a copy to safely modify

    # Prepare the output based on conditions met
    if not conditions_met.empty:
        # Use .loc to modify 'condition' column safely
        conditions_met.loc[:, "condition"] = conditions_met.apply(
            lambda x: "gr" if x["target"] == "right" and x["color"] == "green" else ("gl" if x["target"] == "left" and x["color"] == "green" else ("rr" if x["target"] == "right" and x["color"] == "red" else "rl")), axis=1
        )
        return conditions_met[["condition", "target", "color"]].values.tolist()[0]
    else:
        return ["eq", -1, -1]


def decode_events(event, time):
    """Trial table from a session's `event` and `time` matrices (one row per trial, in order)."""
    df = pd.DataFrame(
        {
            "color": np.select([event[:, 4] == 6101, event[:, 4] == 6102], ["green", "red"], default=None),
            "coherence": np.select([event[:, 3] == 4100, event[:, 3] == 4101, event[:, 3] == 4102, event[:, 3] == 4103], [100, 35, 13, 0], default=np.nan),
            "target": np.select([event[:, 2] == 4000, event[:, 2] == 4001], ["left", "right"], default=None),
        }
    )

    invalid_trials = np.sort(np.where((event[:, 8] == 5007) | (event[:, 7] == 5005) | (event[:, 7] == 0) | (event[:, 7] == 5008))[0])
    df["is_valid"] = True
    df.loc[invalid_trials, "is_valid"] = False

    df["outcome"] = np.nan
    df.loc[np.where(event[:, 8] == 5510)[0], "outcome"] = 1
    df.loc[np.where(event[:, 7] == 5006)[0], "outcome"] = 0

    df["choice"] = df.apply(determine_choice, axis=1)
    df["reaction_time"] = time[:, 7] - time[:, 6]
    return df


def decode_perugini_keypress(xyz):
    """Trial table from a Perugini (2016) key-press `*_TrialByTrial_*.mat` `xyz` matrix (one row per trial).

    Columns: 1 outcome (5510 correct, 5007 no response), 2 coherence, 3 target, 4 colour, 5 response (5006 chose the
    distractor; 0 / 5005 / 5008 invalid), 6 reaction time (ms). Decoded as notebook 0.01 always has.
    """
    data = np.asarray(xyz).astype(int)
    df = pd.DataFrame(
        {
            "color": np.select([data[:, 3] == 6101, data[:, 3] == 6102], ["green", "red"], default=None),
            "coherence": np.select([data[:, 1] == 4100, data[:, 1] == 4101, data[:, 1] == 4102, data[:, 1] == 4103], [100, 35, 13, 0], default=np.nan),
            "target": np.select([data[:, 2] == 4000, data[:, 2] == 4001], ["left", "right"], default=None),
        }
    )
    invalid = (data[:, 0] == 5007) | (data[:, 4] == 0) | (data[:, 4] == 5005) | (data[:, 4] == 5008)
    df["is_valid"] = ~invalid
    df["outcome"] = np.nan
    df.loc[data[:, 0] == 5510, "outcome"] = 1
    df.loc[data[:, 4] == 5006, "outcome"] = 0
    df["choice"] = df.apply(determine_choice, axis=1)
    df["reaction_time"] = data[:, 5]
    return df


def decode_perugini_eye(report):
    """Trial table from a Perugini (2016) eye-movement report (`*_rprt1_*.xlsx`, read with its header row).

    One column per event class: 2-3 target, 4-7 coherence, 8 chose the distractor (5006), 9 correct (5510), 10-11
    colour, 12 reaction time. A trial without an outcome is invalid. Decoded as notebook 0.01 always has.
    """
    data = report.copy()
    data.columns = range(data.shape[1])
    df = pd.DataFrame(
        {
            "color": np.select([data[9] == 6101, data[10] == 6102], ["green", "red"], default=None),
            "coherence": np.select([data[3].isin(["1:4100", 4100]), data[4] == 4101, data[5] == 4102, data[6] == 4103], [100, 35, 13, 0], default=np.nan),
            "target": np.select([data[1] == 4000, data[2] == 4001], ["left", "right"], default=None),
        }
    )
    df["outcome"] = np.nan
    df.loc[(data[7] == 5006).to_numpy(), "outcome"] = 0
    df.loc[(data[8] == 5510).to_numpy(), "outcome"] = 1
    df["is_valid"] = ~df["outcome"].isna()
    df["choice"] = df.apply(determine_choice, axis=1)
    df["reaction_time"] = data[11].to_numpy()
    return df[["color", "coherence", "target", "is_valid", "outcome", "choice", "reaction_time"]]


def combine_fragments(fragments):
    """One session's `event`/`time` from the Harvard trial fragments, in file order.

    Fragments that are not 10-column event matrices (aborted runs) are skipped.
    """
    kept = [f for f in fragments if f["event"].shape[1] == 10]
    return np.concatenate([f["event"] for f in kept]), np.concatenate([f["time"] for f in kept])


def label_session(trials, subject_id, group, medication, session_filename):
    """Add the session-level columns, including the prior inferred from the trials.

    `medication` is "off" / "on" for patients and None for controls, so it is missing in every
    file format rather than a placeholder string.
    """
    trials = trials.copy()
    trials["prior"], trials["prior_direction"], trials["prior_color"] = get_prior_condition(trials)
    trials["subject_id"] = subject_id
    trials["group"] = group
    trials["medication"] = medication
    trials["session_filename"] = session_filename
    return trials[SESSION_COLUMNS]


def session_stem(subject_id, group, medication, prior, source=None):
    """File stem for a compiled session, e.g. `AJL_OFFmeds_gr`, `HC1_healthy_gr`, `AV_asmHC_KP_gr`."""
    if source is not None:
        return f"{subject_id}_asmHC_{source}_{prior}"
    if group == "hc":
        return f"{subject_id}_healthy_{prior}"
    return f"{subject_id}_{medication.upper()}meds_{prior}"


def valid_trials(trials, min_rt):
    """Trials the subject completed correctly enough to analyse: valid, and not faster than `min_rt`."""
    valid = trials[trials["is_valid"] == 1]
    return valid[valid["reaction_time"] >= min_rt]


def easy_accuracy_pct(trials, min_rt):
    """Each session's accuracy (%) on valid 100%-coherence trials, indexed by `session_filename`.

    The quantity the accuracy screen (`catalog.passing_subjects`) is applied to, calculated as the
    preprocessing has always done it. Works on compiled and processed trial tables alike: neither
    changes `coherence` or `outcome`.
    """
    valid = valid_trials(trials, min_rt)
    accuracy = {}
    for session in valid["session_filename"].unique():
        session_data = valid[valid["session_filename"] == session]
        accuracy[session] = np.mean(session_data["outcome"][np.abs(session_data["coherence"]) == 100]) * 100
    return pd.Series(accuracy, name="easy_accuracy_pct", dtype=float)



def prior_design_error(trials, base_rates=(0.75, 0.5)):
    """How far each session's presented targets depart from the intended priors, by `session_filename`.

    Every session should pair one colour whose targets go to the favoured side at `base_rates[0]`
    (75:25) with one colour at `base_rates[1]` (50:50). Trial lists have fixed counts, so a correctly
    run session matches both to within a percentage point. The measure uses every presented trial
    with a colour and a target, valid or not, because the design is what was shown, not what was answered.

    Returns `biased_pct` and `equal_pct` (the more- and less-biased colour's split, in % toward its
    majority side) and `prior_design_error`, the larger of the two departures as a proportion:
    max(| |p_biased - 0.5| - (base_rates[0] - 0.5) |, | |p_equal - 0.5| - |base_rates[1] - 0.5| |).
    The design screen (`catalog.passing_subjects`) is applied to `prior_design_error`.

    In the current raw data correct sessions score at most 0.017. Sessions with a 75:25 prior on both
    colours (HC13, P9 ON, P28 OFF, P29 OFF, P30 ON, P32 ON, P33 ON) score about 0.25, and sessions with
    a weaker prior or an unbalanced equal colour (P3 ON 62.5:37.5, HC7 and P8 OFF 68.75:31.25, P22 ON
    68.75 / 56.25, P33 OFF 56.25 equal) score 0.0625 to 0.125. Works on compiled tables (target
    "left"/"right") and processed tables (target 1 = favoured side) alike.
    """
    shown = trials.dropna(subset=["color", "target"])
    side = shown["target"].eq("right") if shown["target"].dtype == object else shown["target"].astype(float)
    deviation = side.groupby([shown["session_filename"], shown["color"]]).mean().sub(0.5).abs().unstack()
    biased, equal = deviation.max(axis=1), deviation.min(axis=1)
    error = pd.concat([(biased - abs(base_rates[0] - 0.5)).abs(), (equal - abs(base_rates[1] - 0.5)).abs()], axis=1).max(axis=1)
    return pd.DataFrame({"biased_pct": 100 * (0.5 + biased), "equal_pct": 100 * (0.5 + equal), "prior_design_error": error})
