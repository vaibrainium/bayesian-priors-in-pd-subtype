"""
Preprocess aggregated behavioral data: filter, recode, fit psychometrics.

Reads:  compiled/aggregate_all_data.csv
        compiled/aggregate_all_metadata.csv

Writes: processed/processed_metadata_all_data_accu_60.csv
        processed/processed_all_data_accu_60_all.csv
        processed/processed_all_data_accu_60_filtered.csv
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from config import dir_config, main_config
from src.utils import pmf_utils


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fill_prior_from_aggregate(session_metadata: pd.DataFrame, aggregate_data: pd.DataFrame) -> pd.DataFrame:
    """Populate prior_condition and file_name columns in metadata from aggregate data."""
    for subject_id, medication in sorted(
        zip(session_metadata["subject_id"], session_metadata["treatment"]),
        key=lambda x: x[0],
    ):
        if pd.isna(medication):
            session_data = aggregate_data[
                (aggregate_data["subject_id"] == subject_id) & aggregate_data["medication"].isna()
            ]
            idx = session_metadata[
                (session_metadata["subject_id"] == subject_id) & session_metadata["treatment"].isna()
            ].index
        else:
            session_data = aggregate_data[
                (aggregate_data["subject_id"] == subject_id)
                & (aggregate_data["medication"] == medication.lower())
            ]
            idx = session_metadata[
                (session_metadata["subject_id"] == subject_id)
                & (session_metadata["treatment"] == medication)
            ].index

        prior = np.unique(session_data["prior"])
        prior = prior[prior != "eq"]
        if len(prior) > 0:
            session_metadata.loc[idx, "prior_condition"] = prior[0]
        if len(np.unique(session_data["session_filename"])) > 0:
            session_metadata.loc[idx, "file_name"] = np.unique(session_data["session_filename"])[0]

    return session_metadata


def replace_missing_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Replace common missing-value sentinel strings with NaN."""
    missing_strings = ["", "NA", "NaN", "na", "n/a"]
    for ms in missing_strings:
        for variant in {ms, ms.upper(), ms.lower(), ms.capitalize()}:
            df.replace(variant, np.nan, inplace=True)
    return df


def data_preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    """Remove rejected subjects, drop equal-prior sessions, recode directions/colors, add signed_coherence."""
    data = data[~data["subject_id"].isin(main_config.moca_rejection)]
    data = data[data["prior"] != "eq"]
    data["prior_color"].loc[data["prior"] == "eq"] = np.nan
    data["prior_direction"].loc[data["prior"] == "eq"] = np.nan

    left_prior_idx = data["prior_direction"] == "left"
    right_prior_idx = data["prior_direction"] == "right"
    green_prior_idx = data["prior_color"] == "green"
    red_prior_idx = data["prior_color"] == "red"

    data.loc[left_prior_idx, "target"] = data.loc[left_prior_idx, "target"].map({"left": 1, "right": 0})
    data.loc[left_prior_idx, "choice"] = data.loc[left_prior_idx, "choice"].map({"left": 1, "right": 0})
    data.loc[right_prior_idx, "target"] = data.loc[right_prior_idx, "target"].map({"left": 0, "right": 1})
    data.loc[right_prior_idx, "choice"] = data.loc[right_prior_idx, "choice"].map({"left": 0, "right": 1})

    data.loc[green_prior_idx, "color"] = data.loc[green_prior_idx, "color"].map({"green": 1, "red": 0})
    data.loc[red_prior_idx, "color"] = data.loc[red_prior_idx, "color"].map({"green": 0, "red": 1})

    data["signed_coherence"] = data["coherence"] * (2 * data["target"] - 1)

    return data


def apply_data_slice(aggregate_data: pd.DataFrame, valid_data: pd.DataFrame,
                     session_metadata: pd.DataFrame, slicing: int):
    """Optionally slice each session to first or second half."""
    if slicing == 0:
        return aggregate_data, valid_data

    temp_agg, temp_valid = [], []
    for subject_id, medication in sorted(
        zip(session_metadata["subject_id"], session_metadata["treatment"]),
        key=lambda x: x[0],
    ):
        if pd.isna(medication):
            agg_sel = aggregate_data[
                (aggregate_data["subject_id"] == subject_id) & aggregate_data["medication"].isna()
            ]
            val_sel = valid_data[
                (valid_data["subject_id"] == subject_id) & valid_data["medication"].isna()
            ]
        else:
            agg_sel = aggregate_data[
                (aggregate_data["subject_id"] == subject_id)
                & (aggregate_data["medication"] == medication.lower())
            ]
            val_sel = valid_data[
                (valid_data["subject_id"] == subject_id)
                & (valid_data["medication"] == medication.lower())
            ]

        if agg_sel.empty or val_sel.empty:
            continue

        half = len(agg_sel) // 2
        if slicing == 1:
            temp_agg.append(agg_sel.iloc[:half])
            temp_valid.append(val_sel.iloc[:half])
        elif slicing == 2:
            temp_agg.append(agg_sel.iloc[half:])
            temp_valid.append(val_sel.iloc[half:])
        else:
            temp_agg.append(agg_sel)
            temp_valid.append(val_sel)

    return (
        pd.concat(temp_agg, ignore_index=True),
        pd.concat(temp_valid, ignore_index=True),
    )


def reject_low_accuracy_sessions(valid_data: pd.DataFrame, aggregate_data: pd.DataFrame,
                                  min_easy_accuracy: float):
    """Drop sessions where accuracy on 100% coherence trials falls below threshold."""
    reject_sessions = []
    for session in valid_data["session_filename"].unique():
        session_data = valid_data[valid_data["session_filename"] == session]
        easy_trials = session_data[np.abs(session_data["coherence"]) == 100]
        accuracy = np.mean(easy_trials["outcome"]) * 100
        if accuracy < min_easy_accuracy * 100:
            reject_sessions.append(session)
            print(f"  Rejected {session}: accuracy={accuracy:.2f}%, n={len(session_data)}")

    aggregate_data = aggregate_data[~aggregate_data["session_filename"].isin(reject_sessions)]
    valid_data = valid_data[~valid_data["session_filename"].isin(reject_sessions)]
    return aggregate_data, valid_data


def filter_complete_subjects(valid_data: pd.DataFrame):
    """Keep only subjects with complete sessions (HC: any; PD: both ON and OFF)."""
    subject_sessions = valid_data.groupby("subject_id")["medication"].unique()
    valid_subjects, pd_subjects = [], []

    for subject, sessions in subject_sessions.items():
        sessions_clean = [s for s in sessions if pd.notna(s)]
        if len(sessions_clean) == 0:
            valid_subjects.append(subject)
        elif set(sessions_clean) == {"on", "off"}:
            valid_subjects.append(subject)
            pd_subjects.append(subject)
        else:
            print(f"  Dropped {subject}: sessions={sessions_clean}")

    hc_subjects = [s for s in valid_subjects if s not in pd_subjects]
    return valid_subjects, pd_subjects, hc_subjects


def fit_psychometrics(session_metadata: pd.DataFrame, valid_data: pd.DataFrame,
                      pd_subjects: list, hc_subjects: list, psych_model_type: str) -> pd.DataFrame:
    """Add psychometric parameters (positive & equal prior) to session_metadata."""
    for col in ["positive_bias", "positive_psych_bias", "positive_psych_alpha",
                "positive_psych_beta", "positive_psych_lapse", "positive_psych_guess",
                "equal_bias", "equal_psych_bias", "equal_psych_alpha",
                "equal_psych_beta", "equal_psych_lapse", "equal_psych_guess"]:
        session_metadata[col] = np.nan

    pd_sessions = valid_data[valid_data["medication"].notna()].groupby("subject_id")["medication"].unique()

    for subject, sessions in pd_sessions.items():
        for session in sessions:
            session_data = valid_data[
                (valid_data["subject_id"] == subject) & (valid_data["medication"] == session)
            ]
            idx = session_metadata[
                (session_metadata["subject_id"] == subject)
                & (session_metadata["treatment"] == session.upper())
            ].index

            _, pos_psych, pos_model, _, _ = pmf_utils.get_psychometric_data(
                data=session_data[session_data["color"] == 1], model_type=psych_model_type
            )
            _, eq_psych, eq_model, _, _ = pmf_utils.get_psychometric_data(
                data=session_data[session_data["color"] == 0], model_type=psych_model_type
            )

            session_metadata.loc[idx, "positive_bias"] = pos_psych[3]
            session_metadata.loc[idx, "positive_psych_bias"] = pos_model.predict(0)
            session_metadata.loc[idx, "positive_psych_alpha"] = pos_model.coefs_["mean"]
            session_metadata.loc[idx, "positive_psych_beta"] = pos_model.coefs_["var"]
            session_metadata.loc[idx, "positive_psych_lapse"] = pos_model.coefs_["lapse_rate"]
            session_metadata.loc[idx, "positive_psych_guess"] = pos_model.coefs_["guess_rate"]

            session_metadata.loc[idx, "equal_bias"] = eq_psych[3]
            session_metadata.loc[idx, "equal_psych_bias"] = eq_model.predict(0)
            session_metadata.loc[idx, "equal_psych_alpha"] = eq_model.coefs_["mean"]
            session_metadata.loc[idx, "equal_psych_beta"] = eq_model.coefs_["var"]
            session_metadata.loc[idx, "equal_psych_lapse"] = eq_model.coefs_["lapse_rate"]
            session_metadata.loc[idx, "equal_psych_guess"] = eq_model.coefs_["guess_rate"]

    for subject in hc_subjects:
        session_data = valid_data[valid_data["subject_id"] == subject]
        idx = session_metadata[session_metadata["subject_id"] == subject].index

        _, pos_psych, pos_model, _, _ = pmf_utils.get_psychometric_data(
            data=session_data[session_data["color"] == 1], model_type=psych_model_type
        )
        _, eq_psych, eq_model, _, _ = pmf_utils.get_psychometric_data(
            data=session_data[session_data["color"] == 0], model_type=psych_model_type
        )

        session_metadata.loc[idx, "positive_bias"] = pos_psych[3]
        session_metadata.loc[idx, "positive_psych_bias"] = pos_model.predict(0)
        session_metadata.loc[idx, "positive_psych_alpha"] = pos_model.coefs_["mean"]
        session_metadata.loc[idx, "positive_psych_beta"] = pos_model.coefs_["var"]
        session_metadata.loc[idx, "positive_psych_lapse"] = pos_model.coefs_["lapse_rate"]
        session_metadata.loc[idx, "positive_psych_guess"] = pos_model.coefs_["guess_rate"]

        session_metadata.loc[idx, "equal_bias"] = eq_psych[3]
        session_metadata.loc[idx, "equal_psych_bias"] = eq_model.predict(0)
        session_metadata.loc[idx, "equal_psych_alpha"] = eq_model.coefs_["mean"]
        session_metadata.loc[idx, "equal_psych_beta"] = eq_model.coefs_["var"]
        session_metadata.loc[idx, "equal_psych_lapse"] = eq_model.coefs_["lapse_rate"]
        session_metadata.loc[idx, "equal_psych_guess"] = eq_model.coefs_["guess_rate"]

    return session_metadata


def add_session_ids(session_metadata: pd.DataFrame, valid_data: pd.DataFrame,
                    aggregate_data: pd.DataFrame):
    """Add session_id column: subject_id_TREATMENT for PD, subject_id for HC."""
    session_metadata["session_id"] = session_metadata[["subject_id", "treatment"]].apply(
        lambda x: "_".join(x.astype(str).str.upper()), axis=1
    )
    valid_data["session_id"] = valid_data[["subject_id", "medication"]].apply(
        lambda x: "_".join(x.astype(str).str.upper()), axis=1
    )
    aggregate_data["session_id"] = aggregate_data[["subject_id", "medication"]].apply(
        lambda x: "_".join(x.astype(str).str.upper()), axis=1
    )

    session_metadata.loc[session_metadata["is_pd"] == 0, "session_id"] = (
        session_metadata.loc[session_metadata["is_pd"] == 0, "subject_id"]
    )
    valid_data.loc[valid_data["group"] == "hc", "session_id"] = (
        valid_data.loc[valid_data["group"] == "hc", "subject_id"]
    )
    aggregate_data.loc[aggregate_data["group"] == "hc", "session_id"] = (
        aggregate_data.loc[aggregate_data["group"] == "hc", "subject_id"]
    )

    return session_metadata, valid_data, aggregate_data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(output_dir: Path | None = None):
    compiled_dir = Path(dir_config.data.compiled)
    processed_dir = Path(dir_config.data.processed)
    out_dir = Path(output_dir) if output_dir is not None else processed_dir

    psych_model_type = main_config.BEHAVIOR.psych_model
    min_rt = main_config.BEHAVIOR.min_rt
    min_easy_accuracy = main_config.BEHAVIOR.min_easy_accuracy
    slicing = main_config.BEHAVIOR.data_slice

    # Load
    aggregate_data = pd.read_csv(compiled_dir / "aggregate_all_data.csv", index_col=None)
    session_metadata = pd.read_csv(compiled_dir / "aggregate_all_metadata.csv", encoding="latin1", index_col=None)

    # Fill prior info in metadata from aggregate data
    session_metadata = fill_prior_from_aggregate(session_metadata, aggregate_data)

    # Normalise missing value sentinels
    session_metadata = replace_missing_strings(session_metadata)

    # Filter valid trials
    valid_data = aggregate_data[aggregate_data["is_valid"] == 1].copy()
    valid_data = valid_data[valid_data["reaction_time"] >= min_rt]

    # Recode directions/colors, add signed_coherence
    valid_data = data_preprocessing(valid_data)
    aggregate_data = data_preprocessing(aggregate_data)

    # Optional session slicing
    aggregate_data, valid_data = apply_data_slice(aggregate_data, valid_data, session_metadata, slicing)

    # Reject low-accuracy sessions
    print("Rejecting low-accuracy sessions:")
    aggregate_data, valid_data = reject_low_accuracy_sessions(
        valid_data, aggregate_data, min_easy_accuracy
    )

    # Keep only subjects with complete sessions
    valid_subjects, pd_subjects, hc_subjects = filter_complete_subjects(valid_data)
    print(f"Valid PD: {len(pd_subjects)}, HC: {len(hc_subjects)}, total: {len(valid_subjects)}")

    aggregate_data = aggregate_data[aggregate_data["subject_id"].isin(valid_subjects)]
    valid_data = valid_data[valid_data["subject_id"].isin(valid_subjects)]
    session_metadata = session_metadata[session_metadata["subject_id"].isin(valid_subjects)]

    # Fit psychometric functions
    session_metadata = fit_psychometrics(
        session_metadata, valid_data, pd_subjects, hc_subjects, psych_model_type
    )

    # Add session IDs
    session_metadata, valid_data, aggregate_data = add_session_ids(
        session_metadata, valid_data, aggregate_data
    )

    # Final subject filter on metadata
    session_metadata = session_metadata[session_metadata["subject_id"].isin(valid_subjects)]

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    session_metadata.to_csv(out_dir / "processed_metadata_all_data_accu_60.csv", index=False)
    aggregate_data.to_csv(out_dir / "processed_all_data_accu_60_all.csv", index=False)
    valid_data.to_csv(out_dir / "processed_all_data_accu_60_filtered.csv", index=False)

    print(f"Saved metadata      → {out_dir / 'processed_metadata_all_data_accu_60.csv'} ({len(session_metadata)} rows)")
    print(f"Saved all data      → {out_dir / 'processed_all_data_accu_60_all.csv'} ({len(aggregate_data):,} rows)")
    print(f"Saved filtered data → {out_dir / 'processed_all_data_accu_60_filtered.csv'} ({len(valid_data):,} rows)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override output directory (default: processed_dir from dir-config.yaml)")
    args = parser.parse_args()
    main(output_dir=args.output_dir)
