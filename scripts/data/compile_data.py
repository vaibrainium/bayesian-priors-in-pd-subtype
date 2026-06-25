"""
Compile raw behavioral data from all subjects into aggregate CSVs.

Reads:  raw/session_metadata_detailed_all_data.csv
        compiled/*.mat  (HC/PD subjects — hdf5storage format)
        compiled/*.parquet  (asmHC subjects)

Writes: compiled/aggregate_all_data.csv
        compiled/aggregate_all_metadata.csv
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import hdf5storage
from pathlib import Path
from config import dir_config, main_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def determine_choice(row):
    if row["is_valid"]:
        if row["outcome"]:
            return row["target"]
        else:
            return "left" if row["target"] == "right" else "right"
    else:
        return np.nan


def get_prior_condition(df):
    valid_df = df[df["is_valid"]].copy()

    condition_counts = valid_df.groupby(["target", "color"]).size().reset_index(name="counts")
    total_counts = condition_counts.groupby("color")["counts"].transform("sum")
    condition_counts["percentage"] = (condition_counts["counts"] / total_counts) * 100

    conditions_met = condition_counts[(condition_counts["percentage"] > 60)].copy()

    if not conditions_met.empty:
        conditions_met.loc[:, "condition"] = conditions_met.apply(
            lambda x: "gr" if x["target"] == "right" and x["color"] == "green"
            else ("gl" if x["target"] == "left" and x["color"] == "green"
                  else ("rr" if x["target"] == "right" and x["color"] == "red"
                        else "rl")),
            axis=1,
        )
        return conditions_met[["condition", "target", "color"]].values.tolist()[0]
    else:
        return ["eq", -1, -1]


def load_hc_pd_session(session_file: Path) -> pd.DataFrame:
    """Load a single HC/PD .mat session file and return a trial DataFrame."""
    session_data = hdf5storage.loadmat(str(session_file))

    df = pd.DataFrame(
        {
            "color": np.select(
                [session_data["event"][:, 4] == 6101, session_data["event"][:, 4] == 6102],
                ["green", "red"],
                default=None,
            ),
            "coherence": np.select(
                [
                    session_data["event"][:, 3] == 4100,
                    session_data["event"][:, 3] == 4101,
                    session_data["event"][:, 3] == 4102,
                    session_data["event"][:, 3] == 4103,
                ],
                [100, 35, 13, 0],
                default=np.nan,
            ),
            "target": np.select(
                [session_data["event"][:, 2] == 4000, session_data["event"][:, 2] == 4001],
                ["left", "right"],
                default=None,
            ),
        }
    )

    invalid_idx = np.sort(
        np.where(
            (session_data["event"][:, 8] == 5007)
            | (session_data["event"][:, 7] == 5005)
            | (session_data["event"][:, 7] == 0)
            | (session_data["event"][:, 7] == 5008)
        )[0]
    )
    df["is_valid"] = True
    df.loc[invalid_idx, "is_valid"] = False

    df["outcome"] = np.nan
    df.loc[np.where(session_data["event"][:, 8] == 5510)[0], "outcome"] = 1
    df.loc[np.where(session_data["event"][:, 7] == 5006)[0], "outcome"] = 0

    df["choice"] = df.apply(determine_choice, axis=1)
    df["reaction_time"] = session_data["time"][:, 7] - session_data["time"][:, 6]
    df["prior"], df["prior_direction"], df["prior_color"] = get_prior_condition(df)

    df["subject_id"] = session_file.name.split("_")[0]
    if df["subject_id"].iloc[0].startswith("HC"):
        df["group"] = "hc"
        df["medication"] = "None"
    else:
        df["group"] = "pd"
        df["medication"] = session_file.name.split("_")[-2][:-4].lower()

    df["session_filename"] = session_file.name
    return df


# ---------------------------------------------------------------------------
# Site assignment
# ---------------------------------------------------------------------------

UCLA = ["CG", "COH", "MBY", "DP", "FUR", "LBR", "MAR", "SMI", "PAM", "RW", "BBK", "BER", "DCAM", "ALE", "DMO", "AJL", "SKU"]
UCLA_ASMHC_EM = ["CB1", "JW", "KR", "MEG"]
UCLA_ASMHC_KP = ["AV", "BC", "BF", "EM", "ES", "GF", "GP", "JA", "MRM", "SY"]
CASE_WESTERN = ["RBA", "RDE", "SGA", "LHO", "RSH", "RZA", "SNO"]


def assign_site(subject_id: str, stanford_ids: set, harvard_ids: set, harvard_hc_ids: set) -> str:
    if subject_id in UCLA or subject_id in UCLA_ASMHC_EM or subject_id in UCLA_ASMHC_KP:
        return "UCLA"
    elif subject_id in CASE_WESTERN:
        return "Case_Western"
    elif subject_id in stanford_ids:
        return "Stanford"
    elif subject_id in harvard_ids or subject_id in harvard_hc_ids:
        return "Harvard"
    else:
        return "Unknown"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(output_dir: Path | None = None):
    raw_dir = Path(dir_config.data.raw)
    compiled_dir = Path(dir_config.data.compiled)
    out_dir = Path(output_dir) if output_dir is not None else compiled_dir
    include_eye_tracking = main_config.INCLUDE_EYE_TRACKING_SESSIONS

    # Load metadata
    subject_metadata = pd.read_csv(raw_dir / "session_metadata_detailed_all_data.csv", encoding="latin1")

    # Derive site membership from metadata
    stanford_ids = set(
        subject_metadata.loc[
            subject_metadata["subject_id"].str.match(r"^P\d+$", na=False)
            & (subject_metadata["subject_id"].str.extract(r"P(\d+)")[0].astype(float) <= 24),
            "subject_id",
        ]
    )
    harvard_ids = set(
        subject_metadata.loc[
            subject_metadata["subject_id"].str.match(r"^P\d+$", na=False)
            & (subject_metadata["subject_id"].str.extract(r"P(\d+)")[0].astype(float) > 24),
            "subject_id",
        ]
    )
    harvard_hc_ids = set(
        subject_metadata.loc[subject_metadata["subject_id"].str.match(r"^HC\d+$", na=False), "subject_id"]
    )

    # Assign sites
    subject_metadata["experiment_site"] = subject_metadata["subject_id"].apply(
        lambda sid: assign_site(sid, stanford_ids, harvard_ids, harvard_hc_ids)
    )

    # Partition data files — do NOT filter reject_subjects here; that happens in preprocess_data.py
    all_files = list(compiled_dir.glob("*.mat")) + list(compiled_dir.glob("*.parquet"))
    hc_pd_files, asmhc_kp_files, asmhc_em_files = [], [], []
    for f in all_files:
        sid = f.stem.split("_")[0]
        if sid in UCLA_ASMHC_KP:
            asmhc_kp_files.append(f)
        elif sid in UCLA_ASMHC_EM:
            asmhc_em_files.append(f)
        else:
            hc_pd_files.append(f)

    print(f"Files — HC/PD: {len(hc_pd_files)}, asmHC_KP: {len(asmhc_kp_files)}, asmHC_EM: {len(asmhc_em_files)}")

    # Load HC/PD .mat files
    aggregate_df_list = [load_hc_pd_session(f) for f in hc_pd_files]

    # Load asmHC parquet files
    for f in asmhc_kp_files:
        df = pd.read_parquet(f)
        df["subject_id"] = f.name.split("_")[0]
        df["group"] = "hc"
        df["medication"] = "None"
        df["session_filename"] = f.name
        aggregate_df_list.append(df)

    if include_eye_tracking:
        for f in asmhc_em_files:
            df = pd.read_parquet(f)
            df["subject_id"] = f.name.split("_")[0]
            df["group"] = "hc"
            df["medication"] = "None"
            df["session_filename"] = f.name
            aggregate_df_list.append(df)
    else:
        em_subjects = {f.stem.split("_")[0] for f in asmhc_em_files}
        subject_metadata = subject_metadata[~subject_metadata["subject_id"].isin(em_subjects)]

    aggregate_df = pd.concat(aggregate_df_list, ignore_index=True)
    aggregate_df.replace("", np.nan, inplace=True)

    # Canonical column order
    aggregate_df = aggregate_df[
        ["subject_id", "group", "medication", "prior", "prior_direction", "prior_color",
         "color", "coherence", "target", "is_valid", "outcome", "choice", "reaction_time", "session_filename"]
    ]

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregate_df.to_csv(out_dir / "aggregate_all_data.csv", index=False)
    subject_metadata.to_csv(out_dir / "aggregate_all_metadata.csv", index=False)

    print(f"Saved {len(aggregate_df):,} rows → {out_dir / 'aggregate_all_data.csv'}")
    print(f"Saved {len(subject_metadata):,} subjects → {out_dir / 'aggregate_all_metadata.csv'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override output directory (default: compiled_dir from dir-config.yaml)")
    args = parser.parse_args()
    main(output_dir=args.output_dir)
