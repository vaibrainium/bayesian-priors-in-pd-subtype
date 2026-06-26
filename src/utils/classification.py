import pandas as pd


def get_subject_classification_ids(processed_metadata):
    # Build ON/OFF index pairs — PD subjects only
    subject_treatment_idx = pd.DataFrame(columns=["subject", "off", "on"])
    treatment_idx = pd.DataFrame(columns=["subject", "off", "on"])

    has_subtype_col = "trem_by_brady" in processed_metadata.columns

    for sub in processed_metadata["subject_id"].unique():
        sub_df = processed_metadata.loc[processed_metadata["subject_id"] == sub]
        off_idx = sub_df.loc[sub_df["treatment"] == "OFF"].index
        on_idx = sub_df.loc[sub_df["treatment"] == "ON"].index

        if off_idx.empty or on_idx.empty:
            continue  # HC subject or incomplete PD session

        subject_treatment_idx.loc[len(subject_treatment_idx)] = [sub, off_idx[0], on_idx[0]]

        if has_subtype_col and not sub_df["trem_by_brady"].isna().any():
            treatment_idx.loc[len(treatment_idx)] = [sub, off_idx[0], on_idx[0]]

    if not has_subtype_col:
        print("WARNING: 'trem_by_brady' column not found in metadata. Run 1.12 notebook first to compute UPDRS subtypes.")

    print(f"PD subjects with ON+OFF sessions: {len(subject_treatment_idx)}")
    print(f"PD subjects with UPDRS subtype: {len(treatment_idx)}")

    # is_pd: 1=PD, 0=HC (note: P32 ON row has is_pd=11, likely a data entry error — treated as PD)
    pd_subjects_all = processed_metadata[processed_metadata["is_pd"] != 0]["subject_id"].unique()
    hc_subjects = processed_metadata[processed_metadata["is_pd"] == 0]["subject_id"].unique()

    new_trem_subjects = processed_metadata.loc[processed_metadata["trem_vs_brady_type"] == "tremor"]["subject_id"].unique()
    new_brady_subjects = processed_metadata.loc[processed_metadata["trem_vs_brady_type"] == "bradykinetic"]["subject_id"].unique()
    new_intermediate_subjects = processed_metadata.loc[processed_metadata["trem_vs_brady_type"] == "intermediate"]["subject_id"].unique()
    print(f"All PD subjects:           {len(pd_subjects_all)}")
    print(f"  Tremor dominant:         {len(new_trem_subjects)} \t {new_trem_subjects}")
    print(f"  Brady dominant:          {len(new_brady_subjects)} \t {new_brady_subjects}")
    print(f"  Intermediate:            {len(new_intermediate_subjects)} \t {new_intermediate_subjects}")
    print(f"HC subjects:               {len(hc_subjects)} \t {hc_subjects}")

    subject_treatment_idx["subject"].unique()
    treatment_idx["subject"].unique()

    subjects_map = {
        "healthy": hc_subjects,
        "all": pd_subjects_all,
        "tremor_dominant": new_trem_subjects,
        "bradykinesia_dominant": new_brady_subjects,
    }

    grp_indices = {
        "healthy": processed_metadata.loc[processed_metadata["is_pd"] == 0].index,
        "pd_off": subject_treatment_idx["off"].loc[subject_treatment_idx["subject"].isin(subjects_map["all"])],
        "pd_on": subject_treatment_idx["on"].loc[subject_treatment_idx["subject"].isin(subjects_map["all"])],
        "tremor_off": subject_treatment_idx["off"].loc[subject_treatment_idx["subject"].isin(subjects_map["tremor_dominant"])],
        "tremor_on": subject_treatment_idx["on"].loc[subject_treatment_idx["subject"].isin(subjects_map["tremor_dominant"])],
        "brady_off": subject_treatment_idx["off"].loc[subject_treatment_idx["subject"].isin(subjects_map["bradykinesia_dominant"])],
        "brady_on": subject_treatment_idx["on"].loc[subject_treatment_idx["subject"].isin(subjects_map["bradykinesia_dominant"])],
    }
    return subjects_map, grp_indices
