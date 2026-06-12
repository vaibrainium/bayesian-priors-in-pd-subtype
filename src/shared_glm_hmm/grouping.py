"""Session grouping for the shared-basis fit.

Unlike :func:`src.glm_hmm.data_preparation.get_group_session_ids`, which splits sessions
into five independent ``(subtype x medication)`` groups (each fit as its own model), this
pools *every* session (HC + all PD, both medication states) into a single group so the
GLM-HMM learns one common state basis. Medication and subtype then become covariates over
per-session state occupancy rather than separate model fits.
"""

import pandas as pd


def get_pooled_session_ids(metadata: pd.DataFrame) -> dict:
    """Map a single pooled group to every session ID (HC + all PD, both treatments).

    Returns ``{"all_subjects": [...sorted session ids...]}`` so the existing pooled-CV
    machinery (which pools all sessions within a "group") fits one shared basis.
    """
    session_ids = sorted(metadata["session_id"].dropna().unique().tolist())
    return {"all_subjects": session_ids}


def session_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    """Per-session label table for the within-subject analysis.

    One row per session with the fields needed to join occupancy to a within-subject
    medication contrast: ``session_id, subject_id, is_pd, subtype, medication``.
    ``subtype`` is ``HC`` for controls, else ``trem_vs_brady_type`` (tremor / bradykinetic
    / intermediate). ``medication`` is ``off`` / ``on`` / ``none`` (HC).
    """
    cols = ["session_id", "subject_id", "is_pd", "trem_vs_brady_type", "treatment"]
    md = metadata[cols].dropna(subset=["session_id"]).drop_duplicates("session_id").copy()

    md["subtype"] = md.apply(
        lambda r: "HC" if r["is_pd"] == 0 else r["trem_vs_brady_type"], axis=1
    )
    md["medication"] = (
        md["treatment"].astype("string").str.lower().fillna("none")
    )
    return md[["session_id", "subject_id", "is_pd", "subtype", "medication"]].reset_index(drop=True)
