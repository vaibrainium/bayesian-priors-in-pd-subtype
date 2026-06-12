"""Shared-basis GLM-HMM: fit one HC+PD state basis and analyse medication effects.

A parallel, additive layer on top of ``src.glm_hmm``. Nothing here modifies the
original per-(subtype x medication) pipeline; the heavy fitting/CV primitives are
imported and reused from ``src.glm_hmm`` (``process_sessions``, ``global_fit``,
``group_wise_fit_cv``, ``group_wise_fit``, ``select_best_n_states``). Only the thin
orchestration that pools *all* sessions into a single basis, plus principled K-selection
helpers (ICL/BIC/bootstrap stability), live here.

Rationale: all 41 PD subjects have both OFF and ON sessions, so a single shared state
basis turns the wasted between-group design into a within-subject medication design,
where per-session state occupancy is the dependent variable.
"""
