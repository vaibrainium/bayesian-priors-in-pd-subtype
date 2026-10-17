# Manuscript audit (October 2026)

Analyses behind the report "Learned priors in PD: analysis audit and manuscript strategy"
(https://claude.ai/code/artifact/aa4716e6-de04-4297-8168-b4fecd455d6c). Run from the repo root with
`.venv/bin/python -m scripts.audit.<name>`; every output goes to `outputs/audit/`. Nothing writes to the data dirs. The scripts load with `max_equal_prior_bias=None`, so they reproduce
the report (which handled the two-prior sessions itself via `TWO_PRIOR`); the project default now drops those subjects.

| Script | What it does |
| --- | --- |
| common.py | trial/session loaders; `TWO_PRIOR` lists the six sessions with opposing 75% priors on both colours |
| a01_mapping_check | infers each session's prior from its own target sequence |
| a02_session_features | per-session prior use (log-odds, SE) under three definitions, plus performance |
| a03_heterogeneity | two-stage REML tau, Q, I2, variance decomposition, controls vs PD spread |
| a04_session_drivers | design factors (visit, carry-over class, colour, side, site) vs session prior use |
| a05_reliability_dynamics | split-half reliability, change-score reliability, time course by quarter |
| a06_alignment_and_early_experience | early experienced contingency vs session prior use |
| a07_sequence_structure | trial-history models at 0% coherence |
| a08_null_simulation_and_eq_sessions | parametric nulls; UCLA no-prior blocks |
| a09_colour_direction_preference | mapping-sign (k) projection test of a stable colour-side preference |
| a10_hier_models | trial-level PyMC models (M0 reported; M1 + colour-side term; tremor moderators) |
| a11_tremor, a19_tremor_specificity | tremor robustness, measurement error, specificity, LOO-CV |
| a12_flip_vs_same | heterogeneity when mapping sign kept vs flipped, permutation test |
| a13_ddm_compare | exact-likelihood Wiener comparison (none/z/v/zv), 5-fold CV; `recover` mode for model recovery |
| a14_conditional_bias | prior effect by RT quartile |
| a15_performance_covariates, a16_coherence_specific | sensitivity/lapse/RT checks; prior use per coherence |
| a18_ddm_ppc | predictive check of the drift-only model |
