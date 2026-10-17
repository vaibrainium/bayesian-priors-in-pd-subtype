Notebook organization
naming structure: 0.00-xxx-yyy-zzz.ipynb where first number is for modules, first decimal is for major method change and second number is for minor method change

modules:
0: preprocessing
1: behavioral analysis
2: ephys analysis
3: modeling (behavioral and neural)

Data in and out
Notebooks never build paths or call read_csv/to_csv/savefig on files directly. Everything goes
through src/data_io:

    from src.data_io import catalog

    trials = catalog.load_trials()                      # valid trials, subjects passing the accuracy screen
    metadata = catalog.load_metadata()                  # session table + every stage's columns, same screen
    trials90 = catalog.load_trials(min_easy_accuracy=0.9)   # a different screen, no re-run needed
    compiled = catalog.load_trials(stage="compiled")        # decoded sessions, before preprocessing
    catalog.save_table(df, "subtype_comparison.csv")    # results -> outputs/
    catalog.save_figure(fig, "subtype_comparison")      # outputs/<name>.pdf and .png

Where things may be written (enforced by src/data_io/paths.writable):
- raw/ is read only.
- compiled/ and processed/ hold derived data, written only by the pipeline stages (0.x, 3.10, 3.40,
  scripts/) through catalog.save.
- outputs/ (gitignored) holds analysis results; dissemination/ holds the tracked slide decks.

Pipeline: 0.00 / 0.01 decode every raw session into compiled/sessions/<stem>.parquet (one table per
session, same columns) -> 0.10 joins them -> 0.20 preprocesses -> 0.30 / 0.40 add metadata stages.
No stage removes sessions for low accuracy. 0.20 records each session's accuracy on 100%-coherence
trials (easy_accuracy_pct, in %), and the loaders keep only subjects whose sessions all reach the threshold
(config BEHAVIOR.min_easy_accuracy unless given; None keeps everyone). A second load-time screen drops
subjects with a session that did not run the intended design: the presented targets of the prior colour
must split 75:25 and those of the equal colour 50:50 (config BEHAVIOR.prior_design_tolerance and
prior_base_rates, measured by src.utils.sessions.prior_design_error; None switches it off). Pipeline
stages read with min_easy_accuracy=None, prior_design_tolerance=None so their files cover every session.

Session metadata is built in stages, and each stage writes its own file under processed/metadata/,
keyed by session_id. No stage rewrites another's file; catalog.load_metadata() joins them.
  0.20 -> sessions    0.30 -> updrs    0.40 -> psych_params    3.40 -> ddm_params
Re-running a stage only replaces that stage's columns. load_metadata(ddm_params=True) adds the DDM fits.

To check a pipeline change without touching shared data, point the roots at a scratch copy with
PD_RAW_DIR, PD_COMPILED_DIR, PD_PROCESSED_DIR or PD_OUTPUTS_DIR.

