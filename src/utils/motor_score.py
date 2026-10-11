"""Continuous tremor/bradykinesia score, and the trial-level model moderated by it.

The analyses in `notebooks/` split patients into tremor-dominant and bradykinesia-dominant groups by
thresholding the OFF-medication ratio of mean tremor to mean bradykinesia item scores (`TREM_VS_BRADY`
in `config/main.yaml`). With 11 and 10 patients per group, and six tremor-dominant patients sitting
within 0.12 of the ratio-1 threshold, that split spends most of the information the ratio carries.
The notebooks in `notebooks_2/` use the ratio itself, as a continuous moderator, on all 22 patients who
have it (the split also dropped P24, the one "intermediate" patient).

**The score.** `log2(tremor_score / bradykinesia_score)` from the OFF-medication exam, so it is
independent of medication, exactly as the categorical subtype was. The raw ratio runs from 0.10 to 5.8
and is strongly right-skewed; on the log scale it is roughly symmetric, 0 is the old tremor threshold
(ratio 1), and +1 is a doubling of tremor relative to bradykinesia. For model fitting it is standardised
across the 22 patients (`score_z`), so population-level priors mean the same thing as in the reported
model and `col:med` is the medication effect on prior use at the average patient.

**The model.** The reported specification (`mcmc_cache.sample_model`: same priors, same non-centred
by-participant deviations on the seven within-participant terms, same sampler settings), with every
within-participant term also interacted with each between-participant moderator. The moderator terms
get no random slope, since a between-participant variable cannot vary within a participant. The
coefficient of interest is `score:col:med`, the change in the colour x medication interaction per SD of
the score. It is the continuous counterpart of `sub:col:med` in `2.27-subtype-comparison`.

Everything is cached to `CACHE_DIR` under the repo's gitignored `outputs/`, never the data directory,
with a sidecar fingerprint of the exact design matrix as in `mcmc_cache`: changed data or settings force
a refit, otherwise the stored draws are what the recorded seed produces.
"""

import contextlib
import io
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.mcmc_cache import FIT_SETTINGS, LOSO_SETTINGS, TERMS, _fingerprint, _pymc_version, ess_bulk, rhat

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "notebooks_2"
CACHE_DIR = OUTPUT_DIR / "mcmc_cache"

SCORE = "log2_trem_by_brady"
SCORE_LABEL = "tremor dominance, log2(tremor / bradykinesia), OFF exam"
WITHIN = list(TERMS)  # the seven within-participant terms, which carry the random effects


# --------------------------------------------------------------------------------------------- #
# The score
# --------------------------------------------------------------------------------------------- #


def motor_scores(metadata):
    """One row per patient with an OFF-medication tremor/bradykinesia ratio.

    Taken from the OFF session only, as the categorical subtype was, so the score cannot be moved by
    medication. `category` is the old threshold label, kept for reference and plotting only.
    """
    off = metadata[(metadata["treatment"] == "OFF") & (metadata["is_pd"] != 0) & metadata["trem_by_brady"].notna()]
    assert off["subject_id"].is_unique, "expected one OFF session per patient"
    scores = off.set_index("subject_id")[
        ["experiment_site", "tremor_score", "bradykinesia_score", "trem_by_brady", "trem_vs_brady_type", "UPDRS", "off_med_first", "off_meds_hrs"]
    ].rename(columns={"trem_vs_brady_type": "category", "UPDRS": "UPDRS_off", "experiment_site": "site"})
    scores[SCORE] = np.log2(scores["trem_by_brady"])
    scores["score_z"] = standardise(scores[SCORE])
    return scores.sort_values(SCORE)


def standardise(values):
    """Centre and scale by the population SD (ddof=0) of the patients given."""
    values = np.asarray(values, dtype=float)
    return (values - values.mean()) / values.std()


def per_trial(subset, by_subject):
    """Broadcast a per-participant Series (indexed by subject_id) onto a trial table."""
    values = subset["subject_id"].map(by_subject)
    assert values.notna().all(), f"no value for {sorted(set(subset['subject_id']) - set(by_subject.dropna().index))}"
    return values.to_numpy(dtype=float)


# --------------------------------------------------------------------------------------------- #
# Design and sampling
# --------------------------------------------------------------------------------------------- #


def moderated_design(subset, moderators=None):
    """Reported design plus every within-participant term interacted with each moderator.

    `moderators` maps a name to a per-trial array that is constant within participant. Returns the
    design matrix and its column names; the first `len(WITHIN)` columns are the reported design and
    are the ones that receive by-participant deviations.
    """
    within = np.column_stack(
        [np.ones(len(subset)), subset["coh"], subset["col"], subset["med"], subset["col"] * subset["med"], subset["prev_rc"], subset["prev_rc"] * subset["med"]]
    )
    columns, names = [within], list(WITHIN)
    for name, values in (moderators or {}).items():
        columns.append(within * np.asarray(values, dtype=float)[:, None])
        names += [name if t == "intercept" else f"{name}:{t}" for t in WITHIN]
    return np.column_stack(columns).astype("float64"), names


def participant_index(subset):
    """Integer participant index in order of first appearance, and the participant ids."""
    participants = subset["subject_id"].unique()
    return subset["subject_id"].map({s: i for i, s in enumerate(participants)}).to_numpy(), participants


def sample_glmm(design, outcomes, subject_index, n_subjects, n_random=len(WITHIN), settings=None, cores=None):
    """The reported hierarchical logistic regression on an arbitrary design.

    Priors and parameterisation are those of `mcmc_cache.sample_model`; only the first `n_random`
    columns get by-participant deviations.
    """
    import pymc as pm

    settings = dict(settings or FIT_SETTINGS)
    with pm.Model():
        population = pm.Normal("beta", 0, 2.5, shape=design.shape[1])
        subject_sd = pm.HalfNormal("sd_u", 1.0, shape=n_random)
        subject_z = pm.Normal("z_u", 0, 1, shape=(n_subjects, n_random))
        # element-wise sums rather than pm.math.dot, as in mcmc_cache: a multithreaded dot product can
        # change summation order between runs, and the sampler amplifies those rounding differences
        eta = pm.math.sum(design * population, axis=1) + pm.math.sum(design[:, :n_random] * (subject_z * subject_sd)[subject_index], axis=1)
        pm.Bernoulli("obs", logit_p=eta, observed=outcomes)
        return pm.sample(
            settings["draws"],
            tune=settings["tune"],
            chains=settings["chains"],
            cores=cores or settings["chains"],
            target_accept=settings["target_accept"],
            random_seed=settings["seed"],
            progressbar=False,
        )


@dataclass(frozen=True)
class Fit:
    """A cached posterior: population effects, by-participant deviations, and diagnostics."""

    beta: np.ndarray  # (chain, draw, term)
    u: np.ndarray  # (chain, draw, participant, random term): sd_u * z_u
    sd_u: np.ndarray  # (chain, draw, random term)
    diverging: np.ndarray  # (chain, draw)
    names: list
    participants: np.ndarray
    n_trials: int

    def term(self, name):
        """Pooled draws, (chain * draw,), for one population-level effect."""
        return self.beta[..., self.names.index(name)].reshape(-1)

    def deviation(self, name):
        """Pooled by-participant deviations, (chain * draw, participant), for one within-participant term."""
        return self.u[..., WITHIN.index(name)].reshape(-1, len(self.participants))

    @property
    def divergences(self):
        return int(self.diverging.sum())

    def diagnostics(self):
        """R-hat and bulk ESS for every population effect and every random-effect scale."""
        rows = {n: (rhat(self.beta[..., k]), ess_bulk(self.beta[..., k])) for k, n in enumerate(self.names)}
        rows |= {f"sd_u[{t}]": (rhat(self.sd_u[..., k]), ess_bulk(self.sd_u[..., k])) for k, t in enumerate(WITHIN)}
        return pd.DataFrame(rows, index=["r_hat", "ess_bulk"]).T


def fit_or_load(name, subset, moderators=None, extra_columns=None, settings=None, refit=False):
    """Score-moderated posterior for one trial table, from `CACHE_DIR` when the cache is valid.

    `moderators` maps a moderator name to a per-participant Series indexed by subject_id.
    `extra_columns` maps a name to a per-trial array added as a plain population-level effect, for
    covariates that vary within participant (e.g. the session's colour-orientation mapping). Changing
    either changes the design matrix and therefore the fingerprint, so a stale cache is never reused.
    """
    settings = dict(settings or FIT_SETTINGS)
    trial_moderators = {m: per_trial(subset, values) for m, values in (moderators or {}).items()}
    design, names = moderated_design(subset, trial_moderators)
    if extra_columns:
        design = np.column_stack([design] + [np.asarray(v, dtype=float) for v in extra_columns.values()])
        names = names + list(extra_columns)
    outcomes = subset["choice"].to_numpy().astype("int8")
    subject_index, participants = participant_index(subset)
    fingerprint = _fingerprint(design, outcomes, subject_index)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    draws_path, meta_path = CACHE_DIR / f"{name}.npz", CACHE_DIR / f"{name}.json"

    if not refit and draws_path.exists() and meta_path.exists():
        stored = json.loads(meta_path.read_text())
        if stored.get("fingerprint") == fingerprint and stored.get("settings") == settings and stored.get("names") == names:
            with np.load(draws_path) as cached:
                fit = Fit(cached["beta"], cached["u"], cached["sd_u"], cached["diverging"], names, participants, len(subset))
            print(f"{name}: loaded cached draws (seed {settings['seed']}, fingerprint {fingerprint})")
            return fit
        print(f"{name}: data, design or settings changed since the cache was written, refitting")

    idata = sample_glmm(design, outcomes, subject_index, len(participants), settings=settings)
    post = idata.posterior
    sd_u = post["sd_u"].values
    fit = Fit(post["beta"].values, post["z_u"].values * sd_u[:, :, None, :], sd_u, idata.sample_stats["diverging"].values, names, participants, len(subset))
    np.savez_compressed(draws_path, beta=fit.beta, u=fit.u, sd_u=fit.sd_u, diverging=fit.diverging)
    diag = fit.diagnostics()
    meta_path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "settings": settings,
                "names": names,
                "participants": [str(p) for p in participants],
                "n_trials": len(subset),
                "pymc_version": _pymc_version(),
                "numpy_version": np.__version__,
                "max_rhat": float(diag["r_hat"].max()),
                "min_ess_bulk": float(diag["ess_bulk"].min()),
                "divergences": fit.divergences,
                "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"{name}: sampled and cached ({fit.divergences} divergences, max r_hat {diag['r_hat'].max():.4f})")
    return fit


def describe(draws):
    """Posterior summary in the form used throughout the trial-level notebooks."""
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"mean": float(draws.mean()), "SD": float(draws.std()), "2.5%": float(low), "97.5%": float(high), "P(> 0)": float((draws > 0).mean())}


# --------------------------------------------------------------------------------------------- #
# Leave-one-participant-out refits of the moderation term
# --------------------------------------------------------------------------------------------- #


def _loso_worker(args):
    """One leave-one-out refit, run in its own process with chains sampled sequentially."""
    from threadpoolctl import threadpool_limits

    subset, moderators, held_out, settings, terms = args
    reduced = subset[subset["subject_id"] != held_out].reset_index(drop=True)
    design, names = moderated_design(reduced, {m: per_trial(reduced, v) for m, v in moderators.items()})
    subject_index, participants = participant_index(reduced)
    logging.getLogger("pymc").setLevel(logging.ERROR)
    with threadpool_limits(1), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        idata = sample_glmm(design, reduced["choice"].to_numpy().astype("int8"), subject_index, len(participants), settings=settings, cores=1)
    beta = idata.posterior["beta"].values.reshape(-1, len(names))
    row = {"held_out": held_out, "divergences": int(idata.sample_stats["diverging"].values.sum())}
    for term in terms:
        summary = describe(beta[:, names.index(term)])
        row |= {f"{term} {k}": v for k, v in summary.items()}
    return row


def loso_or_load(name, subset, moderators, terms, settings=None, workers=11, refit=False):
    """Refit once without each participant; cache a table of the requested terms' summaries.

    The moderator values are held at their full-sample standardisation, so every refit reports its
    coefficients in the same units as the full fit. Each refit samples its chains sequentially in a
    separate process, `workers` refits at a time.
    """
    settings = dict(settings or LOSO_SETTINGS)
    trial_moderators = {m: per_trial(subset, v) for m, v in moderators.items()}
    design, _ = moderated_design(subset, trial_moderators)
    subject_index, participants = participant_index(subset)
    fingerprint = _fingerprint(design, subset["choice"].to_numpy().astype("int8"), subject_index)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    table_path, meta_path = CACHE_DIR / f"{name}_loso.csv", CACHE_DIR / f"{name}_loso.json"
    if not refit and table_path.exists() and meta_path.exists():
        stored = json.loads(meta_path.read_text())
        if stored.get("fingerprint") == fingerprint and stored.get("settings") == settings and stored.get("terms") == list(terms):
            print(f"{name}: loaded cached leave-one-out refits")
            return pd.read_csv(table_path)
        print(f"{name}: data, settings or terms changed since the cache was written, recomputing")

    held_out = sorted(participants)
    print(f"{name}: {len(held_out)} leave-one-out refits, {workers} at a time...")
    jobs = [(subset, moderators, h, settings, list(terms)) for h in held_out]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        loso = pd.DataFrame(list(pool.map(_loso_worker, jobs)))
    loso.to_csv(table_path, index=False)
    meta_path.write_text(
        json.dumps({"fingerprint": fingerprint, "settings": settings, "terms": list(terms), "refits": len(loso), "written": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=2)
        + "\n"
    )
    print(f"{name}: cached to {table_path.relative_to(OUTPUT_DIR.parent)}")
    return loso


# --------------------------------------------------------------------------------------------- #
# Correlation with small samples
# --------------------------------------------------------------------------------------------- #


def correlate(x, y, n_perm=20_000, n_boot=10_000, seed=0):
    """Spearman and Pearson correlation with permutation p-values and a bootstrap CI for rho.

    With n around 20 the asymptotic p-values are rough, so the reported p is from permuting y against
    x (two-sided). The bootstrap resamples participants and skips resamples with fewer than four
    distinct values, where a rank correlation is degenerate. For a variable with fewer than five levels
    (e.g. Hoehn and Yahr stage) a percentile bootstrap of rho is unreliable, so no interval is given.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    rng = np.random.default_rng(seed)

    rho = stats.spearmanr(x, y).statistic
    r = stats.pearsonr(x, y).statistic
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    perms = [rng.permutation(len(y)) for _ in range(n_perm)]  # shared by both statistics
    null_rho = np.array([np.corrcoef(rx, ry[p])[0, 1] for p in perms])
    null_r = np.array([np.corrcoef(x, y[p])[0, 1] for p in perms])

    boot = []
    if min(len(np.unique(x)), len(np.unique(y))) >= 5:
        for _ in range(n_boot):
            idx = rng.integers(0, len(x), len(x))
            if len(np.unique(x[idx])) >= 4 and len(np.unique(y[idx])) >= 4:
                boot.append(stats.spearmanr(x[idx], y[idx]).statistic)
    low, high = np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan)
    return {
        "n": int(len(x)),
        "spearman rho": rho,
        "rho 2.5%": low,
        "rho 97.5%": high,
        "p (perm, rho)": (np.sum(np.abs(null_rho) >= abs(rho) - 1e-12) + 1) / (n_perm + 1),
        "pearson r": r,
        "p (perm, r)": (np.sum(np.abs(null_r) >= abs(r) - 1e-12) + 1) / (n_perm + 1),
    }


def leave_one_out_rho(x, y, labels):
    """Spearman rho with each participant dropped in turn, to show which points carry a correlation."""
    x, y, labels = np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(labels)
    full = stats.spearmanr(x, y).statistic
    rows = [{"dropped": labels[i], "rho": stats.spearmanr(np.delete(x, i), np.delete(y, i)).statistic} for i in range(len(x))]
    out = pd.DataFrame(rows).set_index("dropped")
    out["shift"] = out["rho"] - full
    return out.sort_values("shift")
