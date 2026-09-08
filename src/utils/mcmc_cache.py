"""Cached NUTS fits for the trial-level colour x medication model.

The same posterior backs `2.25-posterior-interaction-with-mcmc` and the dissemination figure in
`2.30-dissemination`, so the model lives here and both notebooks go through `fit_or_load`. Whichever
notebook runs first samples and writes the draws to `CACHE_DIR`; every later run loads them back, so
the published numbers never depend on which notebook was executed last, and opening the dissemination
notebook does not mean waiting on the sampler.

Each cache entry carries a JSON sidecar recording the seed, the sampler settings and a fingerprint of
the exact design matrix that produced it. Changing the trial data changes the fingerprint and forces
a refit; an unchanged fingerprint means the stored draws are exactly what the recorded seed produces,
so the cache is a shortcut rather than a second source of truth.

Only what the manuscript reports is stored: the population-level `beta` draws (chain x draw x term),
the divergence flags, and the convergence diagnostics, which is everything the tables and figures use.
Re-running with `refit=True` regenerates the full InferenceData if the by-subject deviations are ever
needed.

R-hat and bulk-ESS are computed here rather than through arviz, and cached alongside the draws. They
are the rank-normalised split diagnostics of Vehtari et al. (2021), verified to reproduce
`arviz.rhat(method="rank")` and `arviz.ess(method="bulk")` to floating-point precision. Reported
numbers therefore do not move when arviz changes its API or its defaults, and loading a cached fit
needs nothing beyond numpy and scipy.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from config import dir_config

processed_dir = Path(dir_config.data.processed)
CACHE_DIR = processed_dir / "mcmc_fits"

TERMS = ["intercept", "coh", "col", "med", "col:med", "prev_rc", "prev_rc:med"]

# 2.25 keys its subtypes by the metadata value ("bradykinetic"), 2.30 by the short group name used
# throughout the dissemination figures ("brady"). Both must land on the same cache file, so names are
# normalised here rather than forcing either notebook to rename its variables.
SUBTYPE_CACHE_NAME = {"tremor": "tremor", "brady": "bradykinetic", "bradykinetic": "bradykinetic"}

# Sampler settings for the reported fits. Changing any of these changes the sidecar, which
# invalidates every cache entry written under the old settings.
FIT_SETTINGS = {"draws": 1000, "tune": 1000, "chains": 4, "target_accept": 0.9, "seed": 0}

# Sensitivity refits: half the draws is ample for a posterior mean and a tail probability.
LOSO_SETTINGS = {"draws": 500, "tune": 1000, "chains": 4, "target_accept": 0.9, "seed": 1}


def prepare_trials(trial_data, subject_ids):
    """One subtype's trial table: effect-coded predictors plus previous rewarded choice.

    Colour and medication are coded at +/- 0.5 so that `col:med` is the trial-level analogue of the
    subject-level difference-in-differences contrast. The first trial of each session has no previous
    choice and is dropped.
    """
    df = trial_data[trial_data["subject_id"].isin(subject_ids) & trial_data["medication"].isin(["off", "on"])].copy()
    df["signed_choice"] = df["choice"] * 2 - 1
    df["rewarded_choice"] = df["signed_choice"] * np.where(df["outcome"] == 1, 1, -1)
    df["prev_rc"] = df.groupby("session_filename")["rewarded_choice"].shift()
    df["coh"] = df["signed_coherence"] / 100
    df["col"] = np.where(df["color"] == 1, 0.5, -0.5)
    df["med"] = np.where(df["medication"] == "on", 0.5, -0.5)
    return df.dropna(subset=["prev_rc"]).reset_index(drop=True)


def build_design(subset):
    """Design matrix, outcomes and subject index for one subtype, in `TERMS` order."""
    subjects = subset["subject_id"].unique()
    subject_index = subset["subject_id"].map({s: i for i, s in enumerate(subjects)}).to_numpy()
    design = np.column_stack(
        [
            np.ones(len(subset)),
            subset["coh"],
            subset["col"],
            subset["med"],
            subset["col"] * subset["med"],
            subset["prev_rc"],
            subset["prev_rc"] * subset["med"],
        ]
    ).astype("float64")
    outcomes = subset["choice"].to_numpy().astype("int8")
    return design, outcomes, subject_index, subjects


def _autocovariance(chain):
    """Biased autocovariance at every lag, via FFT."""
    n_draw = len(chain)
    size = 1 << (2 * n_draw - 1).bit_length()
    centred = chain - chain.mean()
    freq = np.fft.rfft(centred, size)
    return np.fft.irfft(freq * np.conjugate(freq), size)[:n_draw] / n_draw


def _split_chains(draws):
    """(chain, draw) -> (2 x chain, draw // 2); a middle draw is dropped when the count is odd."""
    half = draws.shape[1] // 2
    return np.concatenate([draws[:, :half], draws[:, -half:]], axis=0)


def _rank_normalise(draws):
    """Rank-normalise to standard normal scores, using Blom's offset."""
    ranks = sp_stats.rankdata(draws.reshape(-1), method="average")
    return sp_stats.norm.ppf((ranks - 3 / 8) / (ranks.size + 1 / 4)).reshape(draws.shape)


def _rhat_plain(chains):
    """Between- over within-chain variance ratio for already split, normalised chains."""
    _, n_draw = chains.shape
    within = chains.var(axis=1, ddof=1).mean()
    between = n_draw * chains.mean(axis=1).var(ddof=1)
    return float(np.sqrt((between / within + n_draw - 1) / n_draw))


def rhat(draws):
    """Rank-normalised split R-hat, the maximum of the bulk and tail (folded) statistics."""
    split = _split_chains(draws)
    folded = np.abs(split - np.median(split))
    return max(_rhat_plain(_rank_normalise(split)), _rhat_plain(_rank_normalise(folded)))


def _ess(chains):
    """Effective sample size from Geyer's initial positive, monotone autocorrelation sequence."""
    n_chain, n_draw = chains.shape
    if n_draw < 4:
        return np.nan
    acov = np.array([_autocovariance(c) for c in chains])
    mean_var = acov[:, 0].mean() * n_draw / (n_draw - 1.0)
    var_plus = mean_var * (n_draw - 1.0) / n_draw
    if n_chain > 1:
        var_plus += chains.mean(axis=1).var(ddof=1)

    rho = np.zeros(n_draw)
    rho[0] = rho_even = 1.0
    rho[1] = rho_odd = 1.0 - (mean_var - acov[:, 1].mean()) / var_plus

    lag = 1
    while lag < (n_draw - 3) and (rho_even + rho_odd) > 0.0:
        rho_even = 1.0 - (mean_var - acov[:, lag + 1].mean()) / var_plus
        rho_odd = 1.0 - (mean_var - acov[:, lag + 2].mean()) / var_plus
        if (rho_even + rho_odd) >= 0:
            rho[lag + 1], rho[lag + 2] = rho_even, rho_odd
        lag += 2

    last = lag - 2
    if rho_even > 0:
        rho[last + 1] = rho_even

    lag = 1
    while lag <= last - 2:
        if (rho[lag + 1] + rho[lag + 2]) > (rho[lag - 1] + rho[lag]):
            rho[lag + 1] = (rho[lag - 1] + rho[lag]) / 2.0
            rho[lag + 2] = rho[lag + 1]
        lag += 2

    if np.isnan(rho).any():
        return np.nan
    total = n_chain * n_draw
    tau = -1.0 + 2.0 * np.sum(rho[: last + 1]) + np.sum(rho[last + 1 : last + 2])
    return float(total / max(tau, 1 / np.log10(total)))


def ess_bulk(draws):
    """Bulk effective sample size: ESS of the rank-normalised, split chains."""
    return _ess(_rank_normalise(_split_chains(draws)))


@dataclass(frozen=True)
class MCMCFit:
    """One subtype's cached posterior, with everything the manuscript reports about the fit."""

    beta: np.ndarray  # (chain, draw, term), in TERMS order
    diverging: np.ndarray  # (chain, draw)
    rhat: np.ndarray  # (term,)
    ess_bulk: np.ndarray  # (term,)
    n_subjects: int
    n_trials: int
    settings: dict = field(default_factory=dict)

    @property
    def draws(self):
        """Posterior draws with the chains pooled: (chain * draw, term)."""
        return self.beta.reshape(-1, self.beta.shape[-1])

    @property
    def divergences(self):
        return int(self.diverging.sum())

    def term(self, name):
        """Pooled draws for one population-level effect, e.g. ``fit.term("col:med")``."""
        return self.draws[:, TERMS.index(name)]


def _diagnose(beta):
    """Per-term R-hat and bulk ESS for a (chain, draw, term) array."""
    return (
        np.array([rhat(beta[:, :, i]) for i in range(beta.shape[2])]),
        np.array([ess_bulk(beta[:, :, i]) for i in range(beta.shape[2])]),
    )


def _fingerprint(design, outcomes, subject_index):
    """Hash of the exact numbers handed to the sampler, so stale caches cannot go unnoticed."""
    digest = hashlib.sha256()
    for array in (design, outcomes, subject_index):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()[:16]


def _pymc_version():
    """The installed pymc version, or None — loading a cached fit must not require pymc."""
    try:
        import pymc as pm
    except ImportError:
        return None
    return pm.__version__


def _sidecar(fingerprint, settings, subset, extra=None):
    return {
        "fingerprint": fingerprint,
        "settings": dict(settings),
        "terms": list(TERMS),
        "n_subjects": int(subset["subject_id"].nunique()),
        "n_trials": int(len(subset)),
        "pymc_version": _pymc_version(),
        "numpy_version": np.__version__,
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **(extra or {}),
    }


def _check_sidecar(path, fingerprint, settings):
    """Return the stored sidecar if it matches, else None with the reason printed."""
    if not path.exists():
        return None
    stored = json.loads(path.read_text())
    if stored.get("fingerprint") != fingerprint:
        print(f"{path.stem}: trial data changed since the cache was written, refitting")
        return None
    if stored.get("settings") != dict(settings):
        print(f"{path.stem}: sampler settings changed since the cache was written, refitting")
        return None
    current = _pymc_version()
    if current is not None and stored.get("pymc_version") != current:
        print(f"{path.stem}: cached under pymc {stored.get('pymc_version')}, now running {current}. Loading the stored draws — these are the numbers the figures were built from. Pass refit=True to resample under the current version.")
    return stored


def sample_model(subset, draws, tune, chains, target_accept, seed):
    """Hierarchical logistic regression with by-subject deviations on every population-level effect."""
    import pymc as pm

    design, outcomes, subject_index, subjects = build_design(subset)
    with pm.Model():
        population = pm.Normal("beta", 0, 2.5, shape=design.shape[1])
        subject_sd = pm.HalfNormal("sd_u", 1.0, shape=design.shape[1])
        subject_z = pm.Normal("z_u", 0, 1, shape=(len(subjects), design.shape[1]))
        eta = pm.math.sum(design * (population + (subject_z * subject_sd)[subject_index]), axis=1)
        pm.Bernoulli("obs", logit_p=eta, observed=outcomes)
        return pm.sample(
            draws,
            tune=tune,
            chains=chains,
            cores=chains,
            target_accept=target_accept,
            random_seed=seed,
            progressbar=False,
        )


def fit_or_load(name, subset, settings=None, refit=False):
    """Posterior for one subtype as an `MCMCFit`, loaded from `CACHE_DIR` when the cache is valid."""
    name = SUBTYPE_CACHE_NAME.get(name, name)
    settings = dict(settings or FIT_SETTINGS)
    design, outcomes, subject_index, _ = build_design(subset)
    fingerprint = _fingerprint(design, outcomes, subject_index)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    draws_path = CACHE_DIR / f"{name}.npz"
    meta_path = CACHE_DIR / f"{name}.json"

    def as_fit(beta, diverging, term_rhat, term_ess):
        return MCMCFit(
            beta=beta,
            diverging=diverging,
            rhat=term_rhat,
            ess_bulk=term_ess,
            n_subjects=int(subset["subject_id"].nunique()),
            n_trials=int(len(subset)),
            settings=settings,
        )

    if not refit and _check_sidecar(meta_path, fingerprint, settings) is not None and draws_path.exists():
        cached = np.load(draws_path)
        beta, diverging = cached["beta"], cached["diverging"]
        if "rhat" in cached.files:
            term_rhat, term_ess = cached["rhat"], cached["ess_bulk"]
        else:
            # Written before diagnostics were cached. They are a deterministic function of the stored
            # draws, so fill them in and rewrite rather than discarding a good fit.
            term_rhat, term_ess = _diagnose(beta)
            np.savez_compressed(draws_path, beta=beta, diverging=diverging, rhat=term_rhat, ess_bulk=term_ess)
            print(f"{name}: added diagnostics to the existing cache")
        print(f"{name}: loaded cached draws (seed {settings['seed']}, fingerprint {fingerprint})")
        return as_fit(beta, diverging, term_rhat, term_ess)

    idata = sample_model(subset, **settings)
    beta = idata.posterior["beta"].values
    diverging = idata.sample_stats["diverging"].values
    term_rhat, term_ess = _diagnose(beta)
    np.savez_compressed(draws_path, beta=beta, diverging=diverging, rhat=term_rhat, ess_bulk=term_ess)
    meta_path.write_text(
        json.dumps(
            _sidecar(
                fingerprint,
                settings,
                subset,
                {"max_rhat": float(term_rhat.max()), "min_ess_bulk": float(term_ess.min()), "divergences": int(diverging.sum())},
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"{name}: sampled and cached to {draws_path.relative_to(CACHE_DIR.parents[1])}")
    return as_fit(beta, diverging, term_rhat, term_ess)


def loso_or_load(name, subset, settings=None, refit=False):
    """Leave-one-subject-out refits for one subtype, cached as a summary table.

    Only the interaction summary per refit is kept — the mean, SD, 95% interval, P(> 0) and
    divergence count — which is everything the stability table and the forest plot report. About 80 s
    per refit when it does have to run.
    """
    name = SUBTYPE_CACHE_NAME.get(name, name)
    settings = dict(settings or LOSO_SETTINGS)
    design, outcomes, subject_index, _ = build_design(subset)
    fingerprint = _fingerprint(design, outcomes, subject_index)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    table_path = CACHE_DIR / f"{name}_loso.csv"
    meta_path = CACHE_DIR / f"{name}_loso.json"

    if not refit and _check_sidecar(meta_path, fingerprint, settings) is not None and table_path.exists():
        print(f"{name}: loaded cached leave-one-out refits (seed {settings['seed']}, fingerprint {fingerprint})")
        return pd.read_csv(table_path)

    held_out_subjects = sorted(subset["subject_id"].unique())
    print(f"{name}: refitting without each of {len(held_out_subjects)} subjects (~80 s each)...")
    records = []
    for held_out in held_out_subjects:
        reduced = subset[subset["subject_id"] != held_out].reset_index(drop=True)
        idata = sample_model(reduced, **settings)
        draws = idata.posterior["beta"].values.reshape(-1, len(TERMS))[:, TERMS.index("col:med")]
        low, high = np.percentile(draws, [2.5, 97.5])
        records.append(
            {
                "held_out": held_out,
                "mean": draws.mean(),
                "SD": draws.std(),
                "2.5%": low,
                "97.5%": high,
                "P(> 0)": (draws > 0).mean(),
                "CI excludes 0": bool(low > 0 or high < 0),
                "divergences": int(idata.sample_stats["diverging"].values.sum()),
            }
        )

    loso = pd.DataFrame(records)
    loso.to_csv(table_path, index=False)
    meta_path.write_text(json.dumps(_sidecar(fingerprint, settings, subset, {"refits": len(records)}), indent=2) + "\n")
    print(f"{name}: cached to {table_path.relative_to(CACHE_DIR.parents[1])}")
    return loso
