"""Robustness checks for the trial-level colour x medication interaction.

Two questions that the reported posterior and the leave-one-subject-out refits cannot answer on
their own, and that a reviewer looking at Figure 2 will ask:

1. **Is the interaction distinguishable from noise without trusting the model's variance
   assumptions?** `permutation_or_load` runs an exact within-participant sign-flip test. Each
   patient has exactly one OFF and one ON session, so under the null of no medication effect the
   two labels are exchangeable within a patient. With 11 and 10 patients there are only 2,048 and
   1,024 label assignments, so the whole reference distribution is enumerated and the p-value is
   exact rather than Monte Carlo.

   The test statistic is the point estimate of `col:med` from a variational fit of the reported
   specification, which takes about three seconds against roughly eighty for a sampled fit. Only
   the point estimate is used. `2.24-variational-calibration` shows the variational *posterior SD*
   is about three times too narrow, which is why that fit cannot be reported, but its point
   estimate reproduces the sampled posterior mean to two decimals. A permutation test takes its
   uncertainty from the reference distribution, so a fast, correctly centred, badly scaled
   estimator is a valid statistic where it is not a valid posterior.

   `agreement_or_load` is the check on that substitution: it refits a spread of relabellings both
   ways and compares the two estimators on the same datasets, so the claim that the fast statistic
   stands in for the reported one is tested under relabelling rather than only on the real data.

2. **Does the estimate depend on the priors or on the random-effects structure?**
   `spec_or_load` refits the same data under the alternative specifications in `SPECS`: wider and
   tighter priors on the population-level effects, heavier-tailed and tighter priors on the
   by-participant scales, correlated rather than independent random effects, reduced random-effects
   structures, and the centred parameterisation.

Everything here caches to `CACHE_DIR` under the repo's gitignored `outputs/`, never to the data
directory. Each entry carries a JSON sidecar recording the sampler settings, the specification and a
fingerprint of the design matrix, on the same terms as `mcmc_cache`: a changed fingerprint forces a
refit, an unchanged one means the stored numbers are exactly what the recorded seed produces.
"""

import contextlib
import io
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.mcmc_cache import TERMS, _fingerprint, build_design, ess_bulk, rhat

CACHE_DIR = Path(__file__).resolve().parents[2] / "outputs" / "robustness_cache"

COL_MED = TERMS.index("col:med")

# Sensitivity refits. Full draws, because these are compared with the reported 95% intervals rather
# than only with a posterior mean, and a separate seed from the reported fits so that agreement
# between them is not agreement between two runs of the same chain.
SENSITIVITY_SETTINGS = {"draws": 1000, "tune": 1000, "chains": 4, "target_accept": 0.9, "seed": 2}

REPORTED_SPEC = "reported"


@dataclass(frozen=True)
class Spec:
    """One alternative specification of the reported model.

    Only the priors, the random-effects structure and the parameterisation vary. The likelihood, the
    design matrix and the data are those of the reported fit in every case.
    """

    name: str
    population_prior: str = "Normal(0, 2.5)"
    scale_prior: str = "HalfNormal(1)"
    random_terms: tuple = tuple(TERMS)
    correlated: bool = False
    centred: bool = False
    varies: str = ""

    def as_dict(self):
        return {
            "population_prior": self.population_prior,
            "scale_prior": self.scale_prior,
            "random_terms": list(self.random_terms),
            "correlated": self.correlated,
            "centred": self.centred,
        }


SPECS = {
    spec.name: spec
    for spec in [
        Spec(REPORTED_SPEC, varies="the reported fit, resampled under a new seed"),
        Spec("prior N(0, 1)", population_prior="Normal(0, 1)", varies="population-level prior, four times tighter"),
        Spec("prior N(0, 10)", population_prior="Normal(0, 10)", varies="population-level prior, four times wider"),
        Spec("prior t3(0, 2.5)", population_prior="StudentT(3, 0, 2.5)", varies="population-level prior, heavy tailed"),
        Spec("scale HalfNormal(0.5)", scale_prior="HalfNormal(0.5)", varies="by-participant scales, shrunk harder"),
        Spec("scale HalfCauchy(1)", scale_prior="HalfCauchy(1)", varies="by-participant scales, heavy tailed"),
        Spec("scale Exponential(1)", scale_prior="Exponential(1)", varies="by-participant scales, different family"),
        Spec("correlated random effects", correlated=True, varies="random effects correlated (LKJ(2)) rather than independent"),
        Spec("random effects on col:med only", random_terms=("intercept", "col:med"), varies="random-effects structure, reduced"),
        Spec("random intercept only", random_terms=("intercept",), varies="random-effects structure, minimal"),
        Spec("centred parameterisation", centred=True, varies="parameterisation, not the model"),
    ]
}


def _population_prior(pm, name, kind, shape):
    if kind == "StudentT(3, 0, 2.5)":
        return pm.StudentT(name, nu=3, mu=0, sigma=2.5, shape=shape)
    sd = {"Normal(0, 2.5)": 2.5, "Normal(0, 1)": 1.0, "Normal(0, 10)": 10.0}[kind]
    return pm.Normal(name, 0, sd, shape=shape)


def _scale_prior(pm, name, kind, shape):
    if kind == "HalfCauchy(1)":
        return pm.HalfCauchy(name, 1.0, shape=shape)
    if kind == "Exponential(1)":
        return pm.Exponential(name, 1.0, shape=shape)
    sd = {"HalfNormal(1)": 1.0, "HalfNormal(0.5)": 0.5}[kind]
    return pm.HalfNormal(name, sd, shape=shape)


def _scale_dist(pm, kind):
    """The same scale prior as an unregistered distribution, for `sd_dist` in LKJCholeskyCov."""
    if kind == "HalfCauchy(1)":
        return pm.HalfCauchy.dist(1.0)
    if kind == "Exponential(1)":
        return pm.Exponential.dist(1.0)
    return pm.HalfNormal.dist({"HalfNormal(1)": 1.0, "HalfNormal(0.5)": 0.5}[kind])


def sample_spec(subset, spec, settings=None):
    """Sample one subtype under one specification; returns the InferenceData."""
    import pymc as pm

    settings = dict(settings or SENSITIVITY_SETTINGS)
    design, outcomes, subject_index, subjects = build_design(subset)
    random_index = [TERMS.index(t) for t in spec.random_terms]
    n_subjects, n_random = len(subjects), len(random_index)

    with pm.Model():
        population = _population_prior(pm, "beta", spec.population_prior, len(TERMS))

        if spec.correlated:
            chol, _, sd_u = pm.LKJCholeskyCov(
                "chol", n=n_random, eta=2.0, sd_dist=_scale_dist(pm, spec.scale_prior), compute_corr=True
            )
            pm.Deterministic("sd_u", sd_u)
            subject_z = pm.Normal("z_u", 0, 1, shape=(n_subjects, n_random))
            u = subject_z @ chol.T
        elif spec.centred:
            subject_sd = _scale_prior(pm, "sd_u", spec.scale_prior, n_random)
            u = pm.Normal("u", 0, subject_sd, shape=(n_subjects, n_random))
        else:
            subject_sd = _scale_prior(pm, "sd_u", spec.scale_prior, n_random)
            subject_z = pm.Normal("z_u", 0, 1, shape=(n_subjects, n_random))
            u = subject_z * subject_sd

        # element-wise sums rather than pm.math.dot, as in mcmc_cache.sample_model: a multithreaded
        # dot product over 13,000 rows can change summation order between runs
        eta = pm.math.sum(design * population, axis=1) + pm.math.sum(design[:, random_index] * u[subject_index], axis=1)
        pm.Bernoulli("obs", logit_p=eta, observed=outcomes)
        return pm.sample(
            settings["draws"],
            tune=settings["tune"],
            chains=settings["chains"],
            cores=settings["chains"],
            target_accept=settings["target_accept"],
            random_seed=settings["seed"],
            progressbar=False,
        )


def describe(draws):
    """Posterior summary in the form used throughout the trial-level notebooks."""
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"mean": float(draws.mean()), "SD": float(draws.std()), "2.5%": float(low), "97.5%": float(high), "P(> 0)": float((draws > 0).mean())}


def _sidecar(fingerprint, settings, subset, extra):
    return {
        "fingerprint": fingerprint,
        "settings": dict(settings),
        "n_subjects": int(subset["subject_id"].nunique()),
        "n_trials": int(len(subset)),
        "numpy_version": np.__version__,
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **extra,
    }


def _valid_sidecar(path, fingerprint, settings, extra):
    """True when the cached entry was written from these data, settings and specification."""
    if not path.exists():
        return False
    stored = json.loads(path.read_text())
    for key, value in {"fingerprint": fingerprint, "settings": dict(settings), **extra}.items():
        if stored.get(key) != value:
            print(f"{path.stem}: {key} changed since the cache was written, recomputing")
            return False
    return True


def spec_or_load(subtype, subset, spec_name, settings=None, refit=False):
    """`col:med` posterior under one alternative specification, cached to `CACHE_DIR`.

    Returns `(summary_dict, draws)`, where the draws are the pooled `col:med` draws. Only `col:med`
    is kept: it is the coefficient the sensitivity table reports, and storing every term would make
    the cache many times larger for numbers nothing reads.
    """
    spec = SPECS[spec_name]
    settings = dict(settings or SENSITIVITY_SETTINGS)
    design, outcomes, subject_index, _ = build_design(subset)
    fingerprint = _fingerprint(design, outcomes, subject_index)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    slug = spec_name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("/", "-")
    draws_path = CACHE_DIR / f"{subtype}_spec_{slug}.npz"
    meta_path = draws_path.with_suffix(".json")

    if not refit and draws_path.exists() and _valid_sidecar(meta_path, fingerprint, settings, {"spec": spec.as_dict()}):
        with np.load(draws_path) as cached:
            draws = cached["col_med"]
            summary = json.loads(meta_path.read_text())["summary"]
        print(f"{subtype} / {spec_name}: loaded cached draws")
        return summary, draws

    idata = sample_spec(subset, spec, settings)
    beta = idata.posterior["beta"].values[..., COL_MED]  # (chain, draw)
    draws = beta.reshape(-1)
    summary = {
        **describe(draws),
        "divergences": int(idata.sample_stats["diverging"].values.sum()),
        "r_hat": float(rhat(beta)),
        "ESS": float(ess_bulk(beta)),
    }
    np.savez_compressed(draws_path, col_med=draws)
    meta_path.write_text(json.dumps(_sidecar(fingerprint, settings, subset, {"spec": spec.as_dict(), "summary": summary}), indent=2) + "\n")
    print(f"{subtype} / {spec_name}: sampled and cached")
    return summary, draws


# --------------------------------------------------------------------------------------------- #
# Exact within-participant sign-flip test
# --------------------------------------------------------------------------------------------- #

VC_SEVEN = {
    "a": "0+C(subj)",
    "b": "0+C(subj):col",
    "c": "0+C(subj):med",
    "d": "0+C(subj):col:med",
    "e": "0+C(subj):coh",
    "f": "0+C(subj):prev_rc",
    "g": "0+C(subj):prev_rc:med",
}
PERMUTATION_FORMULA = "choice ~ coh + col*med + prev_rc*med"

_WORKER_FRAME = None


def flip_labels(subset, flips):
    """Relabel each participant's OFF and ON sessions according to a +/-1 vector.

    `build_design` derives `med`, `col:med` and `prev_rc:med` from the `med` column, so flipping that
    one column relabels all three medication terms consistently.
    """
    subjects = subset["subject_id"].unique()
    index = subset["subject_id"].map({s: i for i, s in enumerate(subjects)}).to_numpy()
    return subset.assign(med=subset["med"].to_numpy() * flips[index])


def exact_sign_flips(n_subjects):
    """All 2**n within-participant OFF/ON relabellings, as a (2**n, n) array of +/-1.

    Row 0 is the identity, so the observed data are the first entry of the reference distribution,
    which is what makes the p-value include the observed statistic as it must.
    """
    bits = ((np.arange(2**n_subjects)[:, None] >> np.arange(n_subjects)[None, :]) & 1).astype("float64")
    return 1.0 - 2.0 * bits


def interaction_estimate(frame, flips=None):
    """Variational point estimate of `col:med` for one (possibly relabelled) dataset."""
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    from threadpoolctl import threadpool_limits

    if flips is not None:
        subjects = frame["subject_id"].unique()
        index = frame["subject_id"].map({s: i for i, s in enumerate(subjects)}).to_numpy()
        frame = frame.assign(med=frame["med"].to_numpy() * flips[index])
    with threadpool_limits(1), contextlib.redirect_stdout(io.StringIO()):
        fit = BinomialBayesMixedGLM.from_formula(PERMUTATION_FORMULA, VC_SEVEN, frame).fit_vb(verbose=False)
    return float(fit.fe_mean[list(fit.model.exog_names).index("col:med")])


def _init_worker(frame):
    global _WORKER_FRAME
    _WORKER_FRAME = frame


def _worker(flips):
    try:
        return interaction_estimate(_WORKER_FRAME, flips)
    except Exception:  # a relabelling that the variational fit cannot converge on
        return np.nan


def permutation_or_load(subtype, subset, workers=None, refit=False, flips=None):
    """Exact sign-flip reference distribution for `col:med`, cached to `CACHE_DIR`.

    Enumerates every within-participant OFF/ON relabelling unless `flips` is given, refits the
    statistic on each, and returns a DataFrame with one row per relabelling. The observed statistic
    is row 0.
    """
    frame = subset.assign(subj=subset["subject_id"])
    design, outcomes, subject_index, subjects = build_design(subset)
    fingerprint = _fingerprint(design, outcomes, subject_index)
    flips = exact_sign_flips(len(subjects)) if flips is None else flips
    settings = {"formula": PERMUTATION_FORMULA, "relabellings": int(len(flips)), "exhaustive": len(flips) == 2 ** len(subjects)}

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    table_path = CACHE_DIR / f"{subtype}_signflip.csv"
    meta_path = CACHE_DIR / f"{subtype}_signflip.json"

    if not refit and table_path.exists() and _valid_sidecar(meta_path, fingerprint, settings, {}):
        print(f"{subtype}: loaded cached sign-flip distribution ({len(flips):,} relabellings)")
        return pd.read_csv(table_path)

    workers = workers or max(1, min(28, (os.cpu_count() or 2) - 2))
    print(f"{subtype}: refitting {len(flips):,} relabellings on {workers} workers (~3 s each, so ~{len(flips) * 3 / workers / 60:.0f} min)...")
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(frame,)) as pool:
        statistics = list(pool.map(_worker, flips, chunksize=8))

    table = pd.DataFrame(
        {
            "relabelling": np.arange(len(flips)),
            "n_flipped": (flips < 0).sum(axis=1).astype(int),
            "col:med": statistics,
        }
    )
    table.to_csv(table_path, index=False)
    meta_path.write_text(json.dumps(_sidecar(fingerprint, settings, subset, {"failed": int(np.isnan(statistics).sum())}), indent=2) + "\n")
    print(f"{subtype}: cached to {table_path.relative_to(CACHE_DIR.parents[1])}")
    return table


def sign_flip_pvalues(table, observed=None):
    """Exact one- and two-sided p-values from a sign-flip reference distribution."""
    statistics = table["col:med"].to_numpy()
    observed = statistics[0] if observed is None else observed
    finite = statistics[np.isfinite(statistics)]
    n = len(finite)
    return {
        "observed": float(observed),
        "relabellings": int(n),
        "null mean": float(finite.mean()),
        "null SD": float(finite.std(ddof=1)),
        "p one-sided": float((finite >= observed).sum() / n),
        "p two-sided": float((np.abs(finite) >= abs(observed)).sum() / n),
    }


def agreement_or_load(subtype, subset, relabellings, settings=None, refit=False):
    """Fast statistic against a sampled posterior mean, on the same relabelled datasets.

    The sign-flip test would ideally use the reported estimator on every relabelling, which is a
    hundred hours of sampling. This refits a handful of relabellings both ways and asks whether the
    substitution changes anything: if the two estimators agree to well within the spread of the
    reference distribution, the p-value computed from the fast one is the p-value the sampled one
    would have given. `relabellings` maps a row index of the reference distribution to its flip
    vector.
    """
    from src.utils.mcmc_cache import LOSO_SETTINGS

    settings = dict(settings or LOSO_SETTINGS)
    design, outcomes, subject_index, _ = build_design(subset)
    fingerprint = _fingerprint(design, outcomes, subject_index)
    sidecar_settings = {**settings, "rows": sorted(relabellings)}

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    table_path = CACHE_DIR / f"{subtype}_agreement.csv"
    meta_path = CACHE_DIR / f"{subtype}_agreement.json"

    if not refit and table_path.exists() and _valid_sidecar(meta_path, fingerprint, sidecar_settings, {}):
        print(f"{subtype}: loaded cached estimator agreement ({len(relabellings)} relabellings)")
        return pd.read_csv(table_path)

    rows = []
    for row, flips in sorted(relabellings.items()):
        relabelled = flip_labels(subset, flips)
        idata = sample_spec(relabelled, SPECS[REPORTED_SPEC], settings)
        rows.append(
            {
                "relabelling": int(row),
                "n_flipped": int((flips < 0).sum()),
                "variational": interaction_estimate(relabelled.assign(subj=relabelled["subject_id"])),
                "sampled": float(idata.posterior["beta"].values[..., COL_MED].mean()),
                "divergences": int(idata.sample_stats["diverging"].values.sum()),
            }
        )
        print(f"{subtype}: relabelling {row} done ({len(rows)}/{len(relabellings)})", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(table_path, index=False)
    meta_path.write_text(json.dumps(_sidecar(fingerprint, sidecar_settings, subset, {}), indent=2) + "\n")
    print(f"{subtype}: cached to {table_path.relative_to(CACHE_DIR.parents[1])}")
    return table


def agreement_rows(n_subjects, n_rows=16, seed=3):
    """A spread of reference-distribution rows to check the estimators against.

    Row 0, the identity, is always included. The rest are drawn at random, so the comparison is not
    confined to the relabellings that happen to sit near the observed value.
    """
    rng = np.random.default_rng(seed)
    rows = np.concatenate([[0], rng.choice(np.arange(1, 2**n_subjects), n_rows - 1, replace=False)])
    flips = exact_sign_flips(n_subjects)
    return {int(row): flips[row] for row in rows}
