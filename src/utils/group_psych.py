"""Group-mean psychometric curves for the positive- and equal-prior colour conditions (2.20, 2.30)."""

import numpy as np
from scipy import stats

from src.utils import pmf_utils


def new_accumulators():
    """Empty accumulators for positive-prior and equal-prior psychometric data."""
    return {k: {"coh": [], "psych": [], "model": [], "x_hat": [], "y_hat": []} for k in ("pos", "eq")}


def collect_psych(data, accum):
    """Fit psychometric data for both color conditions and append to accum."""
    for key, color_val in [("pos", 1), ("eq", 0)]:
        coh, psych, model, x_hat, y_hat = pmf_utils.get_psychometric_data(data[data["color"] == color_val])
        for field, val in zip(["coh", "psych", "model", "x_hat", "y_hat"], [coh, psych, model, x_hat, y_hat]):
            accum[key][field].append(val)


def plot_mean_psych(ax, accum, pos_color, eq_color, err_style):
    """Plot group-mean psychometric curve with error bars and fitted line."""
    for key, color, label in [("pos", pos_color, "Positive Prior"), ("eq", eq_color, "Equal Prior")]:
        coh_arr = np.array(accum[key]["coh"])
        psych_arr = np.array(accum[key]["psych"])
        x_hat_arr = np.array(accum[key]["x_hat"])
        mean_coh, mean_psych = np.mean(coh_arr, axis=0), np.mean(psych_arr, axis=0)
        mean_x_hat = np.mean(x_hat_arr, axis=0)
        ax.plot(mean_coh, mean_psych, color=color, ls="", marker="o", ms=12, mec="k", label=label)
        ax.errorbar(mean_coh, mean_psych, yerr=stats.sem(psych_arr, axis=0), color=color, ls="", marker="o", ms=12, mec="k", **err_style)
        fit = pmf_utils.fit_psychometric_function(mean_coh, mean_psych, trial_counts=None, model_type="logit_4")
        ax.plot(mean_x_hat, fit.predict(mean_x_hat), color=color, ls="-", lw=4)
