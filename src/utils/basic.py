from pathlib import Path

import dill
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import style

from src.utils.psychometric_function import PsychometricFunction

BASE_DIR = Path("/src/")


# STYLE = BASE_DIR / "src" / "utils" /"dissemination.mplstyle"
# style.use(STYLE)

# plt.rcParams.update({"axes.prop_cycle": plt.cycler(color=['d11149', '1a8fe3', '1ccd6a', 'e6c229', '6610f2', 'f17105',  '65e5f3', 'bd8ad5', 'b16b57'])})


def plot_scatter(xs, ys, **scatter_kwargs):
    """Draw scatter plot with consistent style (e.g. unfilled points)"""
    defaults = {"alpha": 0.6, "lw": 3, "s": 80, "color": "C0", "facecolors": "none", "marker": "."}

    for k, v in defaults.items():
        val = scatter_kwargs.get(k, v)
        scatter_kwargs[k] = val

    plt.scatter(xs, ys, **scatter_kwargs)


def plot_line(xs, ys, **plot_kwargs):
    """Plot line with consistent style (e.g. bordered lines)"""
    linewidth = plot_kwargs.get("linewidth", 3)
    plot_kwargs["linewidth"] = linewidth

    # Copy settings for background
    background_plot_kwargs = {k: v for k, v in plot_kwargs.items()}
    background_plot_kwargs["linewidth"] = linewidth + 2
    background_plot_kwargs["color"] = "white"
    del background_plot_kwargs["label"]  # no legend label for background

    plt.plot(xs, ys, **background_plot_kwargs, zorder=30)
    plt.plot(xs, ys, **plot_kwargs, zorder=31)
    
    
def plot_errorbar(xs, ys, error_lower, error_upper, colors="C0", error_width=12, alpha=0.3):
    if isinstance(colors, str):
        colors = [colors] * len(xs)

    """Draw thick error bars with consistent style"""
    for ii, (x, y, err_l, err_u) in enumerate(zip(xs, ys, error_lower, error_upper)):
        marker, _, bar = plt.errorbar(
            x=x,
            y=y,
            yerr=np.array((err_l, err_u))[:, None],
            ls="none",
            color=colors[ii],
            zorder=1,
        )
        plt.setp(bar[0], capstyle="round")
        marker.set_fillstyle("none")
        bar[0].set_alpha(alpha)
        bar[0].set_linewidth(error_width)
        
    
def plot_x_errorbar(xs, ys, error_lower, error_upper, colors="C0", error_width=12, alpha=0.3):
    if isinstance(colors, str):
        colors = [colors] * len(xs)

    """Draw thick error bars with consistent style"""
    for ii, (x, y, err_l, err_u) in enumerate(zip(xs, ys, error_lower, error_upper)):
        marker, _, bar = plt.errorbar(
            x=x,
            y=y,
            xerr=np.array((err_l, err_u))[:, None],
            ls="none",
            color=colors[ii],
            zorder=1,
        )
        plt.setp(bar[0], capstyle="round")
        marker.set_fillstyle("none")
        bar[0].set_alpha(alpha)
        bar[0].set_linewidth(error_width)
        
        
def fit_psychometric_function(x_data, y_data, **model_kwargs):
    defaults = {"model": "logit_4", "var_lims": (1e-5, 10), "lapse_rate_lims": (1e-5, 0.2), "guess_rate_lims": (1e-5, 0.2)}
    for k, v in defaults.items():
        val = model_kwargs.get(k, v)
        model_kwargs[k] = val
    model = PsychometricFunction(**model_kwargs).fit(x_data, y_data)
    return model


def get_psychometric_data(data, positive_direction='right'):
    x_data = np.asarray([])
    y_data = np.asarray([])
    for _, coh in enumerate(np.unique(data['signed_coherence'])):
        if positive_direction == 'right':
            x_data = np.append(x_data, coh)
            y_data = np.append(y_data, np.sum(data['choice'][data['signed_coherence'] == coh] == 1) / np.sum(data['signed_coherence'] == coh))
        elif positive_direction == 'left':
            x_data = np.append(x_data, -coh)
            y_data = np.append(y_data, np.sum(data['choice'][data['signed_coherence'] == coh] == -1) / np.sum(data['signed_coherence'] == coh))
    # sorting
    x_data, y_data = zip(*sorted(zip(x_data, y_data)))
    
    # fit psychometric function
    x_model = np.linspace(-100, 100, 100)
    model = fit_psychometric_function(x_data, y_data)
    y_model = model.predict(x_model)
    return np.asarray(x_data), np.asarray(y_data), model, np.asarray(x_model), np.asarray(y_model)


def get_chronometric_data(data, positive_direction='right'):
    coherences = np.asarray([])
    chrono = np.asarray([])
    for _, coh in enumerate(np.unique(data['signed_coherence'])):
        if positive_direction == 'right':
            coherences = np.append(coherences, coh)
            chrono = np.append(chrono, np.mean(data['response_time'][(data['signed_coherence'] == coh) & (data['outcome'] == 1)]))
        elif positive_direction == 'letf':
            coherences = np.append(coherences, -coh)
            chrono = np.append(chrono, np.mean(data['response_time'][(data['signed_coherence'] == coh) & (data['outcome'] == 1)]))
            
    # sort coherences and chrono by coherence
    coherences, chrono = zip(*sorted(zip(coherences, chrono)))
    return coherences, chrono


def save_model(model: dict, name: str, dir: str = BASE_DIR / "src" / "models/"):    
    # print(dir.resolve().exists())
    # file_dir = Path(dir) if dir else 
    file = Path(dir) / f"{name}.pkl" 
    with open(file, "wb") as f:
        dill.dump(model, f)
        
def load_model(name: str, dir: str = BASE_DIR / "src" / "models/"):
    file = Path(dir) / f"{name}.pkl" 
    with open(file, "rb") as f:
        model = dill.load(f)
    return model