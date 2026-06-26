import io
import pickle
import sys
import types

import cloudpickle.cloudpickle as _cp
import numpy as np
import pandas as pd
import torch


def _code_compat(*args):
    """Reconstruct a code object pickled on Python 3.10 under Python 3.11+.

    Python 3.11 added `qualname` at position 12 and replaced `lnotab` with
    `linetable` + `exceptiontable`, shifting all subsequent arguments.
    """
    if sys.version_info >= (3, 11) and len(args) == 16:
        (argcount, posonlyargcount, kwonlyargcount, nlocals, stacksize, flags, codestring, constants, names, varnames, filename, name, firstlineno, lnotab, freevars, cellvars) = args
        return types.CodeType(
            argcount,
            posonlyargcount,
            kwonlyargcount,
            nlocals,
            stacksize,
            flags,
            codestring,
            constants,
            names,
            varnames,
            filename,
            name,
            name,  # qualname — not stored in 3.10, fall back to name
            firstlineno,
            lnotab,  # linetable — lnotab is close enough for loading purposes
            b"",  # exceptiontable — not present in 3.10
            freevars,
            cellvars,
        )
    return types.CodeType(*args)


def _builtin_type_compat(name):
    """Wrap cloudpickle's _builtin_type to redirect CodeType to _code_compat."""
    if name == "CodeType":
        return _code_compat
    return _cp._builtin_type(name)


def prepare_data(
    behavior_df: pd.DataFrame,
    session_id,
    color,
) -> pd.DataFrame:

    data = behavior_df[(behavior_df["session_id"] == session_id) & (behavior_df["color"] == color)][["rt", "choice", "signed_coherence"]].copy()

    if data.empty:
        raise ValueError(f"No data for session={session_id}, color={color}")

    data["choice"] = data["choice"].astype(int)

    return data


def build_stimulus(data: pd.DataFrame, rt_buffer: float = 1.5, max_seconds: float = 8.0) -> np.ndarray:
    max_rt = float(np.clip(data["rt"].max() * rt_buffer, 0, max_seconds))
    stimulus_length = max(100, int(max_rt * 1000))

    return np.tile(data["signed_coherence"].to_numpy()[:, None], (1, stimulus_length))


def build_grid(behavior_df: pd.DataFrame) -> list[dict]:

    session_ids = np.sort(behavior_df["session_id"].unique())
    colors = np.sort(behavior_df["color"].unique())

    variants = [
        {"enable_leak": False, "enable_time_constant": False, "enable_sv": True, "enable_sz": True},
        {"enable_leak": False, "enable_time_constant": True, "enable_sv": True, "enable_sz": True},
        {"enable_leak": True, "enable_time_constant": False, "enable_sv": True, "enable_sz": True},
        {"enable_leak": True, "enable_time_constant": True, "enable_sv": True, "enable_sz": True},
        {"enable_leak": False, "enable_time_constant": False, "enable_sv": False, "enable_sz": False},
    ]

    return [
        {
            "session_id": session_id,
            "color": color,
            **variant,
        }
        for variant in variants
        for session_id in session_ids
        for color in colors
    ]


def get_job(grid: list[dict], job_id: int) -> dict:

    if job_id >= len(grid):
        raise ValueError(f"job_id {job_id} out of bounds (max={len(grid) - 1})")

    return grid[job_id]


class CPUUnpickler(pickle.Unpickler):
    """Loads pkl files saved on Python 3.10 under Python 3.11+, on CPU."""

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        if module == "cloudpickle.cloudpickle" and name == "_builtin_type":
            return _builtin_type_compat
        return super().find_class(module, name)


def load_model(path):
    with open(path, "rb") as f:
        return CPUUnpickler(f).load()
