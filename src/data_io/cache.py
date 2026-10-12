"""Storage for fingerprinted caches of expensive fits.

A cache entry is a data file (`.npz` draws or a `.csv` table) plus a JSON sidecar recording what
produced it, including a fingerprint of the exact arrays handed to the fitter. Deciding whether a
sidecar is still valid is the model module's job (`src.utils.mcmc_cache`, `src.utils.robustness`).
This module only hashes, reads and writes, and every write goes through `paths.writable`.
"""

import hashlib
import json

import numpy as np

from src.data_io.paths import writable


def fingerprint(*arrays):
    """Hash of the exact numbers handed to a fitter, so stale caches cannot go unnoticed."""
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()[:16]


def read_sidecar(path):
    """The stored sidecar, or None when there is no entry."""
    return json.loads(path.read_text()) if path.exists() else None


def write_sidecar(path, payload):
    writable(path).write_text(json.dumps(payload, indent=2) + "\n")


def save_arrays(path, **arrays):
    np.savez_compressed(writable(path), **arrays)


def load_arrays(path):
    with np.load(path) as stored:
        return dict(stored)


def save_table(table, path):
    table.to_csv(writable(path), index=False)
