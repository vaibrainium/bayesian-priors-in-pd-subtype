"""Where the data lives, and where code is allowed to write.

The roots come from `config/dir-config.yaml`, with one rule for each:

- `RAW` is read only. Nothing here ever writes to it.
- `COMPILED` and `PROCESSED` hold derived data. The pipeline stages write there, and only through
  `catalog.save` / `writable`.
- `OUTPUTS` (the repo's gitignored `outputs/`) holds analysis results: tables, figures, and caches
  of analysis-only fits.
- `DISSEMINATION` holds the slide decks that are tracked in git.

Every root can be redirected with an environment variable (`PD_RAW_DIR`, `PD_COMPILED_DIR`,
`PD_PROCESSED_DIR`, `PD_OUTPUTS_DIR`). That is how a pipeline change is checked against a scratch
copy before it is allowed near the shared data.
"""

import os
from pathlib import Path

from config import dir_config

REPO = Path(__file__).resolve().parents[2]


def _root(env, default):
    return Path(os.environ.get(env, default)).expanduser()


RAW = _root("PD_RAW_DIR", dir_config.data.raw)
COMPILED = _root("PD_COMPILED_DIR", dir_config.data.compiled)
PROCESSED = _root("PD_PROCESSED_DIR", dir_config.data.processed)
OUTPUTS = _root("PD_OUTPUTS_DIR", REPO / "outputs")
DISSEMINATION = REPO / "dissemination"

ROOTS = {"raw": RAW, "compiled": COMPILED, "processed": PROCESSED, "outputs": OUTPUTS, "dissemination": DISSEMINATION}
WRITABLE_ROOTS = ("compiled", "processed", "outputs", "dissemination")


def _is_under(path, root):
    return path == root or root in path.parents


def writable(path):
    """Resolve `path` for writing, refusing anything outside the writable roots.

    Symlinks are resolved first, so a link inside a writable root that points into `RAW` (or
    anywhere else) is refused rather than written through. The parent directory is created.
    """
    path = Path(path).expanduser().resolve()
    if _is_under(path, RAW.resolve()):
        raise PermissionError(f"{path} is inside the raw data directory, which is read only")
    if not any(_is_under(path, ROOTS[name].resolve()) for name in WRITABLE_ROOTS):
        allowed = ", ".join(str(ROOTS[name]) for name in WRITABLE_ROOTS)
        raise PermissionError(f"{path} is outside the writable roots ({allowed})")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def writable_dir(path):
    """`writable` for a directory: checks it the same way and creates it."""
    path = writable(Path(path) / ".keep").parent
    path.mkdir(exist_ok=True)
    return path
