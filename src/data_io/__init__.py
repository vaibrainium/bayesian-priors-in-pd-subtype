"""File handling for notebooks and scripts: paths, named datasets, and fit caches.

Analysis code takes data in and hands results back through this package; it never builds paths
itself. See `paths` for which directories may be written, and `catalog` for the dataset names.
"""

from src.data_io.catalog import (
    load,
    load_metadata,
    load_trials,
    output_path,
    passing_subjects,
    path,
    save,
    save_arrays,
    save_figure,
    save_metadata_stage,
    save_table,
    save_text,
)

__all__ = [
    "load",
    "load_metadata",
    "load_trials",
    "output_path",
    "passing_subjects",
    "path",
    "save",
    "save_arrays",
    "save_figure",
    "save_metadata_stage",
    "save_table",
    "save_text",
]
