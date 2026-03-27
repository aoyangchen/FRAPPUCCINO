"""
nb_contracts.py
===================

This module centralizes the canonical notebook path contracts and
subdirectory definitions for the GT1 Colab workflow.  It exists to
decouple static configuration from the interactive notebook so that
future refactors can import a single source of truth without
redeclaring string literals throughout the code.  These path constants
are *contractual*: downstream evaluation scripts and reporting tools
expect the same directory names and relative locations.  Do **not**
change ``PROJ_NAME`` or the entries in ``SUBDIRS`` without updating all
dependent code.

No runtime side effects (e.g., file system access or Drive mounting) occur
at import time, ensuring that the module can be imported safely in any
environment.

Only static constants and minimal helpers are provided here.  Logic
that touches Google Drive or performs downloads lives in
``nb_drive_io.py``.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Canonical paths and subdirectories
#
# The GT1 Colab project stores all of its working files under a single
# project root.  When running inside Google Colab with Drive mounted, this
# root resolves to ``/content/drive/MyDrive/esi_baseline_gbdt``.  When not
# using Drive, the root falls back to ``/content/esi_baseline_gbdt``.  The
# notebook initialization cell is responsible for determining which root is
# active and binding it to the ``PROJ`` global.  This module exposes the
# canonical Drive‑backed root and the list of subdirectories, but does not
# decide which root to use.

# Name of the project folder as it appears in the user's Drive.  This
# constant is preserved verbatim from the original notebook and must not
# change between phases of the refactor.
PROJ_NAME: str = "esi_baseline_gbdt"

# Mount point used by ``google.colab.drive``.  The notebook mounts the
# user's Drive at this location before constructing the project root.
MOUNT_POINT: str = "/content/drive"

# Canonical Drive‑backed project root.  When running in Colab and using
# Drive, the project root is ``/content/drive/MyDrive/{PROJ_NAME}``.  This
# constant mirrors that path for static reference.  Note that the
# notebook's initialization code still controls whether Drive is used; this
# constant is not automatically used by the notebook.
PROJ_ROOT: Path = Path(MOUNT_POINT) / "MyDrive" / PROJ_NAME

# List of canonical first‑pass subdirectories within the project.  The
# notebook creates these folders during initialization to ensure that
# downstream code can rely on their existence.  Do not modify this list
# without updating all code that depends on these names.
SUBDIRS: list[str] = [
    "data",
    "features",
    "splits",
    "models",
    "metrics",
    "logs",
    "reports",
]

def get_subdir(proj: Path, subdir: str) -> Path:
    """Return a child path within the given project.

    Parameters
    ----------
    proj : Path
        The resolved project root.
    subdir : str
        One of the names from ``SUBDIRS``.

    Returns
    -------
    Path
        ``proj / subdir`` as a ``Path`` instance.
    """
    return Path(proj) / subdir

def ensure_subdirs(proj: Path, subdirs: list[str] | None = None) -> None:
    """Ensure that the canonical subdirectories exist under ``proj``.

    This helper mirrors the behavior of the original notebook, which
    iterates over the canonical subdirectory names and creates each
    directory with ``parents=True`` and ``exist_ok=True``.  It is
    intentionally side‑effectful and should only be called when the caller
    has determined that the file system should be mutated (e.g., during
    notebook initialization).

    Parameters
    ----------
    proj : Path
        The project root under which to create subdirectories.
    subdirs : list[str] | None, optional
        A custom list of subdirectory names.  If ``None``, the default
        ``SUBDIRS`` is used.
    """
    names = subdirs or SUBDIRS
    for name in names:
        (Path(proj) / name).mkdir(parents=True, exist_ok=True)

__all__ = [
    "PROJ_NAME",
    "MOUNT_POINT",
    "PROJ_ROOT",
    "SUBDIRS",
    "get_subdir",
    "ensure_subdirs",
]